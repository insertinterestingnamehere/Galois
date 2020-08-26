/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2020, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */
#include "structured/grids.h"
#include "structured/utils.h"

void SetKnobs(const std::vector<std::size_t>& d) {
  assert(d.size() == DIM_LIMIT);
  nx = d[0];
  ny = d[1];
  // metric inference
  NUM_CELLS = nx * ny;

  dx = steps[0];
  dy = steps[1];

  xa = domain_start[0];
  ya = domain_start[1];
  xb = xa + dx * data_t(nx - 1);
  yb = ya + dy * data_t(ny - 1);

  if (!RF)
    RF = 1 / std::min({dx, dy, 1.}, std::less<data_t>{});

  galois::gPrint("Domain shape: ", nx, " x ", ny, "\n");
  galois::gPrint("Unit size: ", dx, " x ", dy, "\n");
  galois::gPrint("Space range: [", xa, ", ", xb, "] x [", ya, ", ", yb, "]\n");
  galois::gPrint("RF: ", RF, "\n");
}

#include "google-segystack/segy_file.h"
void fromSegy(Graph& graph) {
  std::unique_ptr<segystack::SegyFile> segy_ptr;
  segy_ptr = std::make_unique<segystack::SegyFile>(input_segy);
  // Obtain shape info
  std::size_t x = segy_ptr->getBinaryHeader().getValueAtOffset<uint16_t>(13);
  std::size_t y = segy_ptr->getBinaryHeader().getValueAtOffset<uint16_t>(21);

  SetKnobs({x, y});

  ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

  // Initialize graph
  segystack::SegyFile::Trace trace;
  std::size_t read_traces = 0;
  segy_ptr->seek(read_traces);
  segy_ptr->read(trace);
  for (GNode node : graph) {
    auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
    curData.solution = INF;
    if (node >= NUM_CELLS)
      continue;

    auto [i, j] = id2ij(node);
    if (i != read_traces) {
      assert(i == read_traces + 1);
      segy_ptr->seek(++read_traces);
      segy_ptr->read(trace);
    }
    curData.speed = trace.samples()[j] * speed_factor;
  }
  assert(read_traces + 1 == x);
  galois::gDebug("Read ", read_traces, " traces.");
}

#include "numpy/cnpy.h"
void fromNpy(Graph& graph) {
  cnpy::NpyArray npy = cnpy::npy_load(input_npy);
  float* npy_data    = npy.data<float>();
  // make sure the loaded data matches the saved data
  GALOIS_ASSERT(npy.word_size == sizeof(float) && "wrong data type");
  assert(npy.shape.size() == 2 && "Data map should be 2-D");
  // Obtain shape info
  std::size_t x = npy.shape[0], y = npy.shape[1];

  SetKnobs({x, y});

  ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

  // Initialize graph
  for (auto node : graph) {
    auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
    curData.solution = INF;
    if (node >= NUM_CELLS)
      continue;
    curData.speed = npy_data[node] * speed_factor;
  }
}

void fromCsv(Graph& graph) {
  std::ifstream incsv(input_csv);
  std::string line;
  // Obtain shape info
  if (domain_shape.empty())
    GALOIS_DIE("Please specify the dimentions of the csv data");
  std::size_t x = domain_shape[0], y = domain_shape[1];
  //// CSV WITH HEADER
  // if (std::getline(incsv, line)) {
  //   const char* header = line.c_str();
  //   char* end;
  //   x = strtoul(header, &end, 0);
  //   assert(*end == ' '); // Does it matter?
  //   y = strtoul(end + 1, &end, 0);
  //   assert(*end == '\0'); // Does it matter?
  // }

  SetKnobs({x, y});

  ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

  // Initialize graph
  auto ni = graph.begin();
  while (std::getline(incsv, line)) {
    const char* beg = line.c_str();
    char* end;
    bool first_time = true;
    do {
      data_t d = strtod(first_time ? beg : end + 1, &end);
      if (first_time)
        first_time = false;
      if (!d)
        GALOIS_DIE("In-line csv parsing failed.");
      {
        auto& curData    = graph.getData(*ni, galois::MethodFlag::UNPROTECTED);
        curData.solution = INF;
        curData.speed    = d * speed_factor;
      }
      ++ni;
    } while (ni != graph.end() && *end == ' ');
    if (ni == graph.end())
      break;
    if (*end != '\0')
      GALOIS_DIE("Bad EOL.");
  }
  galois::gDebug(std::distance(graph.begin(), ni), "/", NUM_CELLS,
                 " cells initialized from csv");
  // Initialize remaining nodes
  galois::do_all(
      galois::iterate(ni, graph.end()),
      [&](auto node) noexcept {
        auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        curData.solution = INF;
        if (node >= NUM_CELLS)
          return;
        curData.speed = SpeedFunction(id2xy(node));
      },
      galois::no_stats(), galois::loopname("initializeRemainingGrids"));
}

void fromShape(Graph& graph) {
  domain_shape.resize(DIM_LIMIT, 1); // enlarge
  SetKnobs(domain_shape);
  ConstructCsrGrids(graph, std::array<std::size_t, 2>{nx, ny});

  // Initialize graph
  galois::do_all(
      galois::iterate(graph.begin(), graph.end()),
      [&](auto node) noexcept {
        auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        curData.solution = INF;
        if (node >= NUM_CELLS)
          return;
        curData.speed = SpeedFunction(id2xy(node)) * speed_factor;
      },
      galois::no_stats(), galois::loopname("initializeGrids"));
}

void setupGrids(Graph& graph) {
  if (!input_segy.empty()) {
    galois::gPrint("Reading from ", input_segy, "\n");
    fromSegy(graph);
  } else if (!input_npy.empty()) {
    galois::gPrint("Reading from ", input_npy, "\n");
    fromNpy(graph);
  } else if (!input_csv.empty()) {
    galois::gPrint("Reading from ", input_csv, "\n");
    fromCsv(graph);
  } else if (!domain_shape.empty()) {
    galois::gPrint("Building grids\n");
    fromShape(graph);
  } else
    GALOIS_DIE("Unknown domain shape. See help with -h.");

  galois::gPrint("Graph construction complete.\n");
}
