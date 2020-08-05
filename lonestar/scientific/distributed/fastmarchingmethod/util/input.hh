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
// #include "segy/SEG-YReader.h"
#include "google-segystack/segy_file.h"
#include "numpy/cnpy.h"
void SetKnobs(const std::vector<std::size_t>& d) {
  assert(d.size() == DIM_LIMIT);
  nx = d[0];
  ny = d[1];

  // metric inference
  NUM_CELLS = nx * ny;
  if (intervals.empty()) {
    dx = (xb - xa) / data_t(nx + 1);
    dy = (yb - ya) / data_t(ny + 1);
  } else {
    dx = intervals[0];
    dy = intervals[1];
    xa = -dx;
    xb = dx * data_t(nx);
    ya = -dy;
    yb = dy * data_t(ny);
  }

  if (!RF)
    RF = 1 / std::min({dx, dy, 1.}, std::less<data_t>{});

  DGDEBUG(nx, " - ", ny);
  DGDEBUG(dx, " - ", dy);
  DGDEBUG("RF: ", RF);
}

void SetupGrids(Graph& graph) {
  if (!input_segy.empty()) {
    std::unique_ptr<segystack::SegyFile> segy_ptr;
    segy_ptr      = std::make_unique<segystack::SegyFile>(input_segy);
    std::size_t x = segy_ptr->getBinaryHeader().getValueAtOffset<uint16_t>(13);
    std::size_t y = segy_ptr->getBinaryHeader().getValueAtOffset<uint16_t>(21);

    xa = -.625 * data_t(x + 1);
    xb = .625 * data_t(x + 1);
    ya = -.625 * data_t(y + 1);
    yb = .625 * data_t(y + 1);
    assert(nx == x && ny == y);

    // ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

    segystack::SegyFile::Trace trace;

    std::size_t read_traces = 0;
    segy_ptr->seek(read_traces);
    segy_ptr->read(trace);
    for (GNode gid = 0; gid < NUM_CELLS; gid++) {
      if (!graph.isLocal(gid))
        continue;
      GNode node    = graph.getLID(gid);
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      // curData.solution = INF; // TODO ghost init?

      auto [i, j] = id2ij(graph.getGID(node));
      if (i != read_traces) {
        // assert(i == read_traces + 1); // not necessarily true for dist.
        segy_ptr->seek(++read_traces);
        segy_ptr->read(trace);
      }
      curData.speed = trace.samples()[j] * .001;
    }
    // assert(read_traces + 1 == x); // not necessarily true for dist.
    galois::gDebug("Read ", read_traces, " traces.");
  } else if (!input_npy.empty()) {
    cnpy::NpyArray npy = cnpy::npy_load(input_npy);
    float* npy_data    = npy.data<float>();

    // make sure the loaded data matches the saved data
    assert(npy.word_size == sizeof(float) && "wrong data type");
    assert(npy.shape.size() == 2 && "Data map should be 2-D");

#if 0 // symmetric, for marmousi
      xa = -.625 * data_t(x + 1);
      xb = .625 * data_t(x + 1);
      ya = -.625 * data_t(y + 1);
      yb = .625 * data_t(y + 1);
#endif
#if 0 // asymmetric, for marmousi
      xa = -1.25;
      xb = 1.25 * data_t(x);
      ya = -1.25;
      yb = 1.25 * data_t(y);
#endif
    assert(nx == npy.shape[0] && ny == npy.shape[1]);

    // ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

    for (GNode gid = 0; gid < NUM_CELLS; gid++) {
      if (!graph.isLocal(gid))
        continue;
      GNode node    = graph.getLID(gid);
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      // curData.solution = INF; // TODO ghost init?
      curData.speed = npy_data[node] * .001;
    }
  } else if (!input_csv.empty()) {
    std::ifstream incsv(input_csv);
    std::string line;
    if (dims.empty())
      GALOIS_DIE("Please specify the dimentions of the csv data (use -d)");
    std::size_t x = dims[0], y = dims[1];
    //// CSV WITH HEADER
    // if (std::getline(incsv, line)) {
    //   const char* header = line.c_str();
    //   char* end;
    //   x = strtoul(header, &end, 0);
    //   assert(*end == ' '); // Does it matter?
    //   y = strtoul(end + 1, &end, 0);
    //   assert(*end == '\0'); // Does it matter?
    // }

    xa = -.625 * data_t(x + 1);
    xb = .625 * data_t(x + 1);
    ya = -.625 * data_t(y + 1);
    yb = .625 * data_t(y + 1);
    assert(nx == x && ny == y);

    // ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});
    std::size_t gid = 0;
    while (std::getline(incsv, line)) {
      const char* beg = line.c_str();
      char* end;
      bool first_time = true;
      do {
        double d = strtod(first_time ? beg : end + 1, &end);
        if (first_time)
          first_time = false;
        if (!d)
          GALOIS_DIE("In-line csv parsing failed.");
        if (!graph.isLocal(gid))
          continue;
        GNode node    = graph.getLID(gid);
        auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        // curData.solution = INF; // TODO ghost init?
        curData.speed = d * .001;
        ++gid;
      } while (*end == ' ');
      if (*end != '\0')
        GALOIS_DIE("Bad EOL.");
    }
  } else {
    if (!dims.empty()) {
      dims.resize(DIM_LIMIT, 1); // padding
      // SetKnobs(dims);
    } else
      GALOIS_DIE("Undefined dimensions. See help with -h.");

    // ConstructCsrGrids(graph, std::array<std::size_t, 2>{nx, ny});

    const auto& all_nodes = graph.allNodesRange();
    galois::do_all(
        galois::iterate(all_nodes.begin(), all_nodes.end()),
        [&](auto node) noexcept {
          auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
          curData.speed = SpeedFunction(id2xy(graph.getGID(node)));
          // curData.solution = INF; // TODO ghost init?
        },
        galois::no_stats(),
        galois::loopname(
            syncSubstrate->get_run_identifier("initializeCells").c_str()));
  }
}
