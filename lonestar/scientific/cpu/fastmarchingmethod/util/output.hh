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

template <typename Graph>
void DumpToNpy(Graph& graph) {
  std::vector<float> data(NUM_CELLS);
  for (auto it = graph.begin(); it != graph.end(); it++) {
    if (*it >= NUM_CELLS)
      break;
    auto& node_data = graph.getData(*it);
    data[*it]       = node_data.solution;
  }
  // save it to file
  cnpy::npy_save(output_npy, &data[0], {nx, ny}, "w");
}

template <typename Graph>
void DumpToCsv(Graph& graph) {
  std::ofstream ofs;
  ofs.open(output_csv);
  unsigned row_counter = 0;
  for (auto i = graph.begin(); i != graph.end(); i++) {
    if (*i >= NUM_CELLS)
      break;
    if (row_counter != 0)
      ofs << ',';
    auto& node_data = graph.getData(*i);
    ofs << node_data.solution;
    if (++row_counter == nx) {
      ofs << '\n';
      row_counter = 0;
    }
  }
  ofs.close();
}

void segy_to_csv(segystack::SegyFile segy, std::string csv = "marmousi.csv") {
  std::ofstream ofs;
  ofs.open(csv);

  int total_traces = segy.getBinaryHeader().getValueAtOffset<uint16_t>(13);
  int read_traces  = 0;
  segystack::SegyFile::Trace trace;
  while (segy.read(trace)) {
    for (auto i = trace.samples().begin(); i != trace.samples().end(); i++) {
      if (i != trace.samples().begin())
        ofs << ' ';
      ofs << *i;
    }
    read_traces++;
    if (read_traces >= total_traces)
      break;
    else
      ofs << std::endl;
    segy.seek(read_traces);
  }

  galois::gDebug("Read ", read_traces, " traces.");
  ofs.close();
}
