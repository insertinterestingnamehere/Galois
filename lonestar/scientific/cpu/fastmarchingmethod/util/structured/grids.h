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
#include "galois/graphs/ReadGraph.h"
#include "galois/graphs/FileGraph.h"
#include "galois/LargeArray.h"

// Only support 2-D or 3-D for now
template <typename EdgeTy, std::size_t D>
static auto WriteCsrGrids(galois::graphs::FileGraphWriter& temp_graph,
                          const std::array<std::size_t, D>& dims) = delete;

template <typename EdgeTy, bool PADDING = true>
static auto WriteCsrGrids(galois::graphs::FileGraphWriter& temp_graph,
                          const std::array<std::size_t, 2>& dims) {
  auto [nx, ny] = dims;

  // Each node represents a grid cell.
  // Each edge represents a face.
  // Ghost nodes are added to represent the exterior
  // of the domain on the other side of each face.
  // This is for boundary condition handling.
  std::size_t num_outer_faces(0);
  std::size_t num_cells = nx * ny;
  std::size_t num_nodes = num_cells;
  std::size_t num_edges = 4 * nx * ny;
  if constexpr (PADDING) {
    num_outer_faces = (nx + ny) * 2; // padding
    num_nodes += num_outer_faces;
    num_edges += num_outer_faces;
  }
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges<EdgeTy>(num_edges);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  // This condition isn't true in irregular meshes,
  // but that'd be a separate mesh generation routine.
  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      std::size_t id = i * ny + j;
      for (std::size_t l = 0; l < 4; l++) {
        temp_graph.incrementDegree(id);
      }
    }
  }

  // Set the degrees for all the ghost cells to 1.
  // No ghost cell should share a boundary with more
  // than one actual cell in the domain.
  if constexpr (PADDING)
    for (std::size_t id = num_cells; id < num_nodes; id++) {
      temp_graph.incrementDegree(id);
    }

  // Now that the degree of each node is known,
  // fill in the actual topology.
  // Also fill in the node data with the vector
  // normal to the face, going out from the current cell.
  temp_graph.phase2();
  // padding
  std::size_t x_low_face_start(0), x_high_face_start(0), y_low_face_start(0),
      y_high_face_start(0);
  if constexpr (PADDING) {
    x_low_face_start  = num_cells;
    x_high_face_start = x_low_face_start + nx;
    y_low_face_start  = x_high_face_start + nx;
    y_high_face_start = y_low_face_start + ny;
    assert(("Error in logic for dividing up node ids for exterior faces." &&
            num_nodes == y_high_face_start + ny));
  }
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      std::size_t id = i * ny + j;
      if (i > 0) {
        temp_graph.addNeighbor(id, id - ny);
      } else {
        if constexpr (PADDING) {
          std::size_t ghost_id = y_low_face_start + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        } else {
          temp_graph.addNeighbor(id, id); // dummy edge
        }
      }
      if (i < nx - 1) {
        temp_graph.addNeighbor(id, id + ny);
      } else {
        if constexpr (PADDING) {
          std::size_t ghost_id = y_high_face_start + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        } else {
          temp_graph.addNeighbor(id, id); // dummy edge
        }
      }
      if (j > 0) {
        temp_graph.addNeighbor(id, id - 1);
      } else {
        if constexpr (PADDING) {
          std::size_t ghost_id = x_low_face_start + i;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        } else {
          temp_graph.addNeighbor(id, id); // dummy edge
        }
      }
      if (j < ny - 1) {
        temp_graph.addNeighbor(id, id + 1);
      } else {
        if constexpr (PADDING) {
          std::size_t ghost_id = x_high_face_start + i;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        } else {
          temp_graph.addNeighbor(id, id); // dummy edge
        }
      }
    }
  }

  temp_graph.finish();

  return std::make_tuple(num_nodes, num_cells, num_outer_faces,
                         x_low_face_start, x_high_face_start, y_low_face_start,
                         y_high_face_start);
}

template <typename EdgeTy>
static auto WriteCsrGrids(galois::graphs::FileGraphWriter& temp_graph,
                          const std::array<std::size_t, 3>& dims) {
  auto [nx, ny, nz] = dims;
  galois::gDebug("Constructing CSR grids in shape (", nx, ", ", ny, ", ", nz,
                 ")");

  // Each node represents a grid cell.
  // Each edge represents a face.
  // Ghost nodes are added to represent the exterior
  // of the domain on the other side of each face.
  // This is for boundary condition handling.
  std::size_t num_outer_faces = (nx * ny + ny * nz + nx * nz) * 2;
  std::size_t num_cells       = nx * ny * nz;
  std::size_t num_nodes       = num_cells + num_outer_faces;
  std::size_t num_edges       = 6 * nx * ny * nz + num_outer_faces;
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges<EdgeTy>(num_edges);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  // This condition isn't true in irregular meshes,
  // but that'd be a separate mesh generation routine.
  galois::gDebug("phase1");
  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * ny * nz + j * nz + k;
        for (std::size_t l = 0; l < 6; l++) {
          temp_graph.incrementDegree(id);
        }
      }
    }
  }
  // Set the degrees for all the ghost cells to 1.
  // No ghost cell should share a boundary with more
  // than one actual cell in the domain.
  for (std::size_t id = num_cells; id < num_nodes; id++) {
    temp_graph.incrementDegree(id);
  }

  // Now that the degree of each node is known,
  // fill in the actual topology.
  // Also fill in the node data with the vector
  // normal to the face, going out from the current cell.
  galois::gDebug("phase2");
  temp_graph.phase2();
  std::size_t xy_low_face_start  = num_cells;
  std::size_t xy_high_face_start = xy_low_face_start + nx * ny;
  std::size_t yz_low_face_start  = xy_high_face_start + nx * ny;
  std::size_t yz_high_face_start = yz_low_face_start + ny * nz;
  std::size_t xz_low_face_start  = yz_high_face_start + ny * nz;
  std::size_t xz_high_face_start = xz_low_face_start + nx * nz;
  assert(("Error in logic for dividing up node ids for exterior faces." &&
          num_nodes == xz_high_face_start + nx * nz));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * ny * nz + j * nz + k;
        if (i > 0) {
          temp_graph.addNeighbor(id, id - ny * nz);
        } else {
          std::size_t ghost_id = yz_low_face_start + j * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (i < nx - 1) {
          temp_graph.addNeighbor(id, id + ny * nz);
        } else {
          std::size_t ghost_id = yz_high_face_start + j * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (j > 0) {
          temp_graph.addNeighbor(id, id - nz);
        } else {
          std::size_t ghost_id = xz_low_face_start + i * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (j < ny - 1) {
          temp_graph.addNeighbor(id, id + nz);
        } else {
          std::size_t ghost_id = xz_high_face_start + i * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (k > 0) {
          temp_graph.addNeighbor(id, id - 1);
        } else {
          std::size_t ghost_id = xy_low_face_start + i * ny + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (k < nz - 1) {
          temp_graph.addNeighbor(id, id + 1);
        } else {
          std::size_t ghost_id = xy_high_face_start + i * ny + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
      }
    }
  }

  galois::gDebug("finish");
  temp_graph.finish();

  return std::make_tuple(num_nodes, num_cells, num_outer_faces,
                         xy_low_face_start, xy_high_face_start,
                         yz_low_face_start, yz_high_face_start,
                         xz_low_face_start, xz_high_face_start);
}

// Routine to initialize graph topology and face normals.
template <typename Graph, std::size_t D>
auto ConstructCsrGrids(Graph& built_graph,
                       std::array<std::size_t, D> dims) noexcept {
  static_assert(D == 2 || D == 3, "Only support 2-D or 3-D for now.");
  galois::graphs::FileGraphWriter temp_graph;
  auto params = WriteCsrGrids<typename Graph::edge_data_type>(temp_graph, dims);
  // temp_graph.toFile("/net/ohm/export/iss/inputs/synth_grids/"
  //                   "p_wave_velocity_x4.gr");
  galois::graphs::readGraph(built_graph, temp_graph);
  return params;
}

// Generate and save to file.
template <typename edge_data_type>
void ConstructCsrGrids(std::string filename, std::size_t nx, std::size_t ny,
                       std::size_t nz) noexcept {
  galois::graphs::FileGraphWriter temp_graph;
  WriteCsrGrids<edge_data_type>(temp_graph,
                                std::array<std::size_t, 3>{nx, ny, nz});
  temp_graph.toFile(filename);
}
