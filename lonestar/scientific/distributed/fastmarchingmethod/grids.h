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

// Routine to initialize graph topology and face normals.
template <typename Graph>
auto generate_grid(Graph& built_graph, std::size_t nx, std::size_t ny,
                   std::size_t nz) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
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
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(0);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  // This condition isn't true in irregular meshes,
  // but that'd be a separate mesh generation routine.
  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
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
  temp_graph.phase2();
  std::size_t xy_low_face_start  = num_cells;
  std::size_t xy_high_face_start = xy_low_face_start + nx * ny;
  std::size_t yz_low_face_start  = xy_high_face_start + nx * ny;
  std::size_t yz_high_face_start = yz_low_face_start + ny * nz;
  std::size_t xz_low_face_start  = yz_high_face_start + ny * nz;
  std::size_t xz_high_face_start = xz_low_face_start + nx * nz;
  assert(("Error in logic for dividing up node ids for exterior faces.",
          num_nodes == xz_high_face_start + nx * nz));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        // (i, j, k) = (z, x, y)
        std::size_t id = i * nx * ny + j * ny + k;
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

  // TODO: is it possible to set the edge data
  // during construction without copying here?
  temp_graph.finish<typename Graph::edge_data_type>();
  //auto* rawEdgeData = temp_graph.finish<Graph::edge_data_type>();
  //std::uninitialized_copy(std::make_move_iterator(edge_data.begin()),
  //                        std::make_move_iterator(edge_data.end()),
  //                        rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
  return std::make_tuple(num_nodes, num_cells, num_outer_faces,
                         xy_low_face_start, xy_high_face_start,
                         yz_low_face_start, yz_high_face_start,
                         xz_low_face_start, xz_high_face_start);
}

/** Utils **/

/**
 * Convert node ID into grid coordinate tuple.
 */
template <typename GNode>
std::array<std::size_t, 3> getPos(const GNode node) {
  if (node >= NUM_CELLS) return {};
  auto res = std::lldiv(node, nx * ny);
  std::size_t i = res.quot;
  res = std::lldiv(res.rem, ny);
  std::size_t j = res.quot;
  std::size_t k = res.rem;

  assert(i < nz && j < nx && k < ny);
  return std::array<std::size_t, 3>({j, k, i});
}
/**
 * Convert node ID into coordinate tuple.
 * `id = i * nx * ny + j * ny + k`
 * `(i, j, k) = (z, x, y)`
 * @param node Node ID
 * @return Coordinate tuple (x, y, z)
 */
template <typename GNode>
std::array<CoordTy, 3> getCoord(const GNode node) {
  if (node >= NUM_CELLS) return {};
  auto res = std::lldiv(node, nx * ny);
  CoordTy i = res.quot;
  res = std::lldiv(res.rem, ny);
  CoordTy j = res.quot;
  CoordTy k = res.rem;

  CoordTy x = xa + dx * (j + 1);
  CoordTy y = ya + dy * (k + 1);
  CoordTy z = za + dz * (i + 1);
  // galois::gDebug(i, " ", j, " ", k);
  // galois::gDebug(dx, " - ", dy, " - ", dz);
  assert(x < xb && y < yb && z < zb);
  return std::array<CoordTy, 3>({x, y, z});
}

/**
 * Convert coordinate tuple into node ID.
 * See getCoord
 */
template <typename GNode>
GNode getNodeID(const std::array<CoordTy, 3>& coords) {
  uint64_t i = std::round((coords[2] - za) / dz - 1.);
  uint64_t j = std::round((coords[0] - xa) / dx - 1.);
  uint64_t k = std::round((coords[1] - ya) / dy - 1.);

  return i * nx * ny + j * ny + k;
}
