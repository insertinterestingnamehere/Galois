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
/**
 * Convert node ID into grid coordinate tuple.
 */
template <typename GNode>
size3d_t getPos(const GNode node) {
  if (node >= NUM_CELLS)
    return {};
  auto res      = std::lldiv(node, nx * ny);
  std::size_t i = res.quot;
  res           = std::lldiv(res.rem, ny);
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
double3d_t getCoord(const GNode node) {
  if (node >= NUM_CELLS)
    return {};
  auto res = std::lldiv(node, nx * ny);
  double i = res.quot;
  res      = std::lldiv(res.rem, ny);
  double j = res.quot;
  double k = res.rem;

  double x = xa + dx * (j + 1);
  double y = ya + dy * (k + 1);
  double z = za + dz * (i + 1);
  // galois::gDebug(i, " ", j, " ", k);
  // galois::gDebug(dx, " - ", dy, " - ", dz);
  assert(x < xb && y < yb && z < zb);
  return std::array<double, 3>({x, y, z});
}

/**
 * Convert coordinate tuple into node ID.
 * See getCoord
 */
template <typename GNode>
GNode getNodeID(const double3d_t& coords) {
  uint64_t i = std::round((coords[2] - za) / dz - 1.);
  uint64_t j = std::round((coords[0] - xa) / dx - 1.);
  uint64_t k = std::round((coords[1] - ya) / dy - 1.);

  return i * nx * ny + j * ny + k;
}
