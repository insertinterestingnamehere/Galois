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
#ifdef FMM_3D
/**
 * Convert node ID into grid coordinate tuple.
 */
template <typename GNode>
size3d_t id2ijk(const GNode node) {
  if (node >= NUM_CELLS)
    return {};
  auto res      = std::lldiv(node, ny * nz);
  std::size_t i = res.quot;
  res           = std::lldiv(res.rem, nz);
  std::size_t j = res.quot;
  std::size_t k = res.rem;

  assert(i < nx && j < ny && k < nz);
  return std::array<std::size_t, 3>({i, j, k});
}
/**
 * Convert node ID into coordinate tuple.
 * `id = i * nx * ny + j * ny + k`
 * `(i, j, k) = (z, x, y)`
 * @param node Node ID
 * @return Coordinate tuple (x, y, z)
 */
template <typename GNode>
data3d_t id2xyz(const GNode node) {
  if (node >= NUM_CELLS)
    return {};
  auto res = std::lldiv(node, ny * nz);
  double i = res.quot;
  res      = std::lldiv(res.rem, nz);
  double j = res.quot;
  double k = res.rem;

  double x = xa + dx * (i + 1);
  double y = ya + dy * (j + 1);
  double z = za + dz * (k + 1);
  // galois::gDebug(i, " ", j, " ", k);
  // galois::gDebug(dx, " - ", dy, " - ", dz);
  assert(x < xb && y < yb && z < zb);
  return std::array<double, 3>({x, y, z});
}

/**
 * Convert coordinate tuple into node ID.
 * See id2xyz
 */
template <typename GNode>
GNode xyz2id(const data3d_t& coords) {
  uint64_t i = std::round((coords[2] - za) / dz - 1.);
  uint64_t j = std::round((coords[0] - xa) / dx - 1.);
  uint64_t k = std::round((coords[1] - ya) / dy - 1.);

  return i * ny * nz + j * nz + k;
}
#endif

/**
 * 2D
 */
size2d_t id2ij(const std::size_t node) {
  if (node >= NUM_CELLS)
    return {};
  auto res      = std::lldiv(node, nx);
  std::size_t i = res.quot;
  std::size_t j = res.rem;

  assert(i < ny && j < nx);
  return std::array<std::size_t, 2>({i, j});
}

inline std::size_t ij2id(const size2d_t ij) {
  auto [i, j] = ij;
  return i * nx + j;
}

data2d_t id2xy(const std::size_t node) {
  if (node >= NUM_CELLS)
    return {};
  auto res = std::lldiv(node, nx);
  double i = res.quot;
  double j = res.rem;

  double y = ya + dy * i;
  double x = xa + dx * j;
  // galois::gDebug(i, " ", j);
  // galois::gDebug(dx, " - ", dy);
  assert(x <= xb && y <= yb);
  return data2d_t({x, y});
}

std::size_t xy2id(const data2d_t& coords) {
  std::size_t i = std::round((coords[0] - ya) / dy);
  std::size_t j = std::round((coords[1] - xa) / dx);

  return i * nx + j;
}
