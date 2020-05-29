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
#include "fastmarchingmethod.h"

template<>
double BoundaryCondition(double2d_t&& coords) {
  [[maybe_unused]] const auto& [x, y] = coords;

  return 0.;
}

template<>
double BoundaryCondition(double3d_t&& coords) {
  [[maybe_unused]] const auto& [x, y, z] = coords;

  return 0.;
}

template<>
bool NonNegativeRegion(double3d_t&& coords) {
  [[maybe_unused]] const auto& [x, y, z] = coords;

  // Example 1: a spherical interface of radius 0.25 centered at the origin
  // return x * x + y * y + z * z >= .25 * .25;

  // Example 2: a plane past through the origin
  return 100. * x + y + 2. * z >= 0.;
}

// TODO scatter boundary
