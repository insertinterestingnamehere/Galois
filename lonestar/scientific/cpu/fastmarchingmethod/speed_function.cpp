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

template <>
double SpeedFunction(double2d_t&& coords) {
  [[maybe_unused]] const auto& [x, y] = coords;

  return 1.;
  // return 1. + .50 * std::sin(20. * PI * x) * std::sin(20. * PI * y);
  // return 1. - .99 * std::sin(2. * PI * x) * std::sin(2. * PI * y);
}

template <>
double SpeedFunction(double3d_t&& coords) {
  [[maybe_unused]] const auto& [x, y, z] = coords;

  return 1.;
  // return 1. + .50 * std::sin(20. * PI * x) * std::sin(20. * PI * y) *
  // std::sin(20. * PI * z); return 1. - .99 * std::sin(2. * PI * x) *
  // std::sin(2. * PI * y) * std::sin(2. * PI * z);
}
