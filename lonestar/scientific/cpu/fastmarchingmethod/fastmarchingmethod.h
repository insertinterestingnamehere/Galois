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
#pragma GCC diagnostic ignored "-Wunused-variable"

using data_t = double;
#include <limits>
constexpr data_t INF = std::numeric_limits<data_t>::infinity();

#include <array>
using data2d_t = std::array<data_t, 2>;
using data3d_t = std::array<data_t, 3>;
using size2d_t = std::array<std::size_t, 2>;
using size3d_t = std::array<std::size_t, 3>;

// Idk why this hasn't been standardized in C++ yet, but here it is.
static constexpr double PI =
    3.1415926535897932384626433832795028841971693993751;

template <typename Tuple>
double SpeedFunction(Tuple&& coords = {});

template <typename Tuple>
double BoundaryCondition(Tuple&& coords = {});

template <typename Tuple>
bool NonNegativeRegion(Tuple&& coords = {});

/**
 * Wrap heap with push(...) and pop() interfaces.
 */
template <typename HeapTy>
struct FMMHeapWrapper {
  typedef typename HeapTy::key_type key_type;
  typedef typename HeapTy::value_type value_type;
  HeapTy wl;

  inline auto begin() { return wl.begin(); }
  inline auto end() { return wl.end(); }

  inline bool empty() { return wl.empty(); }
  inline void clear() { wl.clear(); }

  void push(const value_type& p, const key_type& old_sln = 0) {
    auto [sln_temp, dst] = p;
    auto iter            = wl.lower_bound(old_sln);
    for (; iter != wl.end(); std::advance(iter, 1)) {
      if (iter->second == dst) {
        break;
      }
      if (iter->first != old_sln) {
        iter = wl.end();
        break;
      }
    }
    if (iter == wl.end()) {
      wl.insert({sln_temp, dst});
    } else {
      auto nh  = wl.extract(iter); // node handle
      nh.key() = sln_temp;
      wl.insert(std::move(nh));
    }
  }

  value_type pop() {
    auto pair = *(wl.begin());
    wl.erase(wl.begin()); // TODO serial only
    return pair;
  }
};

/**
 * Unify push w/ or w/o hints.
 */
struct PushWrap {
  // Generic template: ignore redundant arguments
  template <typename C, typename T = typename C::value_type, typename... Args>
  void operator()(C& cont, T&& item, Args&&...) {
    cont.push(std::forward<T>(item));
  }

  // Template for heaps: hint provided for faster insertion
  template <typename H, typename W = FMMHeapWrapper<H>,
            typename T = typename W::value_type,
            typename K = typename W::key_type>
  void operator()(W& cont, T&& item, K&& hint) {
    cont.push(std::forward<T>(item), std::forward<K>(hint));
  }
};
