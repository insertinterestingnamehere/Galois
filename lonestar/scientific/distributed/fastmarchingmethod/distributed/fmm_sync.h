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
#ifdef GALOIS_FMM_DIST_PULL
GALOIS_SYNC_STRUCTURE_REDUCE_MIN(solution, double);
GALOIS_SYNC_STRUCTURE_BITSET(solution);
#endif

#ifdef GALOIS_FMM_DIST_PUSH
// TODO add to AtomicHelpers.h
namespace galois {
template <typename Ty>
bool minArray(Ty& a_arr, const Ty& b_arr) {
  bool ret = false;
  for (unsigned i = 0; i < a_arr.size(); ++i) {
    ret |= (b_arr[i] < galois::min(a_arr[i], b_arr[i]));
  }
  return ret;
}
} // namespace galois
/**
 * Creates a Galois reduction sync structure that does a pairwise min reduction
 * on an array.
 */
#ifdef GALOIS_ENABLE_GPU
// TODO GPU code included
#else
// Non-GPU code
#define GALOIS_SYNC_STRUCTURE_REDUCE_PARE_WISE_MIN(fieldname, fieldtype)       \
  struct Reduce_pair_wise_min_##fieldname {                                    \
    GALOIS_DECL_NON_GPU_SYNC_STRUCTURE;                                        \
    typedef fieldtype ValTy;                                                   \
                                                                               \
    template <typename NodeData>                                               \
    static ValTy extract(uint32_t, const NodeData& node) {                     \
      return node.fieldname;                                                   \
    }                                                                          \
                                                                               \
    template <typename NodeData>                                               \
    static bool reduce(uint32_t, NodeData& node, ValTy y) {                    \
      return minArray(node.fieldname, y);                                      \
    }                                                                          \
                                                                               \
    template <typename NodeData>                                               \
    static void reset(uint32_t, NodeData&) {}                                  \
                                                                               \
    template <typename NodeData>                                               \
    static void setVal(uint32_t, NodeData& node, ValTy y) {                    \
      node.fieldname = y;                                                      \
    }                                                                          \
  }
#endif
GALOIS_SYNC_STRUCTURE_REDUCE_PARE_WISE_MIN(upwind_solution, sync_double3d_t);
GALOIS_SYNC_STRUCTURE_BITSET(upwind_solution);
#endif
