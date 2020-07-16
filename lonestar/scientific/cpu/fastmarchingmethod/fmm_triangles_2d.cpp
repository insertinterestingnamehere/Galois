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

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>

// Silence erroneous warnings from within Boost headers
// that show up with gcc 8.1.
// #pragma GCC diagnostic ignored "-Wparentheses"
// This warning triggers with the assert(("explanation", check));
// syntax since the left hand argument has no side-effects.
// I prefer using the comma operator over && though because
// the parentheses are more readable, so I'm silencing
// the warning for this file.
// #pragma GCC diagnostic ignored "-Wunused-value"

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

#include <galois/AtomicHelpers.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/runtime/Profile.h"

static char const* name = "Fast Marching Method";
static char const* desc =
    "Eikonal equation solver "
    "(https://en.wikipedia.org/wiki/Fast_marching_method)";
static char const* url = "";

using data_t         = double;
constexpr data_t INF = std::numeric_limits<data_t>::max();

static llvm::cl::opt<std::string> filename(llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::Required);

enum Algo { serial = 0, parallel };
enum SourceType { scatter = 0, analytical };

const char* const ALGO_NAMES[] = {"serial", "parallel"};

static llvm::cl::OptionCategory catAlgo("Algorithmic Options");
static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::desc("Choose an algorithm (default parallel):"),
         llvm::cl::values(clEnumVal(serial, "serial heap implementation"),
                          clEnumVal(parallel, "parallel implementation")),
         llvm::cl::init(parallel), llvm::cl::cat(catAlgo));
static llvm::cl::opt<unsigned> RF{"rf",
                                  llvm::cl::desc("round-off factor for OBIM"),
                                  llvm::cl::init(0u), llvm::cl::cat(catAlgo)};

static llvm::cl::OptionCategory catInput("Input Options");
static llvm::cl::opt<SourceType> source_type(
    "sourceFormat", llvm::cl::desc("Choose an source format:"),
    llvm::cl::values(clEnumVal(scatter, "a set of discretized points"),
                     clEnumVal(analytical, "boundary in a analytical form")),
    llvm::cl::init(analytical), llvm::cl::cat(catInput));
// TODO parameterize the following
static constexpr data_t xa = -.5, xb = .5;
static constexpr data_t ya = -.5, yb = .5;
static constexpr data_t za = -.5, zb = .5;

static llvm::cl::OptionCategory catDisc("Discretization options");
static llvm::cl::opt<unsigned long long> nh{
    "nh",
    llvm::cl::desc("number of cells in ALL direction. NOTE: this will override "
                   "nx, ny, nz."),
    llvm::cl::init(0u), llvm::cl::cat(catDisc)};
static llvm::cl::opt<unsigned long long> nx{
    "nx", llvm::cl::desc("number of cells in x direction"), llvm::cl::init(10u),
    llvm::cl::cat(catDisc)};
static llvm::cl::opt<unsigned long long> ny{
    "ny", llvm::cl::desc("number of cells in y direction"), llvm::cl::init(10u),
    llvm::cl::cat(catDisc)};
static llvm::cl::opt<unsigned long long> nz{
    "nz", llvm::cl::desc("number of cells in z direction"), llvm::cl::init(10u),
    llvm::cl::cat(catDisc)};

static std::size_t NUM_CELLS;
static data_t dx, dy, dz;
void ParseOptions() {
  if (nh) {
    // options are not copyable; only copy the value
    nx = ny = nz = nh.getValue();
  }

  // metric inference
  NUM_CELLS = nh ? nh * nh * nh : nx * ny * nz;
  dx        = (xb - xa) / data_t(nx + 1);
  dy        = (yb - ya) / data_t(ny + 1);
  dz        = (zb - za) / data_t(nz + 1);
  if (!RF)
    RF = 1 / std::min({dx, dy, dz}, std::less<data_t>{});

  galois::gDebug(nx, " - ", ny, " - ", nz);
  galois::gDebug(dx, " - ", dy, " - ", dz);
  galois::gDebug("RF: ", RF);
}

///////////////////////////////////////////////////////////////////////////////

constexpr galois::MethodFlag no_lockable_flag = galois::MethodFlag::UNPROTECTED;

// TODO: In Galois, we need a graph type with dynamically sized
// node/edge data for this problem. For now, indexing into a
// separate data structure will have to be sufficient.

// Note: I'm going to use a CSR graph, so each node will already have a
// unique std::size_t id that can be used to index other data structures.
// I'll also use a std::size_t cutoff to distinguish between ghost cells
// that only exist to provide boundary condition data and actual cells.

// Each edge holds the unit normal pointing outward
// from the corresponding source cell in the graph.
// Note: this will be the negative of the vector stored
// on the edge coming the opposite direction.
// Note: In the regular grid case, this could be considered redundant,
// but this code hopefully will be adapted to handle irregular
// geometry at some point.
// Note: The sweeping direction for each direction along each edge
// could just be pre-computed, but that'd noticeably increase
// storage requirements.
// TODO: Try caching sweep directions and see if it's any better.

// Both these limitations could be lifted,
// but in the interest of keeping the buffer management
// code simple, I'm just going to assume them.
static_assert(sizeof(std::atomic<std::size_t>) <= sizeof(double),
              "Current buffer allocation code assumes atomic "
              "counters smaller than sizeof(double).");
static_assert(std::is_trivial_v<std::atomic<std::size_t>> &&
                  std::is_standard_layout_v<std::atomic<std::size_t>>,
              "Current buffer allocation code assumes no special "
              "construction/deletion code is needed for atomic counters.");
///////////////////////////////

// FMM
// enum Tag { KNOWN_FIX, KNOWN_OLD, KNOWN_NEW,
//            BAND_OLD, BAND_NEW, FAR };
enum Tag { KNOWN, BAND, FAR };

struct NonAtomicNodeData {
  bool is_ghost;
  Tag tag;
  double speed;
  double solution;
};

struct NodeData {
  bool is_ghost; // read only
  std::atomic<Tag> tag;
  double speed; // read only
  std::atomic<double> solution;
};
auto constexpr atomic_order = std::memory_order_relaxed;

#include "fastmarchingmethod.h"
#include "unstructured/Mesh.h"
#include "unstructured/Element.h"
#include "unstructured/Verifier.h"

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!

///////////////////////////////////////////////////////////////////////////////

// TODO analytical boundary
// static bool NonNegativeRegion(const std::array<CoordTy, 3>& coords) {
//   const CoordTy& x = coords[0], y = coords[1], z = coords[2];
//
//   // Example 1: a spherical interface of radius 0.25 centered at the origin
//   // return x * x + y * y + z * z >= .25 * .25;
//
//   // Example 2: a plane past through the origin
//   return 100. * x + y + 2. * z >= 0.;
// }
//
// template <typename Graph, typename BL,
//           typename GNode = typename Graph::GraphNode,
//           typename T = typename BL::value_type>
// void AssignBoundary(Graph& graph, BL& boundary) {
//   galois::do_all(
//     galois::iterate(0ul, NUM_CELLS),
//     [&](T node) noexcept {
//       if (node > NUM_CELLS) return;
//
//       if (NonNegativeRegion(id2xyz(node))) {
//         for (auto e : graph.edges(node, no_lockable_flag)) {
//           GNode dst = graph.getEdgeDst(e);
//           if (!NonNegativeRegion(id2xyz(dst))) {
// // #ifndef NDEBUG
// //             auto c = id2xyz(node);
// //             galois::gDebug(node, " (", c[0], " ", c[1], " ", c[2], ")");
// // #endif
//             boundary.push(node);
//             break;
//           }
//         }
//       }
//     },
//     galois::loopname("assignBoundary"));
// }

template <typename Graph, typename BL,
          typename GNode = typename Graph::GraphNode,
          typename T     = typename BL::value_type>
void AssignBoundary(Graph& graph, BL& boundary) {
  Tuple n = {0., 0.};

  galois::do_all(
      galois::iterate(graph.begin(), graph.end()),
      [&](GNode nh) noexcept {
        auto& data = graph.getData(nh, no_lockable_flag);
        if (data.dim() == 2)
          return;

        if (data.inTriangle(n)) {
          boundary.push(nh);
        }
      },
      galois::loopname("assignBoundary"));
}

/////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

template <typename Graph, typename PDM,
          typename GNode = typename Graph::GraphNode>
void initCells(Graph& graph, PDM& getPointData) {
  galois::do_all(
      galois::iterate(graph.begin(), graph.end()),
      [&](GNode nh) noexcept {
        auto& dTriangle = graph.getData(nh, galois::MethodFlag::UNPROTECTED);
        const auto& ids = dTriangle.getIds();
        for (int i = 0; i < dTriangle.dim(); ++i) {
          auto& dPoint = getPointData(ids[i], no_lockable_flag);
          // dPoint.is_ghost = (node >= num_cells);
          dPoint.tag      = FAR;
          auto& p         = dTriangle.getPoint(i);
          dPoint.speed    = SpeedFunction(std::array{p[0], p[1]});
          dPoint.solution = INF; // TODO ghost init?
        }
      },
      galois::no_stats(), galois::loopname("initializeCells"));
}

template <typename Graph, typename BL, typename PDM,
          typename T = typename BL::value_type>
void initBoundary(Graph& graph, BL& boundary, PDM& getPointData) {
  galois::do_all(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](const T& nh) noexcept {
        auto& dTriangle = graph.getData(nh, galois::MethodFlag::UNPROTECTED);
        const auto& ids = dTriangle.getIds();
        for (int i = 0; i < dTriangle.dim(); ++i) {
          auto& dPoint    = getPointData(ids[i], no_lockable_flag);
          dPoint.tag      = KNOWN;
          auto& p         = dTriangle.getPoint(i);
          dPoint.solution = BoundaryCondition(std::array{p[0], p[1]});
        }
      },
      galois::no_stats(), galois::loopname("initializeBoundary"));
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Solver

template <typename TriangleData>
double solveQuadratic(TriangleData& td, int iC, const double speedC, double tA,
                      double tB) {
  const Tuple& C = td.getPoint(iC);
  const Tuple& A = td.getPoint((iC + 1) % 3);
  const Tuple& B = td.getPoint((iC + 2) % 3);

  double c = A.distance(B), a = B.distance(C), b = A.distance(C);

  Tuple AB      = B - A;
  Tuple AC      = C - A;
  Tuple CB      = B - C;
  double cosABC = (AB * CB) / (c * a), cosBAC = (AB * AC) / (c * b),
         rho = a * (1. - cosABC * cosABC);

  return (rho * std::sqrt(speedC * speedC * c * c - (tA - tB) * (tA - tB)) +
          a * tA * cosABC + b * tB * cosBAC) /
         c;
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// first iteration

template <bool CONCURRENT, typename Graph, typename BL, typename WL,
          typename PDM, typename GNode = typename Graph::GraphNode,
          typename BT = typename BL::value_type,
          typename WT = typename WL::value_type>
void FirstIteration(Graph& graph, BL& boundary, WL& init_wl,
                    PDM& getPointData) {
  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;
  Loop loop;

  loop(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](BT nh) noexcept {
        for (auto e : graph.edges(nh, no_lockable_flag)) {
          GNode dsth    = graph.getEdgeDst(e);
          auto& dstData = graph.getData(dsth, no_lockable_flag);
          if (dstData.dim() == 3) {
            const auto& ids = dstData.getIds();
            int unknown     = 3;
            { // TODO make it something like if (dstData.tag != KNOWN)
              for (int i = 0; i < dstData.dim(); ++i) {
                auto& dPoint = getPointData(ids[i], no_lockable_flag);
                if (dPoint.tag != KNOWN) {
                  // assert(dstData.solution.load(atomic_order) == INF ||
                  // dstData.tag.load(atomic_order) != FAR);
                  assert(unknown == 3);
                  unknown = i;
                }
              }
            }
            if (unknown != 3) {
              auto& farPD = getPointData(ids[unknown], no_lockable_flag);
              double tA = getPointData(ids[(unknown + 1) % 3], no_lockable_flag)
                              .solution.load(atomic_order);
              double tB = getPointData(ids[(unknown + 1) % 3], no_lockable_flag)
                              .solution.load(atomic_order);
              [[maybe_unused]] data_t old_sln = farPD.solution.load(
                  atomic_order); // hint for heap update only
              data_t sln_temp =
                  CONCURRENT
                      ?
                      // solveQuadratic(graph, dsth, old_sln, farPD.speed) :
                      solveQuadratic(dstData, unknown, farPD.speed, tA, tB)
                      : solveQuadratic(dstData, unknown, farPD.speed, tA, tB);
              if (sln_temp < galois::atomicMin(farPD.solution, sln_temp)) {
                galois::gDebug("Hi! I'm ", dsth, " I got ", sln_temp);
                if (auto old_tag = farPD.tag.load(atomic_order);
                    old_tag != BAND) {
                  while (!farPD.tag.compare_exchange_weak(
                      old_tag, BAND, std::memory_order_relaxed))
                    ;
                  if constexpr (CONCURRENT)
                    init_wl.push(WT{sln_temp, dsth});
                  else
                    init_wl.push(WT{sln_temp, dsth}, old_sln);
#ifndef NDEBUG
                } else {
                  if constexpr (CONCURRENT) {
                    bool in_wl = false;
                    for (auto [_, i] : init_wl) {
                      if (i == nh) {
                        in_wl = true;
                        break;
                      }
                    }
                    assert(in_wl);
                  }
#endif
                }
              }
            }
          }
        }
      },
      galois::loopname("FirstIteration"));
}

////////////////////////////////////////////////////////////////////////////////

// TODO
template <typename RangeFunc, typename FunctionTy, typename... Args>
void while_wrapper(const RangeFunc& rangeMaker, FunctionTy&& fn,
                   const Args&... args) {
  auto tpl = std::make_tuple(args...);
  // runtime::for_each_gen(rangeMaker(tpl), std::forward<FunctionTy>(fn), tpl);
}

////////////////////////////////////////////////////////////////////////////////
// Operator

////////////////////////////////////////////////////////////////////////////////
// FMM

template <bool CONCURRENT, typename Graph, typename WL, typename PDM,
          typename GNode = typename Graph::GraphNode,
          typename T     = typename WL::value_type>
void FastMarching(Graph& graph, WL& wl, PDM& getPointData) {
  [[maybe_unused]] double max_error = 0;
  std::size_t num_iterations        = 0;

  auto PushOp = [&]<typename ItemTy, typename UC>(const ItemTy& item, UC& wl) {
    galois::gDebug("new item");
    auto [_old_sln, node] = item;
    auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
    assert(curData.dim() == 3);
    { // TODO make it a function
      const auto& ids = curData.getIds();
      galois::gDebug("node ", node, " ", ids[0], " ", ids[1], " ", ids[2]);
      for (int i = 0; i < curData.dim(); ++i) {
        auto& dPoint = getPointData(ids[i], no_lockable_flag);
        if (dPoint.tag != KNOWN) {
          // assert(dstData.solution.load(atomic_order) == INF ||
          // dstData.tag.load(atomic_order) != FAR);
#ifndef NDEBUG
//     {
//       auto [x, y, z] = id2xyz(node);
//       galois::gDebug("Hi! I'm ", node, " (", x, " ", y, " ", z, " ) with ",
//       curData.solution); assert(curData.solution != INF);
//     }
// if constexpr (CONCURRENT) {
//     if (curData.tag == KNOWN) {
//       galois::gDebug(node, " in bag as KNWON");
//     }
//     assert(curData.tag == BAND);
// } else {
//     assert(curData.tag != KNOWN);
//     if (curData.tag != BAND) {
//       galois::gDebug(node, " in heap with tag ", curData.tag);
//       std::abort();
//     }
//
//     {
//       auto [x, y, z] = id2xyz(node);
//       if (curData.solution - std::sqrt(x * x + y * y + z * z) > max_error)
//         max_error = curData.solution - std::sqrt(x * x + y * y + z * z);
//       if (curData.solution - std::sqrt(x * x + y * y + z * z) > 0.2) {
//         galois::gDebug(curData.solution - std::sqrt(x * x + y * y + z * z),
//           " - wrong distance, should be ", std::sqrt(x * x + y * y + z * z));
//         assert(false);
//       }
//     }
// }
#endif
          dPoint.tag.store(KNOWN, atomic_order);
        }
      }
    }

    // UpdateNeighbors
    for (auto e : graph.edges(node, no_lockable_flag)) {
      GNode dsth    = graph.getEdgeDst(e);
      auto& dstData = graph.getData(dsth, no_lockable_flag);
      if (dstData.dim() == 3) {
        const auto& ids = dstData.getIds();
        galois::gDebug(dsth, " ", ids[0], " ", ids[1], " ", ids[2]);
        bool _debug = false;
        // if (ids[0] == 31 && ids[1] == 32 && ids[2] == 19)
        //   _debug = true;
        // if (_debug) galois::gDebug("first");
        int unknown = 3;
        { // TODO make it something like if (dstData.tag != KNOWN)
          for (int i = 0; i < dstData.dim(); ++i) {
            auto& dPoint = getPointData(ids[i], no_lockable_flag);
            if (_debug)
              galois::gDebug("KEY:", ids[i], " ", dPoint.tag.load(atomic_order),
                             " sol. ", dPoint.solution.load(atomic_order));
            if (dPoint.tag != KNOWN) {
              // assert(dstData.solution.load(atomic_order) == INF ||
              // dstData.tag.load(atomic_order) != FAR);
              if (unknown != 3)
                break; // TODO parallel conflict
              unknown = i;
            }
          }
        }
        if (unknown != 3) {
          // if (_debug) galois::gDebug("second");
#ifndef NDEBUG
//           {
//             auto [x, y, z] = id2xyz(dst);
//             galois::gDebug("Update ", dst, " (", x, " ", y, " ", z, " )");
//           }
#endif
          // assert(dstData.solution == INF && dstData.tag == FAR);
          auto& farPD = getPointData(ids[unknown], no_lockable_flag);
          double tA   = getPointData(ids[(unknown + 1) % 3], no_lockable_flag)
                          .solution.load(atomic_order);
          double tB = getPointData(ids[(unknown + 1) % 3], no_lockable_flag)
                          .solution.load(atomic_order);
          data_t old_sln [[maybe_unused]] =
              farPD.solution.load(atomic_order); // TODO remove
          data_t sln_temp =
              CONCURRENT
                  ?
                  // solveQuadratic(graph, dsth, old_sln, farPD.speed) :
                  solveQuadratic(dstData, unknown, farPD.speed, tA, tB)
                  : solveQuadratic(dstData, unknown, farPD.speed, tA, tB);
          if (sln_temp < galois::atomicMin(farPD.solution, sln_temp) ||
              farPD.tag == BAND) {
            // if (_debug) galois::gDebug("third");
            galois::gDebug("Hi! I'm ", dsth, " I got ", sln_temp);
            auto old_tag = farPD.tag.load(atomic_order);
            if constexpr (CONCURRENT) {
              // if (old_tag != BAND) {
              while (!farPD.tag.compare_exchange_weak(
                  old_tag, BAND, std::memory_order_relaxed))
                ;
              wl.push(ItemTy{sln_temp, dsth});
              // }
            } else {
              // if (_debug) galois::gDebug("fourth");
              while (old_tag != BAND &&
                     !farPD.tag.compare_exchange_weak(
                         old_tag, BAND, std::memory_order_relaxed))
                ;
              wl.push(ItemTy{sln_temp, dsth}, old_sln);
            }
          } else {
            galois::gDebug(dsth, " solution not updated: ", sln_temp,
                           " (currently ", farPD.solution, ")");
          }
        }
      } else {
        // segments (dim() == 2)
        const auto& ids = dstData.getIds();
        galois::gDebug(dsth, " ", ids[0], " ", ids[1]);
      }
    }
  };

  if constexpr (CONCURRENT) {
    galois::GReduceMax<double> max_error;
    auto Indexer = [&](const T& item) {
      unsigned t = std::round(item.first * RF);
      // galois::gDebug(item.first, "\t", t, "\n");
      return t;
    };
    using PSchunk = galois::worklists::PerSocketChunkLIFO<32>; // chunk size 16
    using OBIM =
        galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;

    galois::runtime::profileVtune(
        [&]() {
          galois::for_each(galois::iterate(wl.begin(), wl.end()), PushOp,
                           galois::disable_conflict_detection(),
                           // galois::no_stats(),  // stat iterations
                           galois::wl<OBIM>(Indexer), galois::loopname("FMM"));
        },
        "FMM_VTune");
  } else {
    while (!wl.empty()) {

      PushOp(wl.pop(), wl);

      num_iterations++;
    }
    galois::gDebug("max error: ", max_error);
    galois::gPrint("#iterarions: ", num_iterations, "\n");
  }
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Algo

template <bool CONCURRENT, typename WL, typename PDM, typename Graph,
          typename BL>
void runAlgo(Graph& graph, BL& boundary, PDM& getPointData) {
  WL initBag;

  FirstIteration<CONCURRENT>(graph, boundary, initBag, getPointData);

  if (initBag.empty()) {
    galois::gDebug("No cell to be processed!");
    std::abort();
#ifndef NDEBUG
  } else {
    galois::gDebug("vvvvvvvv init band vvvvvvvv");
    for (auto [_, i] : initBag) {
      auto& data = graph.getData(i, no_lockable_flag);
      auto v1 = data.getPoint(0), v2 = data.getPoint(1), v3 = data.getPoint(2);
      auto [i1, i2, i3] = data.getIds();
      double t1 =
                 getPointData(i1, no_lockable_flag).solution.load(atomic_order),
             t2 =
                 getPointData(i2, no_lockable_flag).solution.load(atomic_order),
             t3 =
                 getPointData(i3, no_lockable_flag).solution.load(atomic_order);
      galois::gDebug(i, " (", v1, ", ", v2, ", ", v3, ") with ", "[", t1, " ",
                     t2, " ", t3, "]");
    }
    galois::gDebug("^^^^^^^^ init band ^^^^^^^^");

    galois::do_all(
        galois::iterate(initBag),
        [&](auto pair) {
          galois::gDebug(pair.first, " : ", pair.second);
          auto [_, node] = pair;
          auto& data     = graph.getData(node, no_lockable_flag);
          for (auto i : data.getIds()) {
            if (getPointData(i).tag != KNOWN) {
              galois::gDebug("UNKNOWN");
            }
          }
        },
        galois::no_stats(), galois::loopname("DEBUG_initBag_sanity_check"));
#endif // end of initBag sanity check;
  }

  FastMarching<CONCURRENT>(graph, initBag, getPointData);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Sanity check

template <typename Graph, typename PDM,
          typename GNode = typename Graph::GraphNode>
void SanityCheck2(Graph& graph, PDM& getPointData) {
  for (int i = 0; i < 40; ++i)
    galois::gDebug(i, " ", getPointData(i).tag, " ", getPointData(i).solution);

  galois::GReduceMax<double> max_difference;
  galois::GReduceMax<double> max_edge;

  galois::do_all(
      galois::iterate(graph),
      [&](GNode nh) noexcept {
        auto& triangle_data = graph.getData(nh, no_lockable_flag);
        std::array<Tuple, 3> tuples;
        for (int i = 0; i < triangle_data.dim(); ++i) {
          tuples[i] = triangle_data.getPoint(i);
        }
        auto& [A, B, C] = tuples;
        max_edge.update(A.distance(B));
        if (triangle_data.dim() == 3) {
          max_edge.update(A.distance(C));
          max_edge.update(B.distance(C));
        }
        auto& [iA, iB, iC] = triangle_data.getIds();
        galois::gDebug(nh, " ", iA, " ", iB, " ", iC);
        assert(getPointData(iA).solution < INF);
        assert(getPointData(iB).solution < INF);
        if (triangle_data.dim() == 3) {
          assert(getPointData(iC).solution < INF);
        }
        max_difference.update(std::abs(getPointData(iA).solution -
                                       std::sqrt(A[0] * A[0] + A[1] * A[1])));
        max_difference.update(std::abs(getPointData(iB).solution -
                                       std::sqrt(B[0] * B[0] + B[1] * B[1])));
        if (triangle_data.dim() == 3) {
          max_difference.update(std::abs(getPointData(iC).solution -
                                         std::sqrt(C[0] * C[0] + C[1] * C[1])));
        }
      },
      galois::no_stats(), galois::loopname("sanityCheck2"));

  galois::gPrint("max diff: ", max_difference.reduce(), "\n");
  galois::gPrint("max edge: ", max_edge.reduce(), "\n");
}

template <typename GNode>
void _debug_print() {
  for (GNode i = 0; i < NUM_CELLS; i++) {
    // auto [x, y, z] = id2xyz(i);
    // galois::gDebug(x, " ", y, " ", z, " ", xyz2id({x, y, z}));
  }
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
class PointDataManager {
  typedef T& reference;
  galois::LargeArray<T> pointData;

public:
  PointDataManager(size_t numPoints) {
    pointData.allocateInterleaved(numPoints);
    for (size_t n = 0; n < numPoints; ++n) {
      pointData.constructAt(n);
    }
  }

  reference operator()(size_t N, galois::MethodFlag GALOIS_UNUSED(mflag) =
                                     galois::MethodFlag::WRITE) {
    return pointData[N];
  }
};

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  galois::gDebug(ALGO_NAMES[algo]);

  // global_config();

  using Graph =
      galois::graphs::MorphGraph<Element, void, false>; // directional = false
  using GNode         = Graph::GraphNode;
  using BL            = galois::InsertBag<GNode>;
  using UpdateRequest = std::pair<data_t, GNode>;
  using HeapTy        = FMMHeapWrapper<
      std::multimap<UpdateRequest::first_type, UpdateRequest::second_type>>;
  using WL = galois::InsertBag<UpdateRequest>;
  // read meshes
  Graph graph;
  std::size_t numPoints;
  {
    Mesh mHelper;
    bool parallelAllocate = false;
    numPoints             = mHelper.read(graph, filename.c_str(),
                             parallelAllocate); // detAlgo == nondet);

    for (auto ele : graph) {
      auto& d = graph.getData(ele, no_lockable_flag);
      if (d.isObtuse()) {
        galois::gPrint(d.getId(), " ", d.dim(), " ");
        Tuple A = d.getPoint(0);
        Tuple B = d.getPoint(1);
        Tuple C = d.getPoint(2);
        A.print(std::cout);
        std::cout << " ";
        B.print(std::cout);
        std::cout << " ";
        C.print(std::cout);
        std::cout << " ";
        std::cout << "\n";
        std::cerr << d.getPoint(0).angle(d.getPoint(1), d.getPoint(2)) << "\n";
        std::cerr << d.getPoint(1).angle(d.getPoint(0), d.getPoint(2)) << "\n";
        std::cerr << d.getPoint(2).angle(d.getPoint(0), d.getPoint(1)) << "\n";
        GALOIS_DIE("obtuse triangle detected");
      }
    }

    Verifier<Graph, GNode, Element, Tuple> v;
    if (!skipVerify && !v.verify(graph)) {
      GALOIS_DIE("bad input mesh");
    }
  }
  PointDataManager<NodeData> pdm{numPoints};

  // initialize all cells
  initCells(graph, pdm);

  // TODO better way for boundary settings?
  BL boundary;
  // if (source_type == scatter)
  AssignBoundary(graph, boundary);
  // else
  //  AssignBoundary(graph, boundary);
  assert(!boundary.empty() && "Boundary not defined!");

#ifndef NDEBUG
  // print boundary
  galois::gDebug("vvvvvvvv boundary vvvvvvvv");
  for (GNode b : boundary) {
    auto& data = graph.getData(b, no_lockable_flag);
    auto v1 = data.getPoint(0), v2 = data.getPoint(1), v3 = data.getPoint(2);
    galois::gDebug(b, " (", v1, ", ", v2, ", ", v3, ")");
  }
  galois::gDebug("^^^^^^^^ boundary ^^^^^^^^");
#endif

  initBoundary(graph, boundary, pdm);

  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case serial:
    runAlgo<false, HeapTy>(graph, boundary, pdm);
    break;
  case parallel:
    runAlgo<true, WL>(graph, boundary, pdm);
    break;
  default:
    std::abort();
  }

  Tmain.stop();
  //
  //  SanityCheck(graph);
  SanityCheck2(graph, pdm);

  return 0;
}
