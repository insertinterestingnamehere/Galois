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

#define DIM_LIMIT 2 // support 2-D
using data_t         = double;
constexpr data_t INF = std::numeric_limits<data_t>::max();

enum Algo { serial = 0, parallel };
enum SourceType { scatter = 0, analytical };

const char* const ALGO_NAMES[] = {"serial", "parallel"};

static llvm::cl::OptionCategory catAlgo("1. Algorithmic Options");
static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::value_desc("algo"),
         llvm::cl::desc("Choose an algorithm (default parallel):"),
         llvm::cl::values(clEnumVal(serial, "serial heap implementation"),
                          clEnumVal(parallel, "parallel implementation")),
         llvm::cl::init(parallel), llvm::cl::cat(catAlgo));
static llvm::cl::opt<unsigned> RF{"rf",
                                  llvm::cl::desc("round-off factor for OBIM"),
                                  llvm::cl::init(0u), llvm::cl::cat(catAlgo)};
static llvm::cl::opt<double> tolerance("e", llvm::cl::desc("Final error bound"),
                                       llvm::cl::init(2.e-6),
                                       llvm::cl::cat(catAlgo));

static llvm::cl::OptionCategory catInput("2. Input Options");
static llvm::cl::opt<SourceType> source_type(
    "sourceFormat", llvm::cl::desc("Choose an source format:"),
    llvm::cl::values(clEnumVal(scatter, "a set of discretized points"),
                     clEnumVal(analytical, "boundary in a analytical form")),
    llvm::cl::init(scatter), llvm::cl::cat(catInput));
static llvm::cl::opt<std::string> input_segy(
    "segy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc("Use SEG-Y (rev 1) file as input speed map. NOTE: This will "
                   "determine the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
static llvm::cl::opt<std::string> input_npy(
    "inpy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use npy file (dtype=float32) as input speed map. NOTE: This will "
        "determine the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
static llvm::cl::opt<std::string> input_csv(
    "icsv", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use csv file as input speed map. NOTE: Current implementation "
        "requires explicit definition of the size on each dimensions (see -d)"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
// TODO parameterize the following
static data_t xa = -.5, xb = .5;
static data_t ya = -.5, yb = .5;

static llvm::cl::OptionCategory catOutput("3. Output Options");
static llvm::cl::opt<std::string>
    output_csv("ocsv", llvm::cl::desc("Export results to a csv format file"),
               llvm::cl::init(""), llvm::cl::cat(catOutput));
static llvm::cl::opt<std::string>
    output_npy("onpy", llvm::cl::desc("Export results to a npy format file"),
               llvm::cl::init(""), llvm::cl::cat(catOutput));

static llvm::cl::OptionCategory catDisc("4. Discretization options");
template <typename NumTy, int MAX_SIZE = 0>
struct NumVecParser : public llvm::cl::parser<std::vector<NumTy>> {
  template <typename... Args>
  NumVecParser(Args&... args) : llvm::cl::parser<std::vector<NumTy>>(args...) {}
  // parse - Return true on error.
  bool parse(llvm::cl::Option& O, llvm::StringRef ArgName,
             const std::string& ArgValue, std::vector<NumTy>& Val) {
    const char* beg = ArgValue.c_str();
    char* end;

    do {
      std::size_t d = strtoul(Val.empty() ? beg : end + 1, &end, 0);
      if (!d)
        return O.error("Invalid option value '" + ArgName + "=" + ArgValue +
                       "': should be comma-separated unsigned integers");
      Val.push_back(d);
    } while (*end == ',');
    if (*end != '\0')
      return O.error("Invalid option value '" + ArgName + "=" + ArgValue +
                     "': should be comma-separated unsigned integers");
    if (MAX_SIZE && Val.size() > MAX_SIZE)
      return O.error(ArgName + "=" + ArgValue + ": expect no more than " +
                     std::to_string(MAX_SIZE) + " numbers but get " +
                     std::to_string(Val.size()));
    return false;
  }
};
static llvm::cl::opt<std::vector<std::size_t>, false,
                     NumVecParser<std::size_t, DIM_LIMIT>>
    dims("d", llvm::cl::value_desc("d1,d2"),
         llvm::cl::desc("Size of each dimensions as a comma-separated array "
                        "(support up to 2-D)"),
         llvm::cl::cat(catDisc));

static uint64_t nx, ny;
static std::size_t NUM_CELLS;
static data_t dx, dy;
void SetKnobs(const std::vector<std::size_t>& d) {
  assert(d.size() == DIM_LIMIT);
  nx = d[0];
  ny = d[1];

  // metric inference
  NUM_CELLS = nx * ny;
  dx        = (xb - xa) / data_t(nx + 1);
  dy        = (yb - ya) / data_t(ny + 1);

  if (!RF)
    RF = 1 / std::min({dx, dy, 1.}, std::less<data_t>{});

  galois::gDebug(nx, " - ", ny);
  galois::gDebug(dx, " - ", dy);
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

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!

///////////////////////////////////////////////////////////////////////////////

#include "fastmarchingmethod.h"
#include "structured/grids.h"
#include "structured/utils.h"

template <typename Graph, typename BL,
          typename GNode = typename Graph::GraphNode,
          typename T     = typename BL::value_type>
void AssignBoundary(Graph& graph, BL& boundary) {
  galois::do_all(
      galois::iterate(0ul, NUM_CELLS),
      [&](T node) noexcept {
        if (node > NUM_CELLS)
          return;

        if (NonNegativeRegion(id2xy(node))) {
          for (auto e : graph.edges(node, no_lockable_flag)) {
            GNode dst = graph.getEdgeDst(e);
            if (!NonNegativeRegion(id2xy(dst))) {
              // #ifndef NDEBUG
              //             auto c = id2xyz(node);
              //             galois::gDebug(node, " (", c[0], " ", c[1], " ",
              //             c[2], ")");
              // #endif
              boundary.push(node);
              break;
            }
          }
        }
      },
      galois::loopname("assignBoundary"));
}

// template <typename WL>
// void AssignBoundary(WL& boundary) {
//   for (GNode i = 0; i < nx * ny; i++) {
//     boundary.push(i);
//   }
// }

template <typename BL>
void AssignBoundary(BL& boundary) {
  // #ifndef NDEBUG
  //   GNode n = xyz2id({0., 0., 0.});
  //   auto c = id2xyz(n);
  //   galois::gDebug(n, " (", c[0], " ", c[1], " ", c[2], ")");
  // #endif
  std::size_t n = xy2id({0., 0.});
  boundary.push(n);
}

/////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

template <typename Graph, typename BL, typename T = typename BL::value_type>
void initBoundary(Graph& graph, BL& boundary) {
  galois::do_all(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](const T& node) noexcept {
        auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        curData.tag      = KNOWN;
        curData.solution = BoundaryCondition(id2xy(node));
      },
      galois::no_stats(), galois::loopname("initializeBoundary"));
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Solver

template <typename Graph, typename GNode = typename Graph::GraphNode,
          typename It = typename Graph::edge_iterator>
auto checkDirection(Graph& graph, GNode node, data_t center_sln, It dir) {
#ifndef NDEBUG
  if (dir >= graph.edge_end(node, no_lockable_flag)) {
    galois::gDebug(node, " ",
                   (graph.getData(node, no_lockable_flag).is_ghost
                        ? "ghost"
                        : "non-ghost"),
                   " ",
                   std::distance(graph.edge_begin(node, no_lockable_flag),
                                 graph.edge_end(node, no_lockable_flag)));
    GALOIS_DIE("invalid direction");
  }
#endif
  data_t sln   = center_sln;
  GNode upwind = node;

  auto FindUpwind = [&](It ei) {
    GNode neighbor = graph.getEdgeDst(ei);
    auto& n_data   = graph.getData(neighbor, no_lockable_flag);
    galois::gDebug("Check neighbor ", neighbor, (int)n_data.tag);
    if (data_t n_sln = n_data.solution.load(std::memory_order_relaxed);
        n_sln < sln) {
      sln    = n_sln;
      upwind = neighbor;
    }
  };

  // check one neighbor
  FindUpwind(dir);
  std::advance(dir, 1); // opposite direction of the same dimension
  assert(dir < graph.edge_end(node, no_lockable_flag) &&
         "Ill-formed uni-directional dimension");
  // check the other neighbor
  FindUpwind(dir);

  if (upwind == node)
    return std::make_pair(0., 0.);
  return std::make_pair(sln, dx); // TODO second could be a boolean
}

template <typename Graph, typename GNode = typename Graph::GraphNode>
double solveQuadratic(Graph& graph, GNode node, data_t sln,
                      const data_t speed) {
  // TODO parameterize dimension 3
  std::array<std::pair<data_t, data_t>, DIM_LIMIT> sln_delta{
      std::make_pair(0., dx), std::make_pair(0., dy)};

  int non_zero_counter = 0;
  auto dir             = graph.edge_begin(node, no_lockable_flag);
  for (auto& [s, d] : sln_delta) {
#ifdef NDEBUG
    if (dir >= graph.edge_end(node, no_lockable_flag))
      GALOIS_DIE("Dimension exceeded");
#endif
    auto [si, di] = checkDirection(graph, node, sln, dir);
    if (di) {
      s = si;
      non_zero_counter++;
    } else {
      // s = 0.; // already there
      d = 0.;
    }
    std::advance(dir, 2);
  }
  galois::gDebug("solveQuadratic: ", sln_delta[0].second, " ",
                 sln_delta[1].second, " #non_zero: ", non_zero_counter);
  if (non_zero_counter == 0)
    return INF;
  while (non_zero_counter) {
    auto max_s_d_it = std::max_element(
        sln_delta.begin(), sln_delta.end(),
        [&](std::pair<double, double>& a, std::pair<double, double>& b) {
          return a.first < b.first;
        });
    double a(0.), b(0.), c(0.);
    for (const auto& p : sln_delta) {
      const double &s = p.first, d = p.second;
      galois::gDebug(s, " ", d);
      double temp = (d == 0. ? 0. : (1. / (d * d)));
      a += temp;
      temp *= s;
      b += temp;
      temp *= s;
      c += temp;
      galois::gDebug("tabs: ", temp, " ", a, " ", b, " ", c);
    }
    b *= -2.;
    c -= (1. / (speed * speed));
    double del = b * b - (4. * a * c);
    galois::gDebug(a, " ", b, " ", c, " del=", del);
    if (del >= 0) {
      double new_sln = (-b + std::sqrt(del)) / (2. * a);
      galois::gDebug("new solution: ", new_sln);
      if (new_sln > max_s_d_it->first) {
        galois::gDebug(non_zero_counter, sln, new_sln, max_s_d_it->first);
        // assert(new_sln <= sln);  // false assertion
        sln = std::min(sln, new_sln);
      } else {
        galois::gDebug(non_zero_counter, sln, new_sln, max_s_d_it->first);
        // assert(false && "available solution should not violate the
        // causality"); // false assertion
      }
    }
    max_s_d_it->first  = 0.;
    max_s_d_it->second = 0.;
    non_zero_counter--;
  }
  return sln;
}

template <typename Graph, typename GNode = typename Graph::GraphNode>
double SerialSolveQuadratic(Graph& graph, GNode node, double sln,
                            const double speed) {
  // TODO oarameterize dimension 2
  std::array<std::pair<double, double>, 2> sln_delta{std::make_pair(0., dx),
                                                     std::make_pair(0., dy)};
  int non_zero_counter = 0;
  auto dir             = graph.edge_begin(node, no_lockable_flag);
  for (auto& p : sln_delta) {
    if (dir == graph.edge_end(node, no_lockable_flag))
      break;
    double& s     = p.first;
    double& d     = p.second;
    auto [si, di] = checkDirection(graph, node, sln, dir);
    if (di) {
      s = si;
      non_zero_counter++;
    } else {
      // s = 0.; // already there
      d = 0.;
    }
    std::advance(dir, 2);
  }
  if (non_zero_counter == 0)
    return INF;
  // galois::gDebug("SerialSolveQuadratic: ", sln_delta[0].second, " ",
  // sln_delta[1].second, " ", sln_delta[2].second); for (; non_zero_counter >
  // 0; non_zero_counter--) { auto max_sln_it =
  // std::max_element(solutions.begin(), solutions.end());
  double a(0.), b(0.), c(0.);
  for (const auto& p : sln_delta) {
    const double &s = p.first, d = p.second;
    galois::gDebug(s, " ", d);
    double temp = (d == 0. ? 0. : (1. / (d * d)));
    a += temp;
    temp *= s;
    b += temp;
    temp *= s;
    c += temp;
    // galois::gDebug(temp, " ", a, " ", b, " ", c);
  }
  b *= -2.;
  c -= (1. / (speed * speed));
  double del = b * b - (4. * a * c);
  galois::gDebug(a, " ", b, " ", c, " del=", del);
  if (del >= 0) {
    double new_sln = (-b + std::sqrt(del)) / (2. * a);
    galois::gDebug("new solution: ", new_sln);
    // if (new_sln > *max_sln_it)
    //  sln = std::min(sln, new_sln); // TODO atomic here?
    return new_sln;
  }
  // *max_sln_it = 0.;
  // }
  return INF;
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// first iteration

template <bool CONCURRENT, typename Graph, typename BL, typename WL,
          typename GNode = typename Graph::GraphNode,
          typename BT    = typename BL::value_type,
          typename WT    = typename WL::value_type>
void FirstIteration(Graph& graph, BL& boundary, WL& init_wl) {
  using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                         galois::StdForEach>::type;
  Loop loop;

  loop(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](BT node) noexcept {
        for (auto e : graph.edges(node, no_lockable_flag)) {
          GNode dst = graph.getEdgeDst(e);

          if (dst >= NUM_CELLS)
            continue;
          auto& dstData = graph.getData(dst, no_lockable_flag);
          // assert(!dstData.is_ghost);

          // if (dstData.tag == KNOWN)
          //   continue;
          // assert(dstData.solution.load(atomic_order) == INF ||
          //        dstData.tag.load(atomic_order) == BAND);

          data_t old_sln = dstData.solution.load(atomic_order);
          data_t sln_temp =
              CONCURRENT
                  ? solveQuadratic(graph, dst, old_sln, dstData.speed)
                  : SerialSolveQuadratic(graph, dst, old_sln, dstData.speed);
          if (old_sln == INF)
            assert(sln_temp != INF);
          if (sln_temp < galois::atomicMin(dstData.solution, sln_temp)) {
            galois::gDebug("Hi! I'm ", dst, " I got ", sln_temp);
            if (auto old_tag = dstData.tag.load(atomic_order);
                old_tag != BAND) {
              while (!dstData.tag.compare_exchange_weak(
                  old_tag, BAND, std::memory_order_relaxed))
                ;
              if constexpr (CONCURRENT)
                init_wl.push(WT{sln_temp, dst});
              else
                init_wl.push(WT{sln_temp, dst}, old_sln);
#ifndef NDEBUG
//                 } else {
//                   if constexpr (CONCURRENT) {
//                     bool in_wl = false;
//                     for (auto [_, i] : init_wl) {
//                       if (i == node) {
//                         in_wl = true;
//                         break;
//                       }
//                     }
//                     assert(in_wl);
//                   }
#endif
            }
          }
        }
      },
      galois::loopname("FirstIteration"));
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Operator

////////////////////////////////////////////////////////////////////////////////
// FMM

template <bool CONCURRENT, typename Graph, typename WL,
          typename GNode = typename Graph::GraphNode,
          typename T     = typename WL::value_type>
void FastMarching(Graph& graph, WL& wl) {
  [[maybe_unused]] double max_error = 0;
  std::size_t num_iterations        = 0;

  auto PushOp = [&](const auto& item, auto& wl) {
    // TODO gcc/9.2
    // auto [_old_sln, node] = item;
    using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
    typename ItemTy::second_type node;
    std::tie(std::ignore, node) = item;
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
#ifndef NDEBUG
    {
      assert(!curData.is_ghost && "impossible, asserted before");
      auto [x, y] = id2xy(node);
      galois::gDebug("Hi! I'm ", node, " (", x, " ", y, ") with ",
                     curData.solution);
      assert(curData.solution != INF);
    }
    if constexpr (CONCURRENT) {
      if (curData.tag == KNOWN) {
        galois::gDebug(node, " in bag as KNWON");
      }
      assert(curData.tag == BAND);
    } else {
      assert(curData.tag != KNOWN);
      if (curData.tag != BAND) {
        galois::gDebug(node, " in heap with tag ", curData.tag);
        std::abort();
      }

      // if (curData.solution != _old_sln) {
      //   galois::gDebug("Wrong entry in the heap! ", node, " ", _old_sln, "
      //   but ", curData.solution); for(auto i : wl) {
      //     if (i.second == node)
      //       galois::gDebug(i.first);
      //   }
      //   std::abort();
      // }

      {
        auto [x, y] = id2xy(node);
        if (curData.solution - std::sqrt(x * x + y * y) > max_error)
          max_error = curData.solution - std::sqrt(x * x + y * y);
        if (curData.solution - std::sqrt(x * x + y * y) > 0.2) {
          galois::gDebug(curData.solution - std::sqrt(x * x + y * y),
                         " - wrong distance, should be ",
                         std::sqrt(x * x + y * y));
          assert(false);
        }
      }
    }
#endif
    curData.tag.store(KNOWN, atomic_order);

    // UpdateNeighbors
    for (auto e : graph.edges(node, no_lockable_flag)) {
      GNode dst = graph.getEdgeDst(e);
      if (dst >= NUM_CELLS)
        continue;
      auto& dstData = graph.getData(dst, no_lockable_flag);
      assert(!dstData.is_ghost);
      // chaotic execution: KNOWN neighbor doesn't suffice with smaller value
      //   reason: circle-back update
      //   ... but it holds with ordered serial execution
      // smaller value doesn't suffice to be KNOWN - by-pass to be active
      if (dstData.solution > curData.solution) {
#ifndef NDEBUG
        {
          auto [x, y] = id2xy(dst);
          galois::gDebug("Update ", dst, " (", x, " ", y, ")");
        }
#endif
        // assert(dstData.solution == INF && dstData.tag == FAR);
        galois::gDebug("tag ", dstData.tag, ", sln ", dstData.solution);
        data_t old_sln = dstData.solution.load(atomic_order);
        // assert(old_sln > curData.solution); // not necessarily due to
        // atomics
        double sln_temp =
            CONCURRENT
                ? solveQuadratic(graph, dst, old_sln, dstData.speed)
                : SerialSolveQuadratic(graph, dst, old_sln, dstData.speed);
        if (sln_temp < galois::atomicMin(dstData.solution, sln_temp)) {
          // galois::atomicMin(dstData.solution, sln_temp); // TODO safe to
          // remove?
          auto old_tag = dstData.tag.load(atomic_order);
          if constexpr (CONCURRENT) {
            // if (old_tag != BAND) {
            while (!dstData.tag.compare_exchange_weak(
                old_tag, BAND, std::memory_order_relaxed))
              ;
            wl.push(ItemTy{sln_temp, dst});
            //}
          } else {
            while (old_tag != BAND &&
                   !dstData.tag.compare_exchange_weak(
                       old_tag, BAND, std::memory_order_relaxed))
              ;
            wl.push(ItemTy{sln_temp, dst}, old_sln);
          }
        } else {
          galois::gDebug(dst, " solution not updated: ", sln_temp,
                         " (currently ", dstData.solution, ")");
        }
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
        galois::worklists::AdaptiveOrderedByIntegerMetric<decltype(Indexer),
                                                          PSchunk>;

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

template <bool CONCURRENT, typename WL, typename Graph, typename BL>
void runAlgo(Graph& graph, BL& boundary) {
  WL initBag;

  FirstIteration<CONCURRENT>(graph, boundary, initBag);

  if (initBag.empty()) {
    galois::gDebug("No cell to be processed!");
    std::abort();
#ifndef NDEBUG
  } else {
    galois::gDebug("vvvvvvvv init band vvvvvvvv");
    for (auto [_, i] : initBag) {
      auto [x, y] = id2xy(i);
      galois::gDebug(i, " (", x, " ", y, ") with ",
                     graph.getData(i, no_lockable_flag).solution);
    }
    galois::gDebug("^^^^^^^^ init band ^^^^^^^^");

    galois::do_all(
        galois::iterate(initBag),
        [&](auto pair) {
          galois::gDebug(pair.first, " : ", pair.second);
          auto [_, node] = pair;
          auto& curData  = graph.getData(node, no_lockable_flag);
          if (curData.tag != BAND) {
            galois::gDebug("non-BAND");
          }
        },
        galois::no_stats(), galois::loopname("DEBUG_initBag_sanity_check"));
#endif // end of initBag sanity check;
  }

  FastMarching<CONCURRENT>(graph, initBag);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Sanity check

template <typename Graph, typename GNode = typename Graph::GraphNode>
void SanityCheck(Graph& graph) {
  galois::GReduceMax<double> max_error;

  galois::do_all(
      galois::iterate(0ul, NUM_CELLS),
      [&](GNode node) noexcept {
        if (node >= NUM_CELLS)
          return;
        auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        if (curData.solution == INF) {
          galois::gPrint("Untouched cell: ", node, "\n");
          assert(curData.solution != INF);
        }

        data_t val = 0.;
        std::array<double, 2> dims{dx, dy}; // TODO not exactly x y z order
        auto dir = graph.edge_begin(node, no_lockable_flag);
        for (double& d : dims) {
          if (dir == graph.edge_end(node, no_lockable_flag))
            break;
          GNode neighbor   = graph.getEdgeDst(dir);
          auto& first_data = graph.getData(neighbor, no_lockable_flag);
          // assert(first_data.is_ghost || first_data.tag == KNOWN);
          std::advance(dir, 1); // opposite direction of the same dimension
          assert(dir != graph.edge_end(node, no_lockable_flag));
          neighbor          = graph.getEdgeDst(dir);
          auto& second_data = graph.getData(neighbor, no_lockable_flag);
          // assert(second_data.is_ghost || second_data.tag == KNOWN);
          data_t s1 = (curData.solution - first_data.solution) / d,
                 s2 = (curData.solution - second_data.solution) / d;
          val += std::pow(std::max(0., std::max(s1, s2)), 2);
          std::advance(dir, 1);
        }
        data_t error =
            (std::sqrt(val) - (1. / curData.speed)) / (1. / curData.speed);
        max_error.update(error);
        if (error > tolerance) {
          auto [x, y] = id2xy(node);
          galois::gPrint(
              "Error bound violated at cell: ", node, " (", x, " ", y, ")",
              " with ", curData.solution.load(std::memory_order_relaxed),
              " of error ", error, " (", std::sqrt(x * x + y * y), ")\n");
          return;
        }
      },
      galois::no_stats(), galois::loopname("sanityCheck"));

  galois::gPrint("MAX ERROR: ", max_error.reduce(), "\n");
}

template <typename Graph, typename GNode = typename Graph::GraphNode>
void SanityCheck2(Graph& graph) {
  galois::do_all(
      galois::iterate(0ul, NUM_CELLS),
      [&](GNode node) noexcept {
        auto [x, y]    = id2xy(node);
        auto& solution = graph.getData(node).solution;
        assert(std::abs(solution - std::sqrt(x * x + y * y)));
      },
      galois::no_stats(), galois::loopname("sanityCheck2"));
}

template <typename GNode>
void _debug_print() {
  for (GNode i = 0; i < NUM_CELLS; i++) {
    auto [x, y] = id2xy(i);
    galois::gDebug(x, " ", y, " ", xy2id({x, y}));
  }
}

// #include "segy/SEG-YReader.h"
#include "google-segystack/segy_file.h"
#include "numpy/cnpy.h"
#include "util/output.hh"
std::unique_ptr<segystack::SegyFile> segy_ptr;

template <typename Graph>
void SetupGrids(Graph& graph) {
  if (!input_segy.empty()) {
    segy_ptr      = std::make_unique<segystack::SegyFile>(input_segy);
    std::size_t x = segy_ptr->getBinaryHeader().getValueAtOffset<uint16_t>(13);
    std::size_t y = segy_ptr->getBinaryHeader().getValueAtOffset<uint16_t>(21);

    xa = -.625 * data_t(x + 1);
    xb = .625 * data_t(x + 1);
    ya = -.625 * data_t(y + 1);
    yb = .625 * data_t(y + 1);
    SetKnobs({x, y});

    ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

    segystack::SegyFile::Trace trace;

    std::size_t read_traces = 0;
    segy_ptr->seek(read_traces);
    segy_ptr->read(trace);
    for (auto node : graph) {
      if (node >= NUM_CELLS)
        break;
      auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      curData.is_ghost = (node >= NUM_CELLS);
      curData.tag      = Tag(FAR); // TODO why compile error after import npy?
      curData.solution = INF;      // TODO ghost init?

      auto [i, j] = id2ij(node);
      if (i != read_traces) {
        assert(i == read_traces + 1);
        segy_ptr->seek(++read_traces);
        segy_ptr->read(trace);
      }
      curData.speed = trace.samples()[j] * .001;
    }
    assert(read_traces + 1 == x);
    galois::gDebug("Read ", read_traces, " traces.");
  } else if (!input_npy.empty()) {
    cnpy::NpyArray npy = cnpy::npy_load(input_npy);
    float* npy_data    = npy.data<float>();

    // make sure the loaded data matches the saved data
    assert(npy.word_size == sizeof(float) && "wrong data type");
    assert(npy.shape.size() == 2 && "Data map should be 2-D");
    std::size_t x = npy.shape[0], y = npy.shape[1];

    /* symmetric
    xa = -.625 * data_t(x + 1);
    xb = .625 * data_t(x + 1);
    ya = -.625 * data_t(y + 1);
    yb = .625 * data_t(y + 1);
    */
    xa = -1.25;
    xb = 1.25 * data_t(x);
    ya = -1.25;
    yb = 1.25 * data_t(y);
    SetKnobs({x, y});

    ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});

    for (auto node : graph) {
      if (node >= NUM_CELLS)
        break;
      auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      curData.is_ghost = (node >= NUM_CELLS);
      curData.tag      = FAR;
      curData.solution = INF; // TODO ghost init?
      curData.speed    = npy_data[node] * .001;
    }
  } else if (!input_csv.empty()) {
    std::ifstream incsv(input_csv);
    std::string line;
    if (dims.empty())
      GALOIS_DIE("Please specify the dimentions of the csv data (use -d)");
    std::size_t x = dims[0], y = dims[1];
    //// CSV WITH HEADER
    // if (std::getline(incsv, line)) {
    //   const char* header = line.c_str();
    //   char* end;
    //   x = strtoul(header, &end, 0);
    //   assert(*end == ' '); // Does it matter?
    //   y = strtoul(end + 1, &end, 0);
    //   assert(*end == '\0'); // Does it matter?
    // }

    xa = -.625 * data_t(x + 1);
    xb = .625 * data_t(x + 1);
    ya = -.625 * data_t(y + 1);
    yb = .625 * data_t(y + 1);
    SetKnobs({x, y});

    ConstructCsrGrids(graph, std::array<std::size_t, 2>{x, y});
    std::size_t node = 0;
    while (std::getline(incsv, line)) {
      const char* beg = line.c_str();
      char* end;
      bool first_time = true;
      do {
        double d = strtod(first_time ? beg : end + 1, &end);
        if (first_time)
          first_time = false;
        if (!d)
          GALOIS_DIE("In-line csv parsing failed.");
        auto& curData    = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        curData.is_ghost = (node >= NUM_CELLS);
        curData.tag      = FAR;
        curData.solution = INF; // TODO ghost init?
        curData.speed    = d * .001;
        ++node;
      } while (*end == ' ');
      if (*end != '\0')
        GALOIS_DIE("Bad EOL.");
    }
  } else {
    if (!dims.empty()) {
      dims.resize(DIM_LIMIT, 1); // padding
      SetKnobs(dims);
    } else
      GALOIS_DIE("Undefined dimensions. See help with -h.");

    ConstructCsrGrids(graph, std::array<std::size_t, 2>{nx, ny});

    galois::do_all(
        galois::iterate(graph.begin(), graph.end()),
        [&](auto node) noexcept {
          auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
          curData.is_ghost = (node >= NUM_CELLS);
          curData.tag      = FAR;
          curData.speed    = SpeedFunction(id2xy(node));
          curData.solution = INF; // TODO ghost init?
        },
        galois::no_stats(), galois::loopname("initializeCells"));
  }
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  galois::gDebug(ALGO_NAMES[algo]);

  using Graph =
      galois::graphs::LC_CSR_Graph<NodeData,
                                   void>::with_no_lockable<true>::type;
  using GNode         = Graph::GraphNode;
  using BL            = galois::InsertBag<GNode>;
  using UpdateRequest = std::pair<data_t, GNode>;
  using HeapTy        = FMMHeapWrapper<
      std::multimap<UpdateRequest::first_type, UpdateRequest::second_type>>;
  using WL = galois::InsertBag<UpdateRequest>;

  //! Grids generation and cell initialization
  // generate grids
  Graph graph;
  SetupGrids(graph);

  //! Boundary assignment and initialization
  // TODO better way for boundary settings?
  BL boundary;
  if (source_type == scatter)
    AssignBoundary(boundary);
  else
    AssignBoundary(graph, boundary);
  if (boundary.empty())
    GALOIS_DIE("Boundary not found!");

#ifndef NDEBUG
  // print boundary
  galois::gDebug("vvvvvvvv boundary vvvvvvvv");
  for (GNode b : boundary) {
    auto [x, y] = id2xy(b);
    galois::gDebug(b, " (", x, ", ", y, ")");
  }
  galois::gDebug("^^^^^^^^ boundary ^^^^^^^^");
#endif

  initBoundary(graph, boundary);

  //! run algo
  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case serial:
    runAlgo<false, HeapTy>(graph, boundary);
    break;
  case parallel:
    runAlgo<true, WL>(graph, boundary);
    break;
  default:
    std::abort();
  }

  Tmain.stop();

  SanityCheck(graph);
  // SanityCheck2(graph);
  if (!output_npy.empty())
    DumpToNpy(graph);
  else if (!output_csv.empty())
    DumpToCsv(graph);

  return 0;
}
