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
#ifndef GALOIS_FMM_DIST_PUSH
#define GALOIS_FMM_DIST_PUSH
#endif

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/ArrayWrapper.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "DistBench/Start.h"

#ifdef GALOIS_ENABLE_VTUNE
#include "galois/runtime/Profile.h"
#endif

constexpr static char const* name = "Fast Marching Method";
constexpr static char const* desc =
    "Eikonal equation solver "
    "(https://en.wikipedia.org/wiki/Fast_marching_method)";
constexpr static char const* url         = "";
constexpr static char const* REGION_NAME = "FMM";

#include "fastmarchingmethod.h"
#define DIM_LIMIT 2 // 2-D specific

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
static llvm::cl::opt<double> rounding_scale{
    "rf", llvm::cl::desc("round-off factor for OBIM"), llvm::cl::init(1.),
    llvm::cl::cat(catAlgo)};
static llvm::cl::opt<double> tolerance{
    "e",
    llvm::cl::desc("Final error bound for non-strict differencing operator"),
    llvm::cl::init(1.e-14), llvm::cl::cat(catAlgo)};
static llvm::cl::opt<bool> strict{
    "strict",
    llvm::cl::desc(
        "Force non-increasing update to mitigate catastrophic cancellation"),
    llvm::cl::cat(catAlgo)};

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
static llvm::cl::opt<std::string> verify_npy(
    "vnpy",
    llvm::cl::desc("Canonical results for verification in a npy format file"),
    llvm::cl::init(""), llvm::cl::cat(catInput));
namespace internal {
template <typename T>
struct StrConv;
template <>
struct StrConv<std::size_t> {
  auto operator()(const char* str, char** endptr, int base) {
    return strtoul(str, endptr, base);
  }
};
template <>
struct StrConv<double> {
  auto operator()(const char* str, char** endptr, int) {
    return strtod(str, endptr);
  }
};
} // namespace internal
template <typename NumTy, int FIXED_SIZE = 0>
struct NumVecParser : public llvm::cl::parser<std::vector<NumTy>> {
  template <typename... Args>
  NumVecParser(Args&... args) : llvm::cl::parser<std::vector<NumTy>>(args...) {}
  internal::StrConv<NumTy> strconv;
  // parse - Return true on error.
  bool parse(llvm::cl::Option& O, llvm::StringRef ArgName,
             const std::string& ArgValue, std::vector<NumTy>& Val) {
    const char* beg = ArgValue.c_str();
    char* end;

    do {
      NumTy d = strconv(Val.empty() ? beg : end + 1, &end, 0);
      if (!d)
        return O.error("Invalid option value '" + ArgName + "=" + ArgValue +
                       "': should be comma-separated unsigned integers");
      Val.push_back(d);
    } while (*end == ',');
    if (*end != '\0')
      return O.error("Invalid option value '" + ArgName + "=" + ArgValue +
                     "': should be comma-separated unsigned integers");
    if (FIXED_SIZE && Val.size() != FIXED_SIZE)
      return O.error(ArgName + "=" + ArgValue + ": expect " +
                     std::to_string(FIXED_SIZE) + " numbers but get " +
                     std::to_string(Val.size()));
    return false;
  }
};
static llvm::cl::opt<std::vector<std::size_t>, false,
                     NumVecParser<std::size_t, DIM_LIMIT>>
    domain_shape(
        "ij", llvm::cl::value_desc("nrows,ncols"),
        llvm::cl::desc("Size of each dimensions as a comma-separated array "
                       "(support up to 2-D)"),
        llvm::cl::cat(catInput));
static llvm::cl::opt<double> speed_factor{
    "sf", llvm::cl::desc("speed factor multiplied to speed value"),
    llvm::cl::init(1.), llvm::cl::cat(catInput)};

static llvm::cl::OptionCategory catOutput("3. Output Options");
static llvm::cl::opt<std::string>
    output_csv("ocsv", llvm::cl::desc("Export results to a csv format file"),
               llvm::cl::init(""), llvm::cl::cat(catOutput));
static llvm::cl::opt<std::string>
    output_npy("onpy", llvm::cl::desc("Export results to a npy format file"),
               llvm::cl::init(""), llvm::cl::cat(catOutput));

static llvm::cl::OptionCategory catDisc("4. Discretization options");
static llvm::cl::opt<std::vector<double>, false,
                     NumVecParser<double, DIM_LIMIT>>
    steps("h", llvm::cl::value_desc("dx,dy"),
          llvm::cl::desc("Spacing between discrete samples of each dimensions; "
                         "type as a comma-separated pair"),
          llvm::cl::init(std::vector<double>{1., 1.}), llvm::cl::cat(catDisc));
static llvm::cl::opt<std::vector<double>, false,
                     NumVecParser<double, DIM_LIMIT>>
    domain_start("oxy", llvm::cl::value_desc("x0,y0"),
                 llvm::cl::desc("Coordinate of cell [0, 0] "
                                "<comma-separated array, default (0., 0.)>"),
                 llvm::cl::init(std::vector<double>{0., 0.}),
                 llvm::cl::cat(catDisc));

static std::size_t nx, ny, NUM_CELLS;
static data_t dx, dy, xa, xb, ya, yb;

///////////////////////////////////////////////////////////////////////////////

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!
using sync_array_t =
    galois::CopyableArray<galois::CopyableAtomic<data_t>, DIM_LIMIT>;
struct NodeData {
  data_t speed; // read only
  sync_array_t upwind_solution;
  data_t solution;
};
galois::DynamicBitSet bitset_upwind_solution;

using Graph = galois::graphs::DistGraph<NodeData, void>;
using GNode = Graph::GraphNode;
using BL    = galois::InsertBag<GNode>;
std::unique_ptr<galois::graphs::GluonSubstrate<Graph>> syncSubstrate;

#include "distributed/DgIO.h"
#include "distributed/fmm_sync.h"

#include "util/input.hh"

template <typename Graph, typename BL,
          typename GNode = typename Graph::GraphNode,
          typename T     = typename BL::value_type>
void assignBoundary(Graph& graph, BL& boundary) {
  if (source_type == scatter) {
    GNode g_n = xy2id({0., 0.});
    if (graph.isLocal(g_n))
      boundary.push(graph.getLID(g_n));
    else
      galois::gDebug("not on this host");
  } else {
    const auto& allNodes = graph.allNodesRange();
    galois::do_all(
        galois::iterate(allNodes.begin(), allNodes.end()),
        [&](T node) noexcept {
          if (graph.getGID(node) >= NUM_CELLS)
            return;

          auto [x, y] = id2xy(graph.getGID(node));
          if (NonNegativeRegion(data2d_t{x, y})) {
            if (!NonNegativeRegion(data2d_t{x + dx, y}) ||
                !NonNegativeRegion(data2d_t{x - dx, y}) ||
                !NonNegativeRegion(data2d_t{x, y + dy}) ||
                !NonNegativeRegion(data2d_t{x, y - dy})) {
              boundary.push(node);
            }
          }
        },
        galois::loopname("assignBoundary"));
  }
}

/////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

static void initCells(Graph& graph) {
  const auto& all_nodes = graph.allNodesRange();
  galois::do_all(
      galois::iterate(all_nodes.begin(), all_nodes.end()),
      [&](GNode node) noexcept {
        auto& node_data    = graph.getData(node);
        node_data.solution = INF;
        assert(graph.getGID(node) >= NUM_CELLS || node_data.speed > 0);
        for (auto& i : node_data.upwind_solution) {
          i = INF; // TODO store
        };
      },
      galois::no_stats(),
      galois::loopname(
          syncSubstrate->get_run_identifier("initializeCells").c_str()));
}

static void initBoundary(Graph& graph, BL& local_boundary) {
  galois::DGAccumulator<uint32_t> busy;
  busy.reset();
  if (!local_boundary.empty()) {
    busy += 1;
#ifndef NDEBUG
    // print local_boundary
    for (GNode b : local_boundary) {
      auto [ii, jj] = id2ij(graph.getGID(b));
      DGDEBUG("local_boundary: ", b, "(g", graph.getGID(b),
              (b < graph.numMasters() ? "M" : "m"), ") (", ii, " ", jj,
              ") with ", graph.getData(b).solution);
    }
#endif
    galois::do_all(
        galois::iterate(local_boundary.begin(), local_boundary.end()),
        [&](GNode b) noexcept {
          auto& boundary_data    = graph.getData(b);
          boundary_data.solution = BoundaryCondition(id2xy(graph.getGID(b)));
        },
        galois::no_stats(),
        galois::loopname(
            syncSubstrate->get_run_identifier("initializeBoundary").c_str()));
  } else {
    DGDEBUG("No local_boundary element");
  }
  assert(busy.reduce() && "Boundary not defined!");
}

////////////////////////////////////////////////////////////////////////////////

template <typename NodeData>
data_t quickUpdate(NodeData& my_data) {
  const auto f = my_data.speed;
  assert(dx == dy);
  auto h = dx;
  /*
   * General form of the differencing scheme on uniform grids (same spacing h
   * for all dimensions):
   *
   * Sigma<i=1...N>( (t-t_i)^2 ) = (h / f)^2
   *
   * t: arrival time (to be solved)
   * h: unit step
   * f: speed
   *
   * Particular solutions:
   *
   * 1st order:
   * m0 = h / f
   * t = t_1 + m0
   *
   * 2nd order:
   * m1 = sqrt2 * m0
   * m2 = t-1 - t_2
   * t = (t_1 + t_2 + sqrt{(m1 + m2) * (m1 - m2)}) / 2
   *
   * Particular solutions for non-uniform grids:
   *
   * 1st order:
   * t = t_1 + h_1 / f
   *
   * 2nd order:
   * m0 = h_1^2 + h_2^2
   * m1 = m0 / f^2
   * m2 = t_1 - t_2
   * p0 = h_1 / m0
   * p1 = h_1 * p0
   * p12 = h_2 * p0
   * p2 = h_2 * h_2 / m0
   * t = p2 * t_1 + p1 * t_2 + p3 * sqrt{m1 - m2^2}
   */
  data_t u_V = my_data.upwind_solution[0];
  data_t u_H = my_data.upwind_solution[1];
  data_t div = h / f;
  if (std::isinf(u_H) || std::isinf(u_V) || std::abs(u_H - u_V) >= div) {
    return std::min(u_H, u_V) + div;
  }
  // Use this particular form of the differencing scheme to mitigate
  // precision loss from catastrophic cancellation inside the square root.
  // The loss of precision breaks the monotonicity guarantees of the
  // differencing operator. This mitigatest that issue somewhat.
  static double sqrt2 = std::sqrt(2);
  data_t s2div = sqrt2 * div, dif = u_H - u_V;
  return .5 * (u_H + u_V + std::sqrt((s2div + dif) * (s2div - dif)));
}

template <typename Graph, typename It = typename Graph::edge_iterator>
bool pushUpdate(Graph& graph, data_t& up_sln, It dir) {

  bool didWork = false;

  for (int i = 0; i < DIM_LIMIT; ++i) {
    auto uni_push = [&](It ei) {
      GNode dst = graph.getEdgeDst(ei);
      if (graph.getGID(dst) >= NUM_CELLS)
        return;
      auto& dst_data = graph.getData(dst);
      if (up_sln >= dst_data.solution)
        return;
      if (auto& us = dst_data.upwind_solution[i];
          up_sln < galois::atomicMin(us, up_sln)) {
        bitset_upwind_solution.set(dst);
        didWork = true;
      }
#ifndef NDEBUG
      if (dst == 1104) {
        auto [ii, jj]    = id2ij(graph.getGID(dst));
        auto [unwrapped] = dst_data.upwind_solution;
        auto [a, b]      = unwrapped;
        DGDEBUG("update ", dst, " (g", graph.getGID(dst),
                (dst < graph.numMasters() ? "M" : "m"), ") (", ii, " ", jj,
                ") upwind_solution: ", a, " ", b);
      }
#endif
    };

    // check one neighbor
    uni_push(dir++);
    // check the other neighbor
    uni_push(dir++);
  }
  return didWork;
}
////////////////////////////////////////////////////////////////////////////////
// Solver

template <typename NodeData>
data_t solveQuadraticPush(NodeData& my_data) {
  const auto& upwind_sln = my_data.upwind_solution;
  data_t sln             = my_data.solution;
  const data_t speed     = my_data.speed;

  std::array<std::pair<data_t, data_t>, DIM_LIMIT> sln_delta = {
      std::make_pair(sln, dx), std::make_pair(sln, dy)};

  int non_zero_counter = 0;

  for (int i = 0; i < DIM_LIMIT; i++) {
    auto& [s, d] = sln_delta[i];
    if (data_t us = upwind_sln[i]; us < s) {
      s = us;
      non_zero_counter++;
    }
  }
  // local computation, nothing about graph
  if (non_zero_counter == 0)
    return sln;
  // DGDEBUG("solveQuadratic: #upwind_dirs: ", non_zero_counter);

  std::sort(sln_delta.begin(), sln_delta.end(),
            [&](std::pair<data_t, data_t>& a, std::pair<data_t, data_t>& b) {
              return a.first < b.first;
            });
  auto p = sln_delta.begin();
  data_t a(0.), b_(0.), c_(0.), f(1. / (speed * speed));
  do {
    const auto& [s, d] = *p++;
    // DGDEBUG(s, " ", d);
    // Arrival time may be updated in previous rounds
    // in which case remaining upwind dirs become invalid
    if (sln < s)
      break;

    double temp = 1. / (d * d);
    a += temp;
    temp *= s;
    b_ += temp;
    temp *= s;
    c_ += temp;
    data_t b = -2. * b_, c = c_ - f;
    DGDEBUG("tabc: ", temp, " ", a, " ", b, " ", c);

    double del = b * b - (4. * a * c);
    DGDEBUG(a, " ", b, " ", c, " del=", del);
    if (del >= 0) {
      double new_sln = (-b + std::sqrt(del)) / (2. * a);
      galois::gDebug("new solution: ", new_sln);
      if (new_sln > s) { // conform causality
        sln = std::min(sln, new_sln);
      }
    }
  } while (--non_zero_counter);
  return sln;
}

////////////////////////////////////////////////////////////////////////////////
// first iteration

static void FirstIteration(Graph& graph, BL& boundary) {
  galois::do_all(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](GNode b) noexcept {
        if (graph.getGID(b) < NUM_CELLS) {
          auto& b_data = graph.getData(b);
#ifndef NDEBUG
          auto [i, j] = id2ij(graph.getGID(b));
          DGDEBUG("FirstItr: ", b, " (g", graph.getGID(b),
                  (b < graph.numMasters() ? "M" : "m"), ") (", i, " ", j,
                  ") sln:", b_data.solution);
#endif
          auto dir = graph.edge_begin(b);
          if (dir >= graph.edge_end(b))
            return;
          pushUpdate(graph, b_data.solution, dir);
        }
      },
      galois::loopname("FirstIteration"));

  syncSubstrate
      ->sync<writeDestination, readSource, Reduce_pair_wise_min_upwind_solution,
             Bitset_upwind_solution>("FirstIteration");
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// FMM

void FastMarching(Graph& graph) {
  using DGTerminatorDetector = galois::DGAccumulator<uint32_t>;
  DGTerminatorDetector more_work;
  unsigned _round_counter = 0;

  // const auto& nodes_with_edges = graph.allNodesWithEdgesRange();
  const auto& all_nodes = graph.allNodesRange();

#ifdef GALOIS_ENABLE_VTUNE
  galois::runtime::profileVtune(
      [&]() {
#endif
        do {
#ifndef NDEBUG
          sleep(5); // Debug pause
          galois::gDebug("\n********\n");
#endif
          syncSubstrate->set_num_round(_round_counter);
          more_work.reset();
          galois::do_all(
              galois::iterate(all_nodes),
              [&](GNode node) {
                if (graph.getGID(node) >= NUM_CELLS)
                  return;
                auto& node_data = graph.getData(node);
                data_t new_val  = quickUpdate(node_data);
                if (new_val < galois::min(node_data.solution, new_val)) {
#ifndef NDEBUG
                  if (node == 1104) {
                    auto [i, j] = id2ij(graph.getGID(node));
                    DGDEBUG("Processing ", node, " (g", graph.getGID(node),
                            (node < graph.numMasters() ? "M" : "m"), ") (", i,
                            " ", j, ") sln:", node_data.solution);
                  }
#endif
                  auto dir = graph.edge_begin(node);
                  if (dir >= graph.edge_end(node))
                    return;
                  if (pushUpdate(graph, node_data.solution, dir))
                    more_work += 1;
                }
              },
              galois::no_stats(),
              galois::steal(), // galois::wl<OBIM>(Indexer),
              galois::loopname(
                  syncSubstrate->get_run_identifier("Push").c_str()));

          // sleep(5);
          syncSubstrate->sync<writeDestination, readSource,
                              Reduce_pair_wise_min_upwind_solution,
                              Bitset_upwind_solution>("FastMarching");

          galois::runtime::reportStat_Tsum(
              REGION_NAME,
              "NumWorkItems_" + (syncSubstrate->get_run_identifier()),
              (uint32_t)more_work.read_local());
          ++_round_counter;
        } while (more_work.reduce(syncSubstrate->get_run_identifier().c_str()));
#ifdef GALOIS_ENABLE_VTUNE
      },
      "FMM_VTune");
#endif
}

void FastMarchingOld(Graph& graph) {
  using DGTerminatorDetector = galois::DGAccumulator<uint32_t>;
  DGTerminatorDetector more_work;
  unsigned _round_counter = 0;

  // const auto& nodes_with_edges = graph.allNodesWithEdgesRange();
  const auto& all_nodes = graph.allNodesRange();

#ifdef GALOIS_ENABLE_VTUNE
  galois::runtime::profileVtune(
      [&]() {
#endif
        do {
#ifndef NDEBUG
          sleep(5); // Debug pause
          galois::gDebug("\n********\n");
#endif
          syncSubstrate->set_num_round(_round_counter);
          more_work.reset();
          galois::do_all(
              galois::iterate(all_nodes),
              [&](GNode node) {
                if (graph.getGID(node) >= NUM_CELLS)
                  return;
                auto& node_data = graph.getData(node);
                data_t sln_temp = quickUpdate(node_data);
                if (sln_temp < galois::min(node_data.solution, sln_temp)) {
#ifndef NDEBUG
                  if (node == 1104) {
                    auto [i, j] = id2ij(graph.getGID(node));
                    DGDEBUG("Processing ", node, " (g", graph.getGID(node),
                            (node < graph.numMasters() ? "M" : "m"), ") (", i,
                            " ", j, ") sln:", node_data.solution);
                  }
#endif
                  auto dir = graph.edge_begin(node);
                  if (dir >= graph.edge_end(node))
                    return;
                  if (pushUpdate(graph, node_data.solution, dir))
                    more_work += 1;
                }
              },
              galois::no_stats(),
              galois::steal(), // galois::wl<OBIM>(Indexer),
              galois::loopname(
                  syncSubstrate->get_run_identifier("Push").c_str()));

          // sleep(5);
          syncSubstrate->sync<writeDestination, readSource,
                              Reduce_pair_wise_min_upwind_solution,
                              Bitset_upwind_solution>("FastMarching");

          galois::runtime::reportStat_Tsum(
              REGION_NAME,
              "NumWorkItems_" + (syncSubstrate->get_run_identifier()),
              (uint32_t)more_work.read_local());
          ++_round_counter;
        } while (more_work.reduce(syncSubstrate->get_run_identifier().c_str()));
#ifdef GALOIS_ENABLE_VTUNE
      },
      "FMM_VTune");
#endif
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Sanity check

void SanityCheck(Graph& graph) {
  galois::DGReduceMax<double> max_error;
  galois::DGReduceMax<double> max_val;

  const auto& masterNodes = graph.masterNodesRange();
  galois::do_all(
      galois::iterate(masterNodes.begin(), masterNodes.end()),
      [&](GNode node) noexcept {
        if (graph.getGID(node) >= NUM_CELLS)
          return;
        auto& my_data = graph.getData(node);
        if (my_data.solution == 0.) { // TODO: identify sources?
          return;
        }
        if (my_data.solution == INF) {
          auto [ii, jj] = id2ij(graph.getGID(node));
          galois::gPrint("Untouched cell: ", node, " (g", graph.getGID(node),
                         ") ", (node < graph.numMasters() ? "M" : "m"), " (",
                         ii, " ", jj, ")\n");
          return;
        }
        data_t old_val = my_data.solution;
        max_val.update(old_val);
        data_t new_val = quickUpdate(my_data);
        if (new_val != old_val) {
          data_t error = std::abs(new_val - old_val) / std::abs(old_val);
          max_error.update(error);
          if (error > tolerance) {
            auto [ii, jj] = id2ij(graph.getGID(node));
            galois::gPrint("Error bound violated at cell ", node, " (", ii, " ",
                           jj, "): old_val=", old_val, " new_val=", new_val,
                           " error=", error, "\n");
          }
        }
      },
      galois::no_stats(), galois::loopname("sanityCheck"));

  auto mv = max_val.reduce();
  DGPRINT("max arrival time: ", mv, "\n");
  auto me = max_error.reduce();
  DGPRINT("max err: ", me, "\n");
  galois::runtime::reportStat_Single(std::string(REGION_NAME), "MaxError", me);
}

template <typename Graph, typename GNode = typename Graph::GraphNode>
void SanityCheck2(Graph& graph) {
  galois::do_all(
      galois::iterate(0ul, NUM_CELLS),
      [&](GNode node) noexcept {
        auto [x, y]    = id2xy(graph.getGID(node));
        auto& solution = graph.getData(node).solution;
        assert(std::abs(solution - std::sqrt(x * x + y * y)));
      },
      galois::no_stats(), galois::loopname("sanityCheck2"));
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) noexcept {
  galois::DistMemSys galois_system;
  DistBenchStart(argc, argv, name, desc, url);

  galois::gDebug(ALGO_NAMES[algo]);

  galois::StatTimer Ttotal("TimerTotal");
  Ttotal.start();

  // generate grids
  std::unique_ptr<Graph> graph;
  std::tie(graph, syncSubstrate) = distGraphInitialization<NodeData, void>();

  // _debug_print();

  // initialize all cells
  setupGrids(*graph);
  initCells(*graph);
  galois::runtime::getHostBarrier().wait();

  // TODO better way for boundary settings?
  BL boundary;
  assignBoundary(*graph, boundary);

  bitset_upwind_solution.resize(graph->size());
  galois::runtime::getHostBarrier().wait();

  for (int run = 0; run < numRuns; ++run) {
    DGPRINT("Run ", run, " started\n");
    std::string tn = "Timer_" + std::to_string(run);
    galois::StatTimer Tmain(tn.c_str());

    initBoundary(*graph, boundary);

    Tmain.start();

    FirstIteration(*graph, boundary);
    FastMarching(*graph);

    Tmain.stop();

    galois::runtime::getHostBarrier().wait();
    SanityCheck(*graph);
    // SanityCheck2(graph);

    if ((run + 1) != numRuns) {
      bitset_upwind_solution.reset();

      initCells(*graph);
      galois::runtime::getHostBarrier().wait();
    }
  }

  Ttotal.stop();
  return 0;
}
