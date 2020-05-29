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

#include "galois/runtime/Profile.h"

constexpr static char const* name        = "FMM";
constexpr static char const* desc        = "fmm";
constexpr static char const* url         = "";
constexpr static char const* REGION_NAME = "FMM";

using CoordTy = double;
using SlnTy   = double;

enum Algo { serial = 0, parallel, partition, bipartiteSync };
enum Source { scatter = 0, analytical };

const char* const ALGO_NAMES[] = {"serial", "parallel", "partition",
                                  "bipartiteSync"};

static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::desc("Choose an algorithm:"),
         llvm::cl::values(clEnumVal(serial, "serial"),
                          clEnumVal(parallel, "parallel"),
                          clEnumVal(partition, "partition"),
                          clEnumVal(bipartiteSync, "bipartiteSync")),
         llvm::cl::init(parallel));
static llvm::cl::opt<Source> source_type(
    "source", llvm::cl::desc("Choose an sourceType:"),
    llvm::cl::values(clEnumVal(scatter, "a set of discretized points"),
                     clEnumVal(analytical, "boundary in a analytical form")),
    llvm::cl::init(analytical));
static llvm::cl::opt<unsigned long long> nh{
    "nh", llvm::cl::desc("number of cells in ALL direction"),
    llvm::cl::init(0u)};
static llvm::cl::opt<unsigned long long> _nx{
    "nx", llvm::cl::desc("number of cells in x direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> _ny{
    "ny", llvm::cl::desc("number of cells in y direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> _nz{
    "nz", llvm::cl::desc("number of cells in z direction"),
    llvm::cl::init(10u)};
static std::size_t nx, ny, nz;
static llvm::cl::opt<unsigned> RF{
    "rf", llvm::cl::desc("round-off factor for OBIM"), llvm::cl::init(0u)};
// static llvm::cl::opt<CoordTy> num_groups{
//     "num_groups", llvm::cl::desc("number of frequency groups"),
//     llvm::cl::init(4u)};
// static llvm::cl::opt<CoordTy> num_vert_directions{
//     "num_vert_directions", llvm::cl::desc("number of vertical directions"),
//     llvm::cl::init(16u)};
// static llvm::cl::opt<CoordTy> num_horiz_directions{
//     "num_horiz_directions", llvm::cl::desc("number of horizontal
//     directions."), llvm::cl::init(32u)};
// static llvm::cl::opt<unsigned long long> num_iters{
//    "num_iters", llvm::cl::desc("number of iterations"), llvm::cl::init(10u)};
// static llvm::cl::opt<double> pulse_strength{
//    "pulse_strength", llvm::cl::desc("radiation pulse strength"),
//    llvm::cl::init(1.)};
// static llvm::cl::opt<double> absorption_coef{
//     "absorption_coef",
//     llvm::cl::desc("Absorption coefficient (between 0 and 1), absorption and
//     "
//                    "scattering must sum to less than 1."),
//     llvm::cl::init(.01)};
// static llvm::cl::opt<double> scattering_coef{
//     "scattering_coef",
//     llvm::cl::desc("Scattering coefficient (between 0 and 1), absorption and
//     "
//                    "scattering must sum to less than 1."),
//     llvm::cl::init(.25)};
// static llvm::cl::opt<bool> print_convergence{
//     "print_convergence",
//     llvm::cl::desc("Print the max change in amount of scattering at a given "
//                    "each iteration."),
//     llvm::cl::init(false)};
// static llvm::cl::opt<std::string> scattering_outfile{
//     "scattering_outfile",
//     llvm::cl::desc(
//         "Text file name to use to write final scattering term values "
//         "after each step."),
//     llvm::cl::init("")};

static constexpr double xa = -.5, xb = .5;
static constexpr double ya = -.5, yb = .5;
static constexpr double za = -.5, zb = .5;

static std::size_t NUM_CELLS;
static CoordTy dx, dy, dz;
#include "distributed/DgIO.h"
void global_config() {
  // configure global variables
  if (nh)
    nx = ny = nz = nh;
  else {
    nx = _nx;
    ny = _ny;
    nz = _nz;
  }
  NUM_CELLS = nh ? nh * nh * nh : nx * ny * nz;
  dx        = (xb - xa) / CoordTy(nx + 1);
  dy        = (yb - ya) / CoordTy(ny + 1);
  dz        = (zb - za) / CoordTy(nz + 1);
  if (!RF)
    RF = 1 / std::min({dx, dy, dz}, std::less<CoordTy>{});
  DGDEBUG(nx, " - ", ny, " - ", nz);
  DGDEBUG(dx, " - ", dy, " - ", dz);
  DGDEBUG("RF: ", RF);
}

///////////////////////////////////////////////////////////////////////////////

constexpr SlnTy INF = std::numeric_limits<double>::max();

#include "fastmarchingmethod.h"

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!
using sync_double3d_t =
    galois::CopyableArray<galois::CopyableAtomic<double>, 3>;
struct NodeData {
  double speed; // read only
  sync_double3d_t upwind_solution;
  double solution;
};
galois::DynamicBitSet bitset_upwind_solution;

using Graph = galois::graphs::DistGraph<NodeData, void>;
using GNode = Graph::GraphNode;
using BL    = galois::InsertBag<GNode>;
galois::graphs::GluonSubstrate<Graph>* syncSubstrate;
constexpr auto atomic_order = std::memory_order_relaxed;

#include "distributed/fmm_sync.h"
#include "structured/grids.h"
#include "structured/utils.h"

template <typename Graph, typename BL,
          typename GNode = typename Graph::GraphNode,
          typename T     = typename BL::value_type>
void AssignBoundary(Graph& graph, BL& boundary) {
  const auto& allNodes = graph.allNodesRange();
  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](T node) noexcept {
        if (graph.getGID(node) >= NUM_CELLS)
          return;

        auto [x, y, z] = getCoord(graph.getGID(node));
        if (NonNegativeRegion(double3d_t{x, y, z})) {
          if (!NonNegativeRegion(double3d_t{x + dx, y, z}) ||
              !NonNegativeRegion(double3d_t{x - dx, y, z}) ||
              !NonNegativeRegion(double3d_t{x, y + dy, z}) ||
              !NonNegativeRegion(double3d_t{x, y - dy, z}) ||
              !NonNegativeRegion(double3d_t{x, y, z + dz}) ||
              !NonNegativeRegion(double3d_t{x, y, z - dz})) {
            boundary.push(node);
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

template <typename GNode, typename Graph, typename BL>
void AssignBoundary(Graph& graph, BL& boundary) {
  // #ifndef NDEBUG
  //   GNode n = getNodeID({0., 0., 0.});
  //   auto c = getCoord(graph.getGID(n));
  //   galois::gDebug(n, " (", c[0], " ", c[1], " ", c[2], ")");
  // #endif

  GNode g_n = getNodeID<GNode>({0., 0., 0.});
  if (graph.isLocal(g_n))
    boundary.push(graph.getLID(g_n));
  else
    galois::gDebug("not on this host");
}

/////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

static void initCells(Graph& graph) {
  const auto& all_nodes = graph.allNodesRange();
  galois::do_all(
      galois::iterate(all_nodes.begin(), all_nodes.end()),
      [&](GNode node) noexcept {
        auto& node_data    = graph.getData(node);
        node_data.speed    = SpeedFunction(getCoord(graph.getGID(node)));
        node_data.solution = INF;
        for (auto& i : node_data.upwind_solution) {
          i = INF;
        };
      },
      galois::no_stats(),
      galois::loopname(
          syncSubstrate->get_run_identifier("initializeCells").c_str()));
}

static void initBoundary(Graph& graph, BL& boundary) {
  galois::do_all(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](GNode b) noexcept {
        auto& boundary_data    = graph.getData(b);
        boundary_data.solution = BoundaryCondition(getCoord(graph.getGID(b)));
      },
      galois::no_stats(),
      galois::loopname(
          syncSubstrate->get_run_identifier("initializeBoundary").c_str()));
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Solver

template <typename Graph, typename GNode = typename Graph::GraphNode>
auto checkDirection(Graph& graph, GNode active_node, double center_sln,
                    typename Graph::edge_iterator dir) {
#ifndef NDEBUG
  if (dir >= graph.edge_end(active_node)) {
    galois::gDebug(
        active_node, " (g", graph.getGID(active_node), ") ",
        (graph.getGID(active_node) >= NUM_CELLS ? "ghost" : "non-ghost"), " ",
        std::distance(graph.edge_begin(active_node),
                      graph.edge_end(active_node)));
    GALOIS_DIE("invalid direction");
  }
#endif
  SlnTy sln        = center_sln;
  GNode upwind     = active_node;
  GNode neighbor   = graph.getEdgeDst(dir);
  auto& first_data = graph.getData(neighbor);
  // DGDEBUG("Check neighbor ", neighbor, "(g", graph.getGID(neighbor),
  //         (neighbor < graph.numMasters() ? "M" : "m"), ", tag",
  //         (int)first_data.tag, ") of ", active_node, "(g",
  //         graph.getGID(active_node),
  //         (active_node < graph.numMasters() ? "M" : "m"), ")");
  // if (first_data.tag == KNOWN)
  if (first_data.solution < sln) {
    sln    = first_data.solution;
    upwind = neighbor;
  }
  std::advance(dir, 1); // opposite direction of the same dimension
  if (dir != graph.edge_end(active_node)) {
    neighbor          = graph.getEdgeDst(dir);
    auto& second_data = graph.getData(neighbor);
    // DGDEBUG("Check neighbor ", neighbor, "(tag", (int)first_data.tag, ") of
    // ",
    //         active_node);
    // if (second_data.tag == KNOWN)
    if (second_data.solution < sln) {
      sln    = second_data.solution;
      upwind = neighbor;
    }
  }
  if (upwind == active_node)
    return std::make_pair(0., 0.);
  return std::make_pair(sln, dx);
}

template <typename Array>
double solveQuadraticPush(Array& upwind_sln, double sln, const double speed) {
  // TODO parameterize dimension 3
  std::array<std::pair<double, double>, 3> sln_delta{
      std::make_pair(0., dx), std::make_pair(0., dy), std::make_pair(0., dz)};
  int non_zero_counter = 0;

  for (int i = 0; i < 3; i++) {
    double& s = sln_delta[i].first;
    double& d = sln_delta[i].second;
    double si = upwind_sln[i];
    if (si < sln) {
      s = si;
      non_zero_counter++;
    } else {
      // s = 0.; // already there
      d = 0.;
    }
  }

  if (non_zero_counter == 0)
    return INF; // mirror nodes with no edges
  while (non_zero_counter) {
    auto max_s_d_it = std::max_element(
        sln_delta.begin(), sln_delta.end(),
        [&](std::pair<double, double>& a, std::pair<double, double>& b) {
          return a.first < b.first;
        });
    double a(0.), b(0.), c(0.);
    for (const auto& p : sln_delta) {
      const double &s = p.first, d = p.second;
      // DGDEBUG(s, " ", d);
      double temp = (d == 0. ? 0. : (1. / (d * d)));
      a += temp;
      temp *= s;
      b += temp;
      temp *= s;
      c += temp;
      // DGDEBUG("tabc: ", temp, " ", a, " ", b, " ", c);
    }
    b *= -2.;
    c -= (1. / (speed * speed));
    double del = b * b - (4. * a * c);
    // DGDEBUG(a, " ", b, " ", c, " del=", del);
    if (del >= 0) {
      double new_sln = (-b + std::sqrt(del)) / (2. * a);
      // DGDEBUG("new solution: ", new_sln);
      if (new_sln > max_s_d_it->first) {
        // DGDEBUG("AC: ", non_zero_counter, ", ", sln, "->", new_sln, ">",
        //         max_s_d_it->first);
        // assert(new_sln <= sln);  // false assertion
        sln = std::min(sln, new_sln);
      } else {
        // DGDEBUG("RJ: ", non_zero_counter, ", ", sln, "->", new_sln, ">",
        //         max_s_d_it->first);
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

////////////////////////////////////////////////////////////////////////////////
// first iteration

static void FirstIteration(Graph& graph, BL& boundary) {
  galois::do_all(
      galois::iterate(boundary.begin(), boundary.end()),
      [&](GNode b) noexcept {
        if (graph.getGID(b) < NUM_CELLS) {
          auto& b_data = graph.getData(b);
#ifndef NDEBUG
          auto [i, j, k] = getPos(graph.getGID(b));
          DGDEBUG("FirstItr: ", b, " (g", graph.getGID(b),
                  (b < graph.numMasters() ? "M" : "m"), ") (", i, " ", j, " ",
                  k, ") sln:", b_data.solution);
#endif
          auto dir = graph.edge_begin(b);
          for (int i = 0; i < 3; ++i) {
            if (dir >= graph.edge_end(b))
              break;

            auto unidir_update = [&](GNode dst) {
              if (graph.getGID(dst) < NUM_CELLS) {
                auto& dst_data = graph.getData(dst);
                if (b_data.solution < dst_data.solution) {
                  auto& us = dst_data.upwind_solution[i];
                  if (b_data.solution < galois::atomicMin(us, b_data.solution))
                    bitset_upwind_solution.set(dst);
#ifndef NDEBUG
                  auto [ii, jj, kk] = getPos(graph.getGID(dst));
                  auto [unwrapped]  = dst_data.upwind_solution;
                  auto [a, b, c]    = unwrapped;
                  DGDEBUG("update ", dst, " (g", graph.getGID(dst),
                          (dst < graph.numMasters() ? "M" : "m"), ") (", ii,
                          " ", jj, " ", kk, ") upwind_solution: ", a, " ", b,
                          " ", c);
#endif
                }
              }
            };

            GNode dst = graph.getEdgeDst(dir);
            unidir_update(dst);
            std::advance(dir, 1); // another direction
            dst = graph.getEdgeDst(dir);
            unidir_update(dst);

            std::advance(dir, 1);
          }
        }
      },
      galois::loopname("FirstIteration"));

  syncSubstrate->sync<writeAny, readAny, Reduce_pair_wise_min_upwind_solution,
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

  galois::runtime::profileVtune(
      [&]() {
        do {
          syncSubstrate->set_num_round(_round_counter);
          more_work.reset();
          galois::do_all(
              galois::iterate(all_nodes),
              [&](GNode node) {
                if (graph.getGID(node) < NUM_CELLS) {
                  auto& node_data = graph.getData(node);
                  double sln_temp =
                      solveQuadraticPush(node_data.upwind_solution,
                                         node_data.solution, node_data.speed);
                  if (sln_temp < galois::min(node_data.solution, sln_temp)) {
                    more_work += 1;
#ifndef NDEBUG
                    if (node == 1104) {
                      auto [i, j, k] = getPos(graph.getGID(node));
                      DGDEBUG("Processing ", node, " (g", graph.getGID(node),
                              (node < graph.numMasters() ? "M" : "m"), ") (", i,
                              " ", j, " ", k, ") sln:", node_data.solution);
                    }
#endif

                    auto dir = graph.edge_begin(node);
                    for (int i = 0; i < 3; ++i) {
                      if (dir >= graph.edge_end(node))
                        break;

                      auto unidir_update = [&](GNode dst) {
                        if (graph.getGID(dst) < NUM_CELLS) {
                          auto& dst_data = graph.getData(dst);
                          if (node_data.solution < dst_data.solution) {
                            auto& us = dst_data.upwind_solution[i];
                            if (node_data.solution <
                                galois::atomicMin(us, node_data.solution))
                              bitset_upwind_solution.set(dst);
#ifndef NDEBUG
                            if (dst == 1104) {
                              auto [ii, jj, kk] = getPos(graph.getGID(dst));
                              auto [unwrapped]  = dst_data.upwind_solution;
                              auto [a, b, c]    = unwrapped;
                              DGDEBUG("update ", dst, " (g", graph.getGID(dst),
                                      (dst < graph.numMasters() ? "M" : "m"),
                                      ") (", ii, " ", jj, " ", kk,
                                      ") upwind_solution: ", a, " ", b, " ", c);
                            }
#endif
                          }
                        }
                      };

                      GNode dst = graph.getEdgeDst(dir);
                      unidir_update(dst);
                      std::advance(dir, 1); // another direction
                      dst = graph.getEdgeDst(dir);
                      unidir_update(dst);

                      std::advance(dir, 1);
                    }
                  }
                }
              },
              galois::no_stats(),
              galois::steal(), // galois::wl<OBIM>(Indexer),
              galois::loopname(
                  syncSubstrate->get_run_identifier("Push").c_str()));

          // sleep(5);
          syncSubstrate
              ->sync<writeAny, readAny, Reduce_pair_wise_min_upwind_solution,
                     Bitset_upwind_solution>("FastMarching");

          galois::runtime::reportStat_Tsum(
              REGION_NAME,
              "NumWorkItems_" + (syncSubstrate->get_run_identifier()),
              (uint32_t)more_work.read_local());
          ++_round_counter;
        } while (more_work.reduce(syncSubstrate->get_run_identifier().c_str()));
      },
      "FMM_VTune");
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Sanity check

template <typename Graph, typename GNode = typename Graph::GraphNode>
void SanityCheck(Graph& graph) {
  galois::DGReduceMax<double> max_error;

  const auto& allNodes = graph.allNodesRange();
  galois::do_all(
      galois::iterate(allNodes.begin(), allNodes.end()),
      [&](GNode node) noexcept {
        if (graph.getGID(node) >= NUM_CELLS)
          return;
        auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
        if (curData.solution == INF) {
#ifndef NDEBUG
          auto [ii, jj, kk] = getPos(graph.getGID(node));
          auto [unwrapped]  = curData.upwind_solution;
          auto [a, b, c]    = unwrapped;
          DGDEBUG("untouched ", node, " (g", graph.getGID(node),
                  (node < graph.numMasters() ? "M" : "m"), ") (", ii, " ", jj,
                  " ", kk, ") upwind_solution: ", a, " ", b, " ", c);
#endif
          galois::gPrint("Untouched cell: ", node, " (g", graph.getGID(node),
                         ") ", node < graph.numMasters(), "\n");
          // assert(curData.solution != INF);
        }

        SlnTy val = 0.;
        std::array<double, 3> dims{dx, dy, dz}; // TODO not exactly x y z order
        auto dir = graph.edge_begin(node);
        for (double& d : dims) {
          if (dir == graph.edge_end(node))
            break;
          GNode neighbor   = graph.getEdgeDst(dir);
          auto& first_data = graph.getData(neighbor);
          // assert(first_data.is_ghost || first_data.tag == KNOWN);
          std::advance(dir, 1); // opposite direction of the same dimension
          assert(dir != graph.edge_end(node));
          neighbor          = graph.getEdgeDst(dir);
          auto& second_data = graph.getData(neighbor);
          // assert(second_data.is_ghost || second_data.tag == KNOWN);
          SlnTy s1 = (curData.solution - first_data.solution) / d,
                s2 = (curData.solution - second_data.solution) / d;
          val += std::pow(std::max(0., std::max(s1, s2)), 2);
          std::advance(dir, 1);
        }
        auto tolerance = 2.e-8;
        SlnTy error    = std::sqrt(val) * curData.speed - 1.;
        max_error.update(error);
        if (error > tolerance) {
          auto [x, y, z] = getPos(graph.getGID(node));
          galois::gPrint("Upwind structure violated at cell: ", node, " (", x,
                         " ", y, " ", z, ")", " with ", curData.solution,
                         " of error ", error, " (",
                         std::sqrt(x * x + y * y + z * z), ")\n");
          return;
        }
      },
      galois::no_stats(), galois::loopname("sanityCheck"));

  auto me = max_error.reduce();
  DGPRINT("max err: ", me, "\n");
}

template <typename Graph, typename GNode = typename Graph::GraphNode>
void SanityCheck2(Graph& graph) {
  galois::do_all(
      galois::iterate(0ul, NUM_CELLS),
      [&](GNode node) noexcept {
        auto [x, y, z] = getCoord(graph.getGID(node));
        auto& solution = graph.getData(node).solution;
        assert(std::abs(solution - std::sqrt(x * x + y * y + z * z)));
      },
      galois::no_stats(), galois::loopname("sanityCheck2"));
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) noexcept {
  galois::DistMemSys galois_system;
  DistBenchStart(argc, argv, name, desc, url);

  galois::gDebug(ALGO_NAMES[algo]);

  global_config();

  // if (galois::runtime::getSystemNetworkInterface().ID == 0) {
  //   std::filesystem::exists(inputname);
  // }
  galois::runtime::getHostBarrier().wait();

  galois::StatTimer Ttotal("TimerTotal");
  Ttotal.start();

  // generate grids
  Graph* graph;
  std::tie(graph, syncSubstrate) = distGraphInitialization<NodeData, void>();

  // _debug_print();

  // initialize all cells
  initCells(*graph);
  galois::runtime::getHostBarrier().wait();

  // TODO better way for boundary settings?
  BL boundary;
  if (source_type == scatter)
    AssignBoundary<GNode>(*graph, boundary); // TODO
  else
    AssignBoundary(*graph, boundary);

  bitset_upwind_solution.resize(graph->size());
  galois::runtime::getHostBarrier().wait();

  for (int run = 0; run < numRuns; ++run) {
    DGPRINT("Run ", run, " started\n");
    std::string tn = "Timer_" + std::to_string(run);
    galois::StatTimer Tmain(tn.c_str());

    galois::DGAccumulator<uint32_t> busy;
    busy.reset();
    if (!boundary.empty()) {
      busy += 1;
#ifndef NDEBUG
      // print boundary
      for (GNode b : boundary) {
        auto [x, y, z] = getPos(graph->getGID(b));
        DGDEBUG("boundary: ", b, "(g", graph->getGID(b),
                (b < graph->numMasters() ? "M" : "m"), ") (", x, " ", y, " ", z,
                ") with ", graph->getData(b).solution);
      }
#endif
      initBoundary(*graph, boundary);
    } else {
      DGDEBUG("No boundary element");
    }
    assert(busy.reduce() && "Boundary not defined!");

    Tmain.start();

    FirstIteration(*graph, boundary);
    FastMarching(*graph);

    Tmain.stop();

    galois::runtime::getHostBarrier().wait();
    SanityCheck(*graph);
    // SanityCheck2(graph);

    if ((run + 1) != numRuns) {
      galois::runtime::getHostBarrier().wait();
      bitset_upwind_solution.reset();

      initCells(*graph);
      galois::runtime::getHostBarrier().wait();
    }
  }

  Ttotal.stop();
  return 0;
}
