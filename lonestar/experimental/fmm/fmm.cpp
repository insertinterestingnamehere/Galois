/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2019, The University of Texas at Austin. All rights reserved.
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
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>

// Silence erroneous warnings from within Boost headers
// that show up with gcc 8.1.
#pragma GCC diagnostic ignored "-Wparentheses"
// This warning triggers with the assert(("explanation", check));
// syntax since the left hand argument has no side-effects.
// I prefer using the comma operator over && though because
// the parentheses are more readable, so I'm silencing
// the warning for this file.
#pragma GCC diagnostic ignored "-Wunused-value"

#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

// Vendored from an old version of LLVM for Lonestar app command line handling.
#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "galois/runtime/Profile.h"

static char const* name = "FMM";
static char const* desc =
    "fmm";
static char const* url = "";

using CoordTy = double;
using SlnTy = double;

enum Algo { Serial = 0, Parallel };

const char* const ALGO_NAMES[] = { "serial", "parallel" };

static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::desc("Choose an algorithm:"),
         llvm::cl::values(
           clEnumVal(Serial, "Serial"),
           clEnumVal(Parallel, "Parallel"),
           clEnumValEnd),
         llvm::cl::init(Serial));
static llvm::cl::opt<unsigned long long> nh{
    "nh", llvm::cl::desc("number of cells in ALL direction"),
    llvm::cl::init(0u)};
static llvm::cl::opt<unsigned long long> nx{
    "nx", llvm::cl::desc("number of cells in x direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> ny{
    "ny", llvm::cl::desc("number of cells in y direction"),
    llvm::cl::init(10u)};
static llvm::cl::opt<unsigned long long> nz{
    "nz", llvm::cl::desc("number of cells in z direction"),
    llvm::cl::init(10u)};
// static llvm::cl::opt<CoordTy> num_groups{
//     "num_groups", llvm::cl::desc("number of frequency groups"),
//     llvm::cl::init(4u)};
// static llvm::cl::opt<CoordTy> num_vert_directions{
//     "num_vert_directions", llvm::cl::desc("number of vertical directions"),
//     llvm::cl::init(16u)};
// static llvm::cl::opt<CoordTy> num_horiz_directions{
//     "num_horiz_directions", llvm::cl::desc("number of horizontal directions."),
//     llvm::cl::init(32u)};
// static llvm::cl::opt<unsigned long long> num_iters{
//    "num_iters", llvm::cl::desc("number of iterations"), llvm::cl::init(10u)};
// static llvm::cl::opt<double> pulse_strength{
//    "pulse_strength", llvm::cl::desc("radiation pulse strength"),
//    llvm::cl::init(1.)};
// static llvm::cl::opt<double> absorption_coef{
//     "absorption_coef",
//     llvm::cl::desc("Absorption coefficient (between 0 and 1), absorption and "
//                    "scattering must sum to less than 1."),
//     llvm::cl::init(.01)};
// static llvm::cl::opt<double> scattering_coef{
//     "scattering_coef",
//     llvm::cl::desc("Scattering coefficient (between 0 and 1), absorption and "
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


static constexpr CoordTy xa = -.5, xb = .5;
static constexpr CoordTy ya = -.5, yb = .5;
static constexpr CoordTy za = -.5, zb = .5;

static std::size_t NUM_CELLS;
static CoordTy dx, dy, dz;

///////////////////////////////////////////////////////////////////////////////

// Idk why this hasn't been standardized in C++ yet, but here it is.
static constexpr double PI =
    3.1415926535897932384626433832795028841971693993751;
constexpr SlnTy INF = std::numeric_limits<SlnTy>::max();
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

struct NodeData {
  bool is_ghost;
  Tag tag;
  double speed;
  double solution;
};

// No fine-grained locks built into the graph.
// Use atomics for ALL THE THINGS!
using Graph = galois::graphs::LC_CSR_Graph<NodeData, void>
                ::with_no_lockable<true>::type
                ;
using GNode = Graph::GraphNode;
using WL = galois::InsertBag<GNode>;

/**
 * Convert node ID into coordinate tuple.
 * `id = i * nx * ny + j * ny + k`
 * `(i, j, k) = (z, x, y)`
 * @param node Node ID
 * @return Coordinate tuple (x, y, z)
 */
std::array<CoordTy, 3> getCoord(const GNode node) {
  if (node >= NUM_CELLS) return {};
  auto res = std::lldiv(node, nx * ny);
  CoordTy i = res.quot;
  res = std::lldiv(res.rem, ny);
  CoordTy j = res.quot;
  CoordTy k = res.rem;

  CoordTy x = xa + dx * (j + 1);
  CoordTy y = ya + dy * (k + 1);
  CoordTy z = za + dz * (i + 1);
  // galois::gDebug(i, " ", j, " ", k);
  // galois::gDebug(dx, " - ", dy, " - ", dz);
  assert(x < xb && y < yb && z < zb);
  return std::array<CoordTy, 3>({x, y, z});
}

/**
 * Convert coordinate tuple into node ID.
 * See getCoord
 */
GNode getNodeID(const std::array<CoordTy, 3>& coords) {
  uint64_t i = std::round((coords[2] - za) / dz - 1.);
  uint64_t j = std::round((coords[0] - xa) / dx - 1.);
  uint64_t k = std::round((coords[1] - ya) / dy - 1.);

  return i * nx * ny + j * ny + k;
}

///////////////////////////////////////////////////////////////////////////////

// TODO make it external and customizable
// TODO multi-dimension configuration
template <typename CoordTy>
static double SpeedFunction(std::array<CoordTy, 3> coords) {
  const CoordTy& x = coords[0], y = coords[1], z = coords[2];

  return 1.;
  // return 1. + .50 * std::sin(20. * PI * x) * std::sin(20. * PI * y) * std::sin(20. * PI * z);
  // return 1. - .99 * std::sin(2. * PI * x) * std::sin(2. * PI * y) * std::sin(2. * PI * z);
}

static double BoundaryCondition(std::array<CoordTy, 3> coords = {}) {
  return 0.;
}

static bool NonNegativeRegion(const std::array<CoordTy, 3>& coords) {
  const CoordTy& x = coords[0], y = coords[1], z = coords[2];

  // Example 1: a spherical interface of radius 0.25 centered at the origin
  // return x * x + y * y + z * z >= .25 * .25;

  // Example 2: a plane past through the origin
  return 100. * x + y + 2. * z >= 0.;
}

template <typename Graph, typename WL>
void AssignBoundary(Graph& graph, WL& boundary) {
  galois::do_all(
    galois::iterate(0ul, NUM_CELLS),
    [&](GNode node) noexcept {
      if (node > NUM_CELLS) return;

      if (NonNegativeRegion(getCoord(node))) {
        for (auto e : graph.edges(node, no_lockable_flag)) {
          GNode dst = graph.getEdgeDst(e);
          if (!NonNegativeRegion(getCoord(dst))) {
// #ifndef NDEBUG
//             auto c = getCoord(node);
//             galois::gDebug(node, " (", c[0], " ", c[1], " ", c[2], ")");
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

template <typename WL>
void AssignBoundary(WL& boundary) {
// #ifndef NDEBUG
//   GNode n = getNodeID({0., 0., 0.});
//   auto c = getCoord(n);
//   galois::gDebug(n, " (", c[0], " ", c[1], " ", c[2], ")");
// #endif
  boundary.push(getNodeID({0., 0., 0.}));
}

////////////////////////////////////

/////////////////////////////////////


// Some helper functions for atomic operations with doubles:
// TODO: try switching these to a load/compute/load/compare/CAS
// style loop and see if it speeds it up.

// Atomically do += operation on a double.
void atomic_relaxed_double_increment(std::atomic<double>& base,
                                     double increment) noexcept {
  auto current = base.load(std::memory_order_relaxed);
  /*decltype(current) previous;
  while (true) {
    previous = current;
    current = base.load(std::memory_order_relaxed);
    if (previous == current) {
      if (base.compare_exchange_weak(current, current + increment,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
        break;
      }
    }
  }*/
  while (!base.compare_exchange_weak(current, current + increment,
                                     std::memory_order_relaxed,
                                     std::memory_order_relaxed))
    ;
}

// Atomically do base = max(base, newval)
void atomic_relaxed_double_max(std::atomic<double>& base,
                               double newval) noexcept {
  auto current = base.load(std::memory_order_relaxed);
  while (current != std::max(current, newval)) {
    base.compare_exchange_weak(current, newval, std::memory_order_relaxed,
                               std::memory_order_relaxed);
  }
}


/////////////////////////////////////////////

// The label for a piece of work in the for_each loop
// that actually runs the sweeping computation.
struct work_t {
  std::size_t node_index;
  std::size_t direction_index;
};

// Routine to initialize graph topology and face normals.
auto generate_grid(Graph& built_graph, std::size_t nx, std::size_t ny,
                   std::size_t nz) noexcept {

  galois::graphs::FileGraphWriter temp_graph;
  // Each node represents a grid cell.
  // Each edge represents a face.
  // Ghost nodes are added to represent the exterior
  // of the domain on the other side of each face.
  // This is for boundary condition handling.
  std::size_t num_outer_faces = (nx * ny + ny * nz + nx * nz) * 2;
  std::size_t num_cells       = nx * ny * nz;
  std::size_t num_nodes       = num_cells + num_outer_faces;
  std::size_t num_edges       = 6 * nx * ny * nz + num_outer_faces;
  temp_graph.setNumNodes(num_nodes);
  temp_graph.setNumEdges(num_edges);
  temp_graph.setSizeofEdgeData(0);

  // Interior cells will have degree 6
  // since they will have either other cells or
  // ghost cells on every side.
  // This condition isn't true in irregular meshes,
  // but that'd be a separate mesh generation routine.
  temp_graph.phase1();
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        std::size_t id = i * nx * ny + j * ny + k;
        for (std::size_t l = 0; l < 6; l++) {
          temp_graph.incrementDegree(id);
        }
      }
    }
  }
  // Set the degrees for all the ghost cells to 1.
  // No ghost cell should share a boundary with more
  // than one actual cell in the domain.
  for (std::size_t id = num_cells; id < num_nodes; id++) {
    temp_graph.incrementDegree(id);
  }

  // Now that the degree of each node is known,
  // fill in the actual topology.
  // Also fill in the node data with the vector
  // normal to the face, going out from the current cell.
  temp_graph.phase2();
  std::size_t xy_low_face_start  = num_cells;
  std::size_t xy_high_face_start = xy_low_face_start + nx * ny;
  std::size_t yz_low_face_start  = xy_high_face_start + nx * ny;
  std::size_t yz_high_face_start = yz_low_face_start + ny * nz;
  std::size_t xz_low_face_start  = yz_high_face_start + ny * nz;
  std::size_t xz_high_face_start = xz_low_face_start + nx * nz;
  assert(("Error in logic for dividing up node ids for exterior faces.",
          num_nodes == xz_high_face_start + nx * nz));
  for (std::size_t i = 0; i < nx; i++) {
    for (std::size_t j = 0; j < ny; j++) {
      for (std::size_t k = 0; k < nz; k++) {
        // (i, j, k) = (z, x, y)
        std::size_t id = i * nx * ny + j * ny + k;
        if (i > 0) {
          temp_graph.addNeighbor(id, id - ny * nz);
        } else {
          std::size_t ghost_id = yz_low_face_start + j * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (i < nx - 1) {
          temp_graph.addNeighbor(id, id + ny * nz);
        } else {
          std::size_t ghost_id = yz_high_face_start + j * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (j > 0) {
          temp_graph.addNeighbor(id, id - nz);
        } else {
          std::size_t ghost_id = xz_low_face_start + i * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (j < ny - 1) {
          temp_graph.addNeighbor(id, id + nz);
        } else {
          std::size_t ghost_id = xz_high_face_start + i * nz + k;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (k > 0) {
          temp_graph.addNeighbor(id, id - 1);
        } else {
          std::size_t ghost_id = xy_low_face_start + i * ny + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
        if (k < nz - 1) {
          temp_graph.addNeighbor(id, id + 1);
        } else {
          std::size_t ghost_id = xy_high_face_start + i * ny + j;
          temp_graph.addNeighbor(ghost_id, id);
          temp_graph.addNeighbor(id, ghost_id);
        }
      }
    }
  }

  // TODO: is it possible to set the edge data
  // during construction without copying here?
  temp_graph.finish<Graph::edge_data_type>();
  //auto* rawEdgeData = temp_graph.finish<Graph::edge_data_type>();
  //std::uninitialized_copy(std::make_move_iterator(edge_data.begin()),
  //                        std::make_move_iterator(edge_data.end()),
  //                        rawEdgeData);

  galois::graphs::readGraph(built_graph, temp_graph);
  return std::make_tuple(num_nodes, num_cells, num_outer_faces,
                         xy_low_face_start, xy_high_face_start,
                         yz_low_face_start, yz_high_face_start,
                         xz_low_face_start, xz_high_face_start);
}

///////////////////////////////////////////////////////////////////////////////

void initCells(Graph& graph, size_t num_cells) {
  galois::do_all(
    galois::iterate(graph.begin(), graph.end()),
    [&](GNode node) noexcept {
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      curData.is_ghost = (node >= num_cells);
      curData.tag = FAR;
      curData.speed = SpeedFunction(getCoord(node));
      curData.solution = INF; // TODO ghost init?
    },
    galois::no_stats(),
    galois::loopname("initializeCells"));
}

void initBoundary(Graph& graph, WL& boundary) {
  galois::do_all(
    galois::iterate(boundary.begin(), boundary.end()),
    [&](GNode node) noexcept {
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      curData.tag = KNOWN;
      curData.solution = BoundaryCondition(getCoord(node));
    },
    galois::no_stats(),
    galois::loopname("initializeBoundary"));
}

void initializeBag(Graph& graph, WL& boundary, WL& initBag) {
  galois::do_all(
    galois::iterate(boundary.begin(), boundary.end()),
    [&](GNode node) noexcept {
      for (auto e : graph.edges(node, no_lockable_flag)) {
        GNode dst = graph.getEdgeDst(e);
        if (dst < NUM_CELLS) {
          auto& dstData = graph.getData(dst, no_lockable_flag);
          if (dstData.solution == INF)
            initBag.push(dst);
        }
      }
    },
    galois::loopname("initializeBag"));
}


// void updateNeighbors() {}

auto checkDirection(Graph& graph, GNode node, double sln, Graph::edge_iterator dir) {
  if (dir >= graph.edge_end(node, no_lockable_flag)) {
    galois::gDebug(node, (graph.getData(node, no_lockable_flag).is_ghost? " ghost":" non-ghost"), " ",
                     std::distance(graph.edge_begin(node, no_lockable_flag), graph.edge_end(node, no_lockable_flag)) );
    GALOIS_DIE("invalid direction");
  }
  // GNode upwind = node;
  GNode neighbor = graph.getEdgeDst(dir);
  auto& first_data = graph.getData(neighbor, no_lockable_flag);
  if (first_data.solution < sln) {
    sln = first_data.solution;
    // upwind = neighbor;
  }
  std::advance(dir, 1); // opposite direction of the same dimension
  if (dir != graph.edge_end(node, no_lockable_flag)) {
    neighbor = graph.getEdgeDst(dir);
    auto& second_data = graph.getData(neighbor, no_lockable_flag);
    if (second_data.solution < sln) {
      // upwind = neighbor;
      sln = second_data.solution;
    }
  }
  // since it's uniform discretization (regular grids), delta is 1
  // return std::make_pair(delta, sln)
  return sln;
}

double solveQuadratic(Graph& graph, GNode node, double sln, const double speed) {
  // TODO oarameterize dimension 3
  std::array<double, 3> solutions = {0., 0., 0.};
  int non_zero_counter = 0;
  auto dir = graph.edge_begin(node, no_lockable_flag);
  for (double& s : solutions) {
    if (dir == graph.edge_end(node, no_lockable_flag))
      break;
    s = checkDirection(graph, node, sln, dir);
    if (s != sln)
      non_zero_counter++;
    else
      s = 0.;
    std::advance(dir, 2);
  }
  galois::gDebug("solveQuadratic: ", solutions[0], " ", solutions[1], " ", solutions[2]);
  for (; non_zero_counter > 0; non_zero_counter--) {
    auto max_sln_it = std::max_element(solutions.begin(), solutions.end());
    double a(0.), b(0.), c(0.);
    for (const double& s : solutions) {
      double temp = 1. / (1. * 1.); // TODO delta
      a += temp;
      temp *= s;
      b += temp;
      temp *= s;
      c += temp;
    }
    b *= -2.;
    c -= (1. / (speed * speed));
    double d = b * b - (4. * a * c);
    if (d >= 0) {
      double new_sln = (-b + std::sqrt(d)) / (2. * a);
      if (new_sln > *max_sln_it)
        sln = std::min(sln, new_sln); // TODO atomic here?
    }
    *max_sln_it = 0.;
  }
  return sln;
}

auto SerialCheckDirection(Graph& graph, GNode node, double center_sln, Graph::edge_iterator dir) {
  SlnTy sln = 0;
  if (dir >= graph.edge_end(node, no_lockable_flag)) {
    galois::gDebug(node, (graph.getData(node, no_lockable_flag).is_ghost? " ghost":" non-ghost"), " ",
                     std::distance(graph.edge_begin(node, no_lockable_flag), graph.edge_end(node, no_lockable_flag)) );
    GALOIS_DIE("invalid direction");
  }
  GNode upwind = node;
  GNode neighbor = graph.getEdgeDst(dir);
  auto& first_data = graph.getData(neighbor, no_lockable_flag);
  // galois::gDebug("Check neighbor ", neighbor, (int)first_data.tag);
  if (first_data.tag == KNOWN) {
    sln = first_data.solution;
    upwind = neighbor;
  }
  std::advance(dir, 1); // opposite direction of the same dimension
  if (dir != graph.edge_end(node, no_lockable_flag)) {
    neighbor = graph.getEdgeDst(dir);
    auto& second_data = graph.getData(neighbor, no_lockable_flag);
    // galois::gDebug("Check neighbor ", neighbor, (int)second_data.tag);
    if (second_data.tag == KNOWN) {
      upwind = neighbor;
      sln = second_data.solution;
    }
  }
  if (upwind == node)
    return std::make_pair(0., 0.);
  return std::make_pair(sln, dx);
}

double SerialSolveQuadratic(Graph& graph, GNode node, double sln, const double speed) {
  // TODO oarameterize dimension 3
  std::array<std::pair<double, double>, 3> sln_delta {
                                            std::make_pair(0., dx),
                                            std::make_pair(0., dy),
                                            std::make_pair(0., dz)
                                          };
  // int non_zero_counter = 0;
  auto dir = graph.edge_begin(node, no_lockable_flag);
  for (auto& p : sln_delta) {
    if (dir == graph.edge_end(node, no_lockable_flag))
      break;
    double& s = p.first;
    double& d = p.second;
    auto [si, di] = SerialCheckDirection(graph, node, sln, dir);
    if (di) {
      s = si;
      // non_zero_counter++;
    }
    else {
      // s = 0.; // already there
      d = 0.;
    }
    std::advance(dir, 2);
  }
  // galois::gDebug("SerialSolveQuadratic: ", sln_delta[0].second, " ", sln_delta[1].second, " ", sln_delta[2].second);
  // for (; non_zero_counter > 0; non_zero_counter--) {
    // auto max_sln_it = std::max_element(solutions.begin(), solutions.end());
    double a(0.), b(0.), c(0.);
    for (const auto& p : sln_delta) {
      const double& s = p.first, d = p.second;
      galois::gDebug(s, " ", d);
      double temp = d == 0? 0. : 1. / (d * d); // TODO delta
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

/////
// serial
template <typename Graph, typename BL, typename WL>
void SerialInitializeBag(Graph& graph, BL& boundary, WL& wl) {
  for (GNode node : boundary) {
    for (auto e : graph.edges(node, no_lockable_flag)) {
      GNode dst = graph.getEdgeDst(e);
      if (dst < NUM_CELLS) {
        auto& dstData = graph.getData(dst, no_lockable_flag);
        if (!dstData.is_ghost && dstData.tag != KNOWN) {
          assert(dstData.solution == INF && dstData.tag == FAR);
          SlnTy old_sln = dstData.solution;
          double sln_temp = SerialSolveQuadratic(graph, dst, old_sln, dstData.speed);
          if (sln_temp < old_sln) {
            galois::gDebug("Hi! I'm ", dst, " ", sln_temp);
            dstData.solution = sln_temp;
            dstData.tag = BAND;
            auto iter = wl.lower_bound(old_sln);
            for (; iter != wl.end(); iter++) {
              if (iter->second == dst)
                break;
              if (iter->first != sln_temp) {
                iter = wl.end();
                break;
              }
            }
            if (iter == wl.end()) {
              wl.insert({sln_temp, dst});
            } else {
              auto nh = wl.extract(iter); // node handle
              nh.key() = sln_temp;
              wl.insert(std::move(nh));
            }
          }
        }
      }
    }
  }
}

template <typename Graph, typename WL>
void SerialFastMarching(Graph& graph, WL& wl) {
  double max_error = 0;
  while (!wl.empty()) {
    auto beg = wl.begin();
    GNode node = beg->second;
    // SlnTy old_sln = beg.first;
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
#ifndef NDEBUG
    {
      auto [x, y, z] = getCoord(node);
      galois::gDebug("Hi! I'm ", node, " (", x, " ", y, " ", z, " ) with ", curData.solution);
    }
    if (curData.tag != BAND) {
      galois::gDebug(node, " in heap with tag ", curData.tag);
      std::abort();
    }

    if (curData.solution != beg->first) {
      galois::gDebug("Wrong entry in the heap! ", beg->second, " ", beg->first, " but ", curData.solution);
      for(auto i : wl) {
        if (i.second == node)
          galois::gDebug(i.first);
      }
      std::abort();
    }
    if (curData.is_ghost) {
      // should not get here, asserted before
      std::abort();
    }

    {
      auto [x, y, z] = getCoord(node);
      if (curData.solution - std::sqrt(x * x + y * y + z * z) > max_error)
        max_error = curData.solution - std::sqrt(x * x + y * y + z * z);
      if (curData.solution - std::sqrt(x * x + y * y + z * z) > 0.2) {
        galois::gDebug(curData.solution - std::sqrt(x * x + y * y + z * z),
          " - wrong distance, should be ", std::sqrt(x * x + y * y + z * z));
        assert(false);
      }
    }
#endif
    curData.tag = KNOWN;
    // TODO Delta stepping squeezes in here
    wl.erase(beg);

    // UpdateNeighbors
    for (auto e : graph.edges(node, no_lockable_flag)) {
      GNode dst = graph.getEdgeDst(e);
      if (dst < NUM_CELLS) {
        auto& dstData = graph.getData(dst, no_lockable_flag);
        if (!dstData.is_ghost && dstData.tag != KNOWN) {
          {
            auto [x, y, z] = getCoord(dst);
            galois::gDebug("Update ", dst, " (", x, " ", y, " ", z, " )");
          }
          // assert(dstData.solution == INF && dstData.tag == FAR);
          SlnTy old_sln = dstData.solution;
          double sln_temp = SerialSolveQuadratic(graph, dst, old_sln, dstData.speed);
          if (sln_temp < old_sln) {
            dstData.solution = sln_temp;
            dstData.tag = BAND;
            auto iter = wl.lower_bound(old_sln);
            for (; iter != wl.end(); std::advance(iter, 1)) {
              if (dst == 274) {
                galois::gDebug("274: ", old_sln, " iter: ", iter->first, " ", iter->second);
              }
              if (iter->second == dst) {
              if (dst == 274) {
                galois::gDebug("274 catch: ", old_sln, " iter: ", iter->first, " ", iter->second);
              }
                
                break;
              }
              if (iter->first != old_sln) {
                iter = wl.end();
                break;
              }
            }
              if (dst == 274) {
                galois::gDebug("274 finished: ", old_sln, " iter: ", iter->first, " ", iter->second);
              }
            if (iter == wl.end()) {
              if (dst == 274) {
                galois::gDebug("dumping heap ...");
                for (auto i : wl) {
                  galois::gDebug(i.first, " ", i.second);
                  assert(i.second != 274);
                }
                galois::gDebug(dstData.tag);
                assert(dstData.tag == BAND);
              }
              wl.insert({sln_temp, dst});
            } else {
              if (dst == 274) assert(dstData.tag == BAND);
              auto nh = wl.extract(iter); // node handle
              nh.key() = sln_temp;
              wl.insert(std::move(nh));
            }
          }
        }
      }
    }
  }
  galois::gDebug("max error: ", max_error);
}

void FastMarching(Graph& graph, WL& initBag) {
  auto Indexer = [&](const double& sln) {
    unsigned t = std::round(sln * 128);
    return t;
  };
  using PSchunk = galois::worklists::PerSocketChunkLIFO<32>; // chunk size 16
  using OBIM    = galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;

galois::runtime::profileVtune(
[&](){
  galois::for_each(
    galois::iterate(initBag.begin(), initBag.end()),
    [&](GNode node, auto& ctx) {
      assert(node < NUM_CELLS && "Ghost Point!");
      auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      if (curData.is_ghost) {
        // should not get here, asserted before
        std::abort();
      }
      // assert(curData.tag == BAND);
      // if (curData.tag != KNOWN) {
        double sln_temp = solveQuadratic(graph, node, curData.solution, curData.speed);
        // updated
        if (sln_temp < curData.solution) {
          curData.solution = sln_temp;
          // if (finalized) {
            for (auto e : graph.edges(node, no_lockable_flag)) {
              GNode dst = graph.getEdgeDst(e);
              if (dst < NUM_CELLS) {
                auto& dstData = graph.getData(dst, no_lockable_flag);
                if (dstData.solution > sln_temp)
                  ctx.push(dst);
              }
            }
            curData.tag = KNOWN;
          // }
          // else {
          //   if (curData.tag != BAND) {
          //     assert(curData.tag == FAR);
          //     curData.tag = BAND;
          //   }
          //   ctx.push(node);
          // }
        }
        // not updated
        // else {
        //   assert(curData.tag == BAND);
        //   ctx.push(node);
        // }
      // }
    },
    galois::no_conflicts(),
    // galois::no_stats(),
    galois::wl<OBIM>(Indexer),
    galois::loopname("FMM")
  );
},
"FMM_VTune");
}

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
        return;
      }

      SlnTy val = 0.;
      std::array<double, 3> dims { dx, dy, dz }; // TODO not exactly x y z order
      auto dir = graph.edge_begin(node, no_lockable_flag);
      for (double& d : dims) {
        if (dir == graph.edge_end(node, no_lockable_flag))
          break;
        GNode neighbor = graph.getEdgeDst(dir);
        auto& first_data = graph.getData(neighbor, no_lockable_flag);
        assert(first_data.is_ghost || first_data.tag == KNOWN);
        std::advance(dir, 1); // opposite direction of the same dimension
        assert(dir != graph.edge_end(node, no_lockable_flag));
        neighbor = graph.getEdgeDst(dir);
        auto& second_data = graph.getData(neighbor, no_lockable_flag);
        assert(second_data.is_ghost || second_data.tag == KNOWN);
        SlnTy
          s1 = (curData.solution - first_data.solution) / d,
          s2 = (curData.solution - second_data.solution) / d;
        val += std::pow(std::max(0., std::max(s1, s2)), 2);
        std::advance(dir, 1);
      }
      auto tolerance = 2.e-1;
      SlnTy error = std::sqrt(val) * curData.speed - 1.;
      max_error.update(error);
      if (error > tolerance) {
        auto [x, y, z] = getCoord(node);
        galois::gPrint("Upwind structure violated at cell: ", node,
                        " (", x, " ", y, " ", z, ")",
                        " with ", curData.solution, " of error ", error,
                        " (", std::sqrt(x * x + y * y + z * z), ")\n");
        return;
      }
    },
    galois::no_stats(),
    galois::loopname("sanityCheck"));

  galois::gPrint("max err: ", max_error.reduce());
}

void SanityCheck2(Graph& graph) {
  galois::do_all(
    galois::iterate(0ul, NUM_CELLS),
    [&](GNode node) noexcept {
      auto [x, y, z] = getCoord(node);
      auto &solution = graph.getData(node).solution;
      assert(std::abs(solution - std::sqrt(x * x + y * y + z * z)));
    },
    galois::no_stats(),
    galois::loopname("sanityCheck2"));
}

void _sanity_coord() {
  for (GNode i = 0; i < NUM_CELLS; i++) {
    auto c = getCoord(i);
    galois::gDebug(c[0], " ", c[1], " ", c[2], " ", getNodeID(c));
  }
}

template<typename Graph, typename WL>
void serial(Graph& graph, WL& boundary) {
  //  using HeapElemTy = std::pair<GNode, double>
  //  auto heapCmp = [&](HeapElemTy a, HeapElemTy b) { return a.second < b.second; }
  //  using HeapTy = galois::MinHeap<std::pair<GNode, double>>
  using HeapTy = std::multimap<SlnTy, GNode>;
  HeapTy wl;

  SerialInitializeBag(graph, boundary, wl);

  if (wl.empty()) {
    galois::gDebug("No cell to be processed!");
    std::abort();
  } else {
#ifndef NDEBUG
  galois::gDebug("vvvvvvvv init band vvvvvvvv");
  for (auto i : wl) {
    auto coords = getCoord(i.second);
    galois::gDebug(i.second, " (", coords[0], " ", coords[1], " ", coords[2], ") with ", i.first);
  }
  galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
  }

#ifndef NDEBUG
// initBag sanity check
  galois::do_all(
    galois::iterate(wl),
    [&](auto pair) {
      galois::gDebug(pair.first, " : ", pair.second);
      GNode node = pair.second;
      auto& curData = graph.getData(node, no_lockable_flag);
      // if (curData.tag != BAND) {
      //   GALOIS_DIE("Problem with initBag");
      // }
      if (curData.tag != BAND) {
        galois::gDebug("non-BAND");
      }
    },
    //galois::no_stats(),
    galois::loopname("sanity_check_initBag")
  );
#endif // end of initBag sanity check;

  SerialFastMarching(graph, wl);
}

template<typename Graph, typename WL>
void parallel(Graph& graph, WL& boundary) {
  WL initBag;

  initializeBag(graph, boundary, initBag);

#ifndef NDEBUG
// initBag sanity check
  galois::do_all(
    galois::iterate(initBag),
    [&](GNode node) {
      auto& curData = graph.getData(node, no_lockable_flag);
      // if (curData.tag != BAND) {
      //   GALOIS_DIE("Problem with initBag");
      // }
      if (curData.solution != INF) {
        galois::gDebug("non-FAR");
      }
    },
    galois::no_stats(),
    galois::loopname("sanity_check_initBag")
  );
#endif // end of initBag sanity check;

  FastMarching(graph, initBag);
}

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url);

  Graph graph;

  galois::gDebug(ALGO_NAMES[algo]);

  if (nh) nx = ny = nz = nh;
  NUM_CELLS = nh? nh * nh * nh : nx * ny * nz;
  dx = (xb - xa) / CoordTy(nx + 1);
  dy = (yb - ya) / CoordTy(ny + 1);
  dz = (zb - za) / CoordTy(nz + 1);
  galois::gDebug(nx, " - ", ny, " - ", nz);
  galois::gDebug(dx, " - ", dy, " - ", dz);

  auto [num_nodes, num_cells, num_outer_faces, xy_low, xy_high, yz_low, yz_high,
        xz_low, xz_high] = generate_grid(graph, nx, ny, nz);

  // _sanity_coord();

  initCells(graph, num_cells);

  // TODO boundary settings
  WL boundary;
  //AssignBoundary(graph, boundary);
  AssignBoundary(boundary);
  assert(!boundary.empty() && "Boundary not defined!");

#ifndef NDEBUG
  galois::gDebug("vvvvvvvv boundary vvvvvvvv");
  for (GNode b : boundary) {
    auto coords = getCoord(b);
    galois::gDebug(b, " (", coords[0], " ", coords[1], " ", coords[2], ")");
  }
  galois::gDebug("^^^^^^^^ boundary ^^^^^^^^");
#endif

  initBoundary(graph, boundary);

  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case Serial:
    serial(graph, boundary);
    break;
  case Parallel:
    parallel(graph, boundary);
    break;
  default:
    std::abort();
  }

  Tmain.stop();

  SanityCheck(graph);
  //SanityCheck2(graph);

  return 0;
}

