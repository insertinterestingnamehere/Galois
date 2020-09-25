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
#include <fstream>
#include <iostream>
#include <type_traits>
#include <utility>

#include <llvm/Support/CommandLine.h>

#include <galois/AtomicHelpers.h>
#include <galois/Galois.h>
#include <galois/graphs/FileGraph.h>
#include <galois/graphs/Graph.h>
#include <galois/graphs/LCGraph.h>

#ifdef GALOIS_ENABLE_VTUNE
#include "galois/runtime/Profile.h"
#endif

#include "Lonestar/BoilerPlate.h"

static char const* name = "Fast Marching Method";
static char const* desc =
    "Eikonal equation solver "
    "(https://en.wikipedia.org/wiki/Fast_marching_method)";
static char const* url = "";

#include "fastmarchingmethod.h"
#define DIM_LIMIT 2 // 2-D specific

enum Algo { serial = 0, parallel, fim };
enum SourceType { scatter = 0, analytical };

const char* const ALGO_NAMES[] = {"serial", "parallel",
                                  "Fast Iterative Method"};

static llvm::cl::OptionCategory catAlgo("1. Algorithmic Options");
static llvm::cl::opt<Algo>
    algo("algo", llvm::cl::value_desc("algo"),
         llvm::cl::desc("Choose an algorithm (default parallel):"),
         llvm::cl::values(clEnumVal(serial, "serial heap implementation"),
                          clEnumVal(parallel, "parallel implementation"),
                          clEnumVal(fim, "Fast Iterative Method")),
         llvm::cl::init(parallel), llvm::cl::cat(catAlgo));
static llvm::cl::opt<double> RF{"rf",
                                llvm::cl::desc("round-off factor for OBIM"),
                                llvm::cl::init(1.), llvm::cl::cat(catAlgo)};
static llvm::cl::opt<double> tolerance{"e", llvm::cl::desc("Final error bound"),
                                       llvm::cl::init(1.e-14),
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
        "requires explicit definition of the size on each dimensions"),
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
static llvm::cl::opt<std::string> verify_npy(
    "vnpy",
    llvm::cl::desc("Canonical results for verification in a npy format file"),
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

constexpr galois::MethodFlag UNPROTECTED = galois::MethodFlag::UNPROTECTED;

static_assert(sizeof(std::atomic<std::size_t>) <= sizeof(double),
              "Current buffer allocation code assumes atomic "
              "counters smaller than sizeof(double).");
static_assert(std::is_trivial_v<std::atomic<std::size_t>> &&
                  std::is_standard_layout_v<std::atomic<std::size_t>>,
              "Current buffer allocation code assumes no special "
              "construction/deletion code is needed for atomic counters.");

struct NodeData {
  double speed; // read only
  std::atomic<double> solution{INF};
};

using Graph =
    galois::graphs::LC_CSR_Graph<NodeData, void>::with_no_lockable<true>::type;
using GNode         = Graph::GraphNode;
using BL            = galois::InsertBag<GNode>;
using UpdateRequest = std::pair<data_t, GNode>;
using HeapTy        = FMMHeapWrapper<
    std::multimap<UpdateRequest::first_type, UpdateRequest::second_type>>;
using WL = galois::InsertBag<UpdateRequest>;

// These reference the input parameters as global data so they have to be
// included after the input parameters are defined.
#include "util/input.hh"
#include "util/output.hh"

class Solver {
  Graph& graph;
  BL boundary;
  PushWrap pushWrap;
  galois::substrate::PerThreadStorage<
      std::array<std::pair<data_t, data_t>, DIM_LIMIT>>
      local_pairs;

  /**
   * Slow update process. Accumulate coefficients of quadratic equations of the
   * differencing scheme and solve each.
   */
  template <bool CONCURRENT, typename NodeData, typename It>
  data_t slowUpdate(NodeData& my_data, It edge_begin, It edge_end) {
    data_t sln         = my_data.solution.load(std::memory_order_relaxed);
    const data_t speed = my_data.speed;

    auto& sln_delta = *(local_pairs.getLocal());
    sln_delta       = {std::make_pair(sln, dx), std::make_pair(sln, dy)};

    int upwind_counter = 0;
    auto FindUpwind    = [&](data_t& up_sln, It back, It forth) {
      auto& A = graph.getData(graph.getEdgeDst(back), UNPROTECTED);
      auto& B = graph.getData(graph.getEdgeDst(forth), UNPROTECTED);
      if (data_t candidate =
              std::min(A.solution.load(std::memory_order_relaxed),
                       B.solution.load(std::memory_order_relaxed));
          candidate < up_sln) {
        up_sln = candidate;
        ++upwind_counter;
      }
    };

    auto dir = edge_begin;
    for (auto& [s, d] : sln_delta) {
      if (dir >= edge_end)
        GALOIS_DIE("Dimension exceeded");

      FindUpwind(s, dir++, dir++);
    }

    if (upwind_counter == 0)
      return sln;
    galois::gDebug("#upwind_dirs: ", upwind_counter);

    std::sort(sln_delta.begin(), sln_delta.end(),
              [&](std::pair<data_t, data_t>& a, std::pair<data_t, data_t>& b) {
                return a.first < b.first;
              });

    /*
     * General form of the differencing scheme:
     *
     * Sigma<i=1...N>( (t-t_i)^2 / h_i^2 ) = 1 / f^2
     *
     * t: arrival time (to be solved)
     * h: unit step of dimension i
     * f: speed
     *
     * 1st order:
     * (1 / h^2) * t^2 + (-2 * t_i / h^2) * t + (t_i^2 / h^2 - 1 / f^2) = 0
     * 2nd order:
     * ...
     *
     * General form of quadratic equations:
     * a * t^2 + b * t + c = 0
     *
     * General form of coefficients:
     * a = Sigma<i=1...N>( 1 / h_i^2 )
     * b = -2 * Sigma<i=1...N>( t_i / h_i^2 )
     * c = Sigma<i=1...N>( t_i^2 / h_i^2 ) - 1 / f^2
     */
    auto p = sln_delta.begin();
    data_t a(0.), b_(0.), c_(0.), F{1. / (speed * speed)};
    do {
      const auto& [s, d] = *p++;
      galois::gDebug("Upwind neighbor: v=", s, " h=", d);
      // Arrival time may be updated in previous rounds
      // in which case remaining upwind dirs become invalid
      if (sln < s)
        break;

      data_t ii = 1. / (d * d); // 1 / h_i^2
      a += ii;
      ii *= s; // t_i / h_i^2
      b_ += ii;
      ii *= s; // t_i^2 / h_i^2
      c_ += ii;
      if (CONCURRENT || upwind_counter == 1) {
        data_t b = -2. * b_, c = c_ - F;
        data_t del = b * b - (4. * a * c);
        galois::gDebug("Coefficients: ", a, " ", b, " ", c, " Delta=", del);
        if (del >= 0) {
          data_t new_sln = (-b + std::sqrt(del)) / (2. * a); // larger root
          galois::gDebug("new solution: ", new_sln);
          if constexpr (CONCURRENT) { // actually also apply to serial
            if (new_sln > s) {        // conform causality
              sln = std::min(sln, new_sln);
            }
          } else {
            return new_sln;
          }
        }
      }
    } while (--upwind_counter);
    return sln;
  }

  /**
   * Quick update process. Hard code for all possible solutions.
   */
  template <bool CONCURRENT, typename NodeData, typename It>
  data_t quickUpdate(NodeData& my_data, It edge_begin,
                     [[maybe_unused]] It edge_end) {
    const auto f = my_data.speed;
    assert(dx == dy);
    auto h = dx;
    auto N = graph.getEdgeDst(*edge_begin),
         S = graph.getEdgeDst(*(++edge_begin)),
         E = graph.getEdgeDst(*(++edge_begin)),
         W = graph.getEdgeDst(*(++edge_begin));
    assert(++edge_begin == edge_end);
    auto u_N = graph.getData(N).solution.load(std::memory_order_relaxed),
         u_S = graph.getData(S).solution.load(std::memory_order_relaxed),
         u_E = graph.getData(E).solution.load(std::memory_order_relaxed),
         u_W = graph.getData(W).solution.load(std::memory_order_relaxed);
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
    auto u_V = std::min(u_N, u_S), u_H = std::min(u_E, u_W);
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

  /**
   * Core process: apply operators on active nodes.
   */
  template <bool CONCURRENT, typename WL, typename T = typename WL::value_type>
  void FastMarching(WL& wl) {
    galois::GAccumulator<std::size_t> emptyWork;
    galois::GAccumulator<std::size_t> nonEmptyWork;
    emptyWork.reset();
    nonEmptyWork.reset();

    auto PushOp = [&](const auto& item, auto& wl) {
      using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
      // typename ItemTy::second_type node;
      // std::tie(std::ignore, node) = item;
      auto [old_sln, node] = item;
      galois::gDebug(old_sln, " ", node);
      assert(node < NUM_CELLS && "Ghost Point!");
      auto& curData      = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      const data_t s_sln = curData.solution.load(std::memory_order_relaxed);

      // Ignore stale repetitive work items
      if (s_sln < old_sln) {
        return;
      }

      // UpdateNeighbors
      for (auto e : graph.edges(node, UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(e);

        if (dst >= NUM_CELLS)
          continue;
        auto& dstData = graph.getData(dst, UNPROTECTED);

        // Given that the arrival time propagation is non-descending
        data_t old_neighbor_val =
            dstData.solution.load(std::memory_order_relaxed);
        if (old_neighbor_val <= s_sln) {
          continue;
        }

        data_t sln_temp =
            quickUpdate<CONCURRENT>(dstData, graph.edge_begin(dst, UNPROTECTED),
                                    graph.edge_end(dst, UNPROTECTED));

        do {
          if (sln_temp >= old_neighbor_val)
            goto continue_outer;
        } while (!dstData.solution.compare_exchange_weak(
            old_neighbor_val, sln_temp, std::memory_order_relaxed));
        pushWrap(wl, ItemTy{sln_temp, dst}, old_neighbor_val);
      continue_outer:;
      }
    };

    //! Run algo
    if constexpr (CONCURRENT) {
      auto Indexer = [&](const T& item) {
        unsigned t = std::round(item.first * RF);
        // galois::gDebug(item.first, "\t", t, "\n");
        return t;
      };
      using PSchunk = galois::worklists::PerSocketChunkFIFO<128>;
      using OBIM =
          galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;
#ifdef GALOIS_ENABLE_VTUNE
      galois::runtime::profileVtune(
          [&]() {
#endif
            galois::for_each(galois::iterate(wl.begin(), wl.end()), PushOp,
                             galois::disable_conflict_detection(),
                             // galois::no_stats(),  // stat iterations
                             galois::wl<OBIM>(Indexer),
                             galois::loopname("FMM"));
#ifdef GALOIS_ENABLE_VTUNE
          },
          "FMM_VTune");
#endif
    } else {
      std::size_t num_iterations = 0;

      while (!wl.empty()) {

        PushOp(wl.pop(), wl);

        num_iterations++;

#ifndef NDEBUG
        sleep(1); // Debug pause
        galois::gDebug("\n********\n");
#endif
      }

      galois::gPrint("#iterarions: ", num_iterations, "\n");
    }

    // galois::runtime::reportParam("Statistics", "EmptyWork",
    // emptyWork.reduce()); galois::runtime::reportParam("Statistics",
    // "NonEmptyWork",
    //                              nonEmptyWork.reduce());
  }

  /**
   * Core process: apply operators on active nodes.
   */
  template <bool CONCURRENT, typename WL, typename T = typename WL::value_type>
  void FastIterativeMethod(WL* wl) {
    galois::GAccumulator<std::size_t> emptyWork;
    galois::GAccumulator<std::size_t> nonEmptyWork;
    emptyWork.reset();
    nonEmptyWork.reset();

    // Jacobi iteration
    WL* next = new WL(); // TODO memory leak!

    auto PushOp = [&](const auto& item) {
      using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
      // typename ItemTy::second_type node;
      // std::tie(std::ignore, node) = item;
      auto [old_sln, node] = item;
      galois::gDebug(old_sln, " ", node);
      assert(node < NUM_CELLS && "Ghost Point!");
      auto& curData      = graph.getData(node, galois::MethodFlag::UNPROTECTED);
      const data_t s_sln = curData.solution.load(std::memory_order_relaxed);

      // Ignore stale repetitive work items
      if (s_sln < old_sln) {
        return;
      }

      data_t new_val =
          quickUpdate<CONCURRENT>(curData, graph.edge_begin(node, UNPROTECTED),
                                  graph.edge_end(node, UNPROTECTED));
      if (new_val != INF &&
          std::abs(new_val - galois::atomicMin(curData.solution, new_val)) >=
              tolerance) { // not converged
        galois::gDebug("not converged:", new_val, " ", node);
        pushWrap(*next, ItemTy{new_val, node}, old_sln);
        return;
      }

      // UpdateNeighbors
      for (auto e : graph.edges(node, UNPROTECTED)) {
        GNode dst = graph.getEdgeDst(e);

        if (dst >= NUM_CELLS)
          continue;
        auto& dstData = graph.getData(dst, UNPROTECTED);

        // Given that the arrival time propagation is non-descending
        data_t old_neighbor_val =
            dstData.solution.load(std::memory_order_relaxed);
        if (old_neighbor_val <= s_sln) {
          continue;
        }

        data_t sln_temp =
            quickUpdate<CONCURRENT>(dstData, graph.edge_begin(dst, UNPROTECTED),
                                    graph.edge_end(dst, UNPROTECTED));

        do {
          if (sln_temp >= old_neighbor_val)
            goto continue_outer;
        } while (!dstData.solution.compare_exchange_weak(
            old_neighbor_val, sln_temp, std::memory_order_relaxed));
        assert(sln_temp < INF);
        pushWrap(*next, ItemTy{sln_temp, dst}, old_neighbor_val);
      continue_outer:;
      }
    };

    //! Run algo
    if constexpr (CONCURRENT) {
      auto Indexer = [&](const T& item) {
        unsigned t = std::round(item.first * RF);
        // galois::gDebug(item.first, "\t", t, "\n");
        return t;
      };
      using PSchunk = galois::worklists::PerSocketChunkFIFO<128>;
      using OBIM =
          galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;

      while (!wl->empty()) {
#ifdef GALOIS_ENABLE_VTUNE
        galois::runtime::profileVtune(
            [&]() {
#endif
              galois::do_all(galois::iterate(wl->begin(), wl->end()), PushOp,
                             galois::steal(),
                             // galois::no_stats(),  // stat iterations
                             galois::wl<OBIM>(Indexer),
                             galois::loopname("FMM"));
#ifdef GALOIS_ENABLE_VTUNE
            },
            "FMM_VTune");
#endif
        wl->clear();
        std::swap(wl, next);
      }
    } else {
      // TODO: not implemented yet!
      /*
      std::size_t num_iterations = 0;

      while (!wl.empty()) {

        PushOp(wl.pop(), wl);

        num_iterations++;

#ifndef NDEBUG
        sleep(1); // Debug pause
        galois::gDebug("\n********\n");
#endif
      }

      galois::gPrint("#iterarions: ", num_iterations, "\n");
      */
    }

    // galois::runtime::reportParam("Statistics", "EmptyWork",
    // emptyWork.reduce()); galois::runtime::reportParam("Statistics",
    // "NonEmptyWork",
    //                              nonEmptyWork.reduce());
  }

  struct Verifier {
    virtual ~Verifier() {}
    virtual data_t get(GNode) const                         = 0;
    virtual void alert(GNode, data_t, data_t, double) const = 0;
  };
  friend struct SelfVerifier;
  friend struct FileVerifier;
  /**
   * Verify results by re-apply operators on each nodes and observe the
   * differences.
   */
  struct SelfVerifier : Verifier {
    Solver& p;

    explicit SelfVerifier(Solver& p_) : p(p_) {}
    ~SelfVerifier() = default;

    data_t get(GNode node) const {
      return p.quickUpdate<true>(p.graph.getData(node, UNPROTECTED),
                                 p.graph.edge_begin(node, UNPROTECTED),
                                 p.graph.edge_end(node, UNPROTECTED));
    }

    void alert(GNode node, data_t old_val, data_t new_val, double error) const {
      auto [x, y] = id2xy(node);
      galois::gPrint("Error bound violated at cell ", node, " (", x, " ", y,
                     "): old_val=", old_val, " new_val=", new_val,
                     " error=", error, "\n");
    }
  };

  /**
   * Verify results by comparing to pre-stored results (for now .npy only)
   */
  template <typename T>
  struct FileVerifier : Verifier {
    std::shared_ptr<T[]> npy_data;

    explicit FileVerifier(std::string verify_npy) {
      cnpy::NpyArray npy = cnpy::npy_load(verify_npy);
      GALOIS_ASSERT(npy.word_size == sizeof(T),
                    "wrong data type: should be float64/double");
      npy_data = std::reinterpret_pointer_cast<T[]>(npy.data_holder);
      GALOIS_ASSERT(npy_data,
                    "Failed to load pre-stored results for verification!");
      galois::gPrint("Pre-stored results loaded for verification.\n");
    }
    ~FileVerifier() = default;

    inline data_t get(GNode node) const { return npy_data[node]; }

    void alert(GNode node, data_t old_val, data_t new_val, double error) const {
      auto [x, y] = id2xy(node);
      galois::gDebug("Results mismatch: nodeID=", node, " xy=(", x, ",", y,
                     ") exp_val=", old_val, " ref_val=", new_val,
                     " reldiff=", error, "\n");
    }
  };

public:
  explicit Solver(Graph& g) : graph(g) {}

  /**
   * Pick boundary nodes and assign them to the boundary list.
   */
  void assignBoundary() {
    switch (source_type) {
    case scatter: {
      std::size_t n = xy2id({0., 0.});
      boundary.push(n);
    } break;
    case analytical: {
      galois::do_all(
          galois::iterate(0ul, NUM_CELLS),
          [&](GNode node) noexcept {
            if (node > NUM_CELLS)
              return;

            if (NonNegativeRegion(id2xy(node))) {
              for (auto e : graph.edges(node, UNPROTECTED)) {
                GNode dst = graph.getEdgeDst(e);
                if (!NonNegativeRegion(id2xy(dst))) {
                  boundary.push(node);
                  break;
                }
              }
            }
          },
          galois::loopname("assignBoundary"));
    } break;
    default:
      GALOIS_DIE("Unknown boundary type.");
    }

    GALOIS_ASSERT(!boundary.empty(), "Boundary not found!");

#ifndef NDEBUG // Print boundary: node ID (bx, by)
    galois::gDebug("vvvvvvvv boundary vvvvvvvv");
    for (GNode b : boundary) {
      auto [x, y] = id2xy(b);
      galois::gDebug(b, " (", x, ", ", y, ")");
    }
    galois::gDebug("^^^^^^^^ boundary ^^^^^^^^");
#endif
  }

  /**
   * Initialize boundary nodes with boundary conditions.
   */
  void initBoundary() {
    galois::do_all(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](const BL::value_type& node) noexcept {
          auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
          curData.solution.store(BoundaryCondition(id2xy(node)),
                                 std::memory_order_relaxed);
        },
        galois::no_stats(), galois::loopname("initializeBoundary"));
  }

  /**
   * Start interface.
   */
  template <bool CONCURRENT, typename WL>
  void runAlgo() {
    WL initBag;

    using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                           galois::StdForEach>::type;
    Loop loop;

    loop(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](GNode node) noexcept {
          auto& curData = graph.getData(node, galois::MethodFlag::UNPROTECTED);
          pushWrap(initBag,
                   {curData.solution.load(std::memory_order_relaxed), node});
        },
        galois::loopname("FirstIteration"));

    if (initBag.empty()) {
      galois::gPrint("No cell other than boundary nodes to be processed.\n");
      return;
#ifndef NDEBUG // Print work items for the first round
    } else {
      galois::gDebug("vvvvvvvv init band vvvvvvvv");
      for (auto [k, i] : initBag) {
        auto [x, y] = id2xy(i);
        galois::gDebug(k, " - ", i, " (", x, " ", y, "): arrival_time=",
                       graph.getData(i, UNPROTECTED)
                           .solution.load(std::memory_order_relaxed));
      }
      galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
    }

    FastMarching<CONCURRENT>(initBag);
  }

  /**
   * Start interface.
   */
  template <bool CONCURRENT, typename WL>
  void FIM() {
    WL initBag;

    using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                           galois::StdForEach>::type;
    Loop loop;

    loop(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](GNode node) noexcept {
          for (auto e : graph.edges(node, UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(e);
            if (dst >= NUM_CELLS)
              continue;
            auto& dst_data =
                graph.getData(dst, galois::MethodFlag::UNPROTECTED);
            // will have duplicates but doesn't matter
            pushWrap(initBag,
                     {dst_data.solution.load(std::memory_order_relaxed), dst});
          }
        },
        galois::loopname("FirstIteration"));

    if (initBag.empty()) {
      galois::gPrint("No cell other than boundary nodes to be processed.\n");
      return;
#ifndef NDEBUG // Print work items for the first round
    } else {
      galois::gDebug("vvvvvvvv init band vvvvvvvv");
      for (auto [k, i] : initBag) {
        auto [x, y] = id2xy(i);
        galois::gDebug(k, " - ", i, " (", x, " ", y, "): arrival_time=",
                       graph.getData(i, UNPROTECTED)
                           .solution.load(std::memory_order_relaxed));
      }
      galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
    }

    FastIterativeMethod<CONCURRENT>(&initBag);
  }

  /**
   * Verify results.
   */
  void sanityCheck() {
    if (skipVerify) {
      galois::gPrint("Skip result verification.");
      return;
    }

    std::unique_ptr<Verifier> verifier;
    if (!verify_npy.empty())
      verifier.reset(new FileVerifier<data_t>(verify_npy));
    else
      verifier.reset(new SelfVerifier(*this));

    galois::GReduceMax<double> max_error;

    galois::do_all(
        galois::iterate(graph.begin(), graph.end()),
        [&](GNode node) noexcept {
          if (node >= NUM_CELLS)
            return;

          auto& my_data = graph.getData(node, UNPROTECTED);
          if (my_data.solution.load(std::memory_order_relaxed) == INF) {
            galois::gPrint("Untouched cell: ", node, "\n");
          }
          if (my_data.solution.load(std::memory_order_relaxed) == 0.) {
            // Skip checking starting nodes.
            return;
          }
          data_t new_val = verifier->get(node);
          if (data_t old_val = my_data.solution.load(std::memory_order_relaxed);
              new_val != old_val) {
            data_t error =
                std::abs(new_val - old_val) / std::max(new_val, old_val);
            max_error.update(error);
            if (error > tolerance) {
              verifier->alert(node, old_val, new_val, error);
            }
          }
        },
        galois::no_stats(), galois::loopname("sanityCheck"));

    galois::gPrint("MAX ERROR: ", max_error.reduce(), "\n");
  }

  void exportResults() {
    if (!output_npy.empty())
      DumpToNpy(graph);
    else if (!output_csv.empty())
      DumpToCsv(graph);
  }
};

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  galois::gDebug(ALGO_NAMES[algo]);

  Graph graph;

  //! Grids generation and cell initialization
  setupGrids(graph);

  Solver solver(graph);
  //! Boundary assignment and initialization
  // TODO better way for boundary settings?
  solver.assignBoundary();
  solver.initBoundary();

  //! Go!
  galois::StatTimer Tmain;
  Tmain.start();

  switch (algo) {
  case serial:
    solver.runAlgo<false, HeapTy>();
    break;
  case parallel:
    solver.runAlgo<true, WL>();
    break;
  case fim:
    solver.FIM<true, WL>();
    break;
  default:
    std::abort();
  }

  Tmain.stop();

  solver.sanityCheck();
  solver.exportResults();

  return 0;
}
