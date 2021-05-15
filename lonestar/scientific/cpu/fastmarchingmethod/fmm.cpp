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
#include <galois/DynamicBitset.h>

#ifdef GALOIS_ENABLE_VTUNE
#include "galois/runtime/Profile.h"
#endif

#include "Lonestar/BoilerPlate.h"

static char const* name = "Eikonal Solvers";
static char const* desc =
    "Eikonal equation solver "
    "(https://en.wikipedia.org/wiki/Fast_marching_method)";
static char const* url = "";

#include "fastmarchingmethod.h"
#define DIM_LIMIT 2 // 2-D specific

#define WORK_ITEM_STAT true

enum Algo { fmm = 0, fim, fsm };
enum SourceType { scatter = 0, analytical };

const char* const ALGO_NAMES[] = {
    "Fast Marching Method", "Fast Iterative Method", "Fast Sweeping Method"};

static llvm::cl::OptionCategory catAlgo{"1. Algorithmic Options"};
static llvm::cl::opt<Algo> algo{
    "algo",
    llvm::cl::value_desc("algo"),
    llvm::cl::desc("Choose an algorithm (default parallel):"),
    llvm::cl::values(clEnumVal(fmm, "Fast Marching Method"),
                     clEnumVal(fim, "Fast Iterative Method"),
                     clEnumVal(fsm, "Fast Sweeping Method")),
    llvm::cl::init(fmm),
    llvm::cl::cat(catAlgo)};
static llvm::cl::opt<bool> useSerial{
    "serial", llvm::cl::desc("Use serial implementation"),
    llvm::cl::cat(catAlgo)};
static llvm::cl::opt<bool> useDense{
    "dense", llvm::cl::desc("Use dense representation of grids"),
    llvm::cl::cat(catAlgo)};
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
static llvm::cl::opt<bool> conservative{
    "conservative",
    llvm::cl::desc("Alternative method to get determinism: conservative "
                   "approach to pushing redundant work items (dense FMM only)"),
    llvm::cl::cat(catAlgo)};

static llvm::cl::OptionCategory catInput{"2. Input Options"};
static llvm::cl::opt<SourceType> source_type{
    "sourceFormat", llvm::cl::desc("Choose an source format:"),
    llvm::cl::values(clEnumVal(scatter, "a set of discretized points"),
                     clEnumVal(analytical, "boundary in a analytical form")),
    llvm::cl::init(scatter), llvm::cl::cat(catInput)};
static llvm::cl::opt<std::string> input_segy{
    "segy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc("Use SEG-Y (rev 1) file as input speed map. NOTE: This will "
                   "determine the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput)};
static llvm::cl::opt<std::string> input_npy{
    "inpy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use npy file (dtype=float32) as input speed map. NOTE: This will "
        "determine the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput)};
static llvm::cl::opt<std::string> input_csv{
    "icsv", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use csv file as input speed map. NOTE: Current implementation "
        "requires explicit definition of the size on each dimensions"),
    llvm::cl::init(""), llvm::cl::cat(catInput)};
static llvm::cl::opt<std::string> verify_npy{
    "vnpy",
    llvm::cl::desc("Canonical results for verification in a npy format file"),
    llvm::cl::init(""), llvm::cl::cat(catInput)};
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
    domain_shape{
        "ij", llvm::cl::value_desc("nrows,ncols"),
        llvm::cl::desc("Size of each dimensions as a comma-separated array "
                       "(support up to 2-D)"),
        llvm::cl::cat(catInput)};
static llvm::cl::opt<double> speed_factor{
    "sf", llvm::cl::desc("speed factor multiplied to speed value"),
    llvm::cl::init(1.), llvm::cl::cat(catInput)};

static llvm::cl::OptionCategory catOutput{"3. Output Options"};
static llvm::cl::opt<std::string> output_csv{
    "ocsv", llvm::cl::desc("Export results to a csv format file"),
    llvm::cl::init(""), llvm::cl::cat(catOutput)};
static llvm::cl::opt<std::string> output_npy{
    "onpy", llvm::cl::desc("Export results to a npy format file"),
    llvm::cl::init(""), llvm::cl::cat(catOutput)};

static llvm::cl::OptionCategory catDisc{"4. Discretization options"};
static llvm::cl::opt<std::vector<double>, false,
                     NumVecParser<double, DIM_LIMIT>>
    steps{"h", llvm::cl::value_desc("dx,dy"),
          llvm::cl::desc("Spacing between discrete samples of each dimensions; "
                         "type as a comma-separated pair"),
          llvm::cl::init(std::vector<double>{1., 1.}), llvm::cl::cat(catDisc)};
static llvm::cl::opt<std::vector<double>, false,
                     NumVecParser<double, DIM_LIMIT>>
    domain_start{"oxy", llvm::cl::value_desc("x0,y0"),
                 llvm::cl::desc("Coordinate of cell [0, 0] "
                                "<comma-separated array, default (0., 0.)>"),
                 llvm::cl::init(std::vector<double>{0., 0.}),
                 llvm::cl::cat(catDisc)};

static std::size_t nx, ny, NUM_CELLS;
static data_t dx, dy, xa, xb, ya, yb;

constexpr galois::MethodFlag UNPROTECTED = galois::MethodFlag::UNPROTECTED;

// TODO: still necessary?
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
using NodeListTy    = galois::InsertBag<GNode>;
using UpdateRequest = std::pair<data_t, GNode>;
using HeapTy        = FMMHeapWrapper<
    std::multimap<UpdateRequest::first_type, UpdateRequest::second_type>>;
using ReqListTy = galois::InsertBag<UpdateRequest>;
#define CHUNK_SIZE 128

// These reference the input parameters as global data so they have to be
// included after the input parameters are defined.
#include "util/input.hh"
#include "util/output.hh"

/**
 * Generic Eikonal solver type.
 *
 * The public interfaces show the work flow of an Eikonal solver.
 */
class EikonalSolver {
protected:
  struct Verifier {
    virtual ~Verifier() {}
    virtual data_t get(std::size_t) const                         = 0;
    virtual void alert(std::size_t, data_t, data_t, double) const = 0;
  };

public:
  virtual void determineBoundary() = 0;
  virtual void initBoundary()      = 0;
  virtual void runAlgo()           = 0;
  virtual void sanityCheck()       = 0;
  virtual void exportResults() {
    if (!output_npy.empty() || !output_csv.empty())
      galois::gWarn("exportResults: not implemented.");
  }
};

/**
 * Solver implementation w/ dense layout (SoA).
 */
class DenseSolver : public EikonalSolver {
protected:
  galois::LargeArray<std::atomic<data_t>> u_buffer;
  std::shared_ptr<float[]> speed;
  std::size_t source_i, source_j; // single source TODO: source bag

  // Use a lambda for indexing into the speed function.
  float f(size_t i, size_t j) noexcept {
    assert(i < ny && j < nx && "Out of bounds access.");
    return std::cref(speed[i * nx + j]) * speed_factor;
  };

  // Also use a lambda for indexing into the solution.
  std::atomic<data_t>& u(size_t i, size_t j) noexcept {
    assert(i < ny && j < nx && "Out of bounds access.");
    std::atomic<data_t>& ret = u_buffer[i * nx + j];
    return ret;
  };

  std::array<data_t, 4> gather_neighbors(std::size_t i,
                                         std::size_t j) noexcept {
    data_t uN = std::numeric_limits<data_t>::infinity(), uS = uN, uE = uS,
           uW = uE;
    if (i)
      uS = u(i - 1, j).load(std::memory_order_relaxed);
    if (i < ny - 1)
      uN = u(i + 1, j).load(std::memory_order_relaxed);
    if (j)
      uW = u(i, j - 1).load(std::memory_order_relaxed);
    if (j < nx - 1)
      uE = u(i, j + 1).load(std::memory_order_relaxed);
    return {uN, uS, uE, uW};
  };

  data_t difference_scheme(data_t f, data_t h, data_t uH, data_t uV) noexcept {
    data_t div = h / f, dif = uH - uV;
    if (!std::isinf(uH) && !std::isinf(uV) && std::abs(dif) < div) {
      // This variant of the differencing scheme is set up to avoid catastrophic
      // cancellation. The resulting precision loss can cause violations of the
      // nonincreasing property of the operator.
      data_t current  = .5 * (uH + uV + std::sqrt(2 * div * div - dif * dif));
      data_t previous = std::numeric_limits<data_t>::infinity();
      if (strict) {
        // Use Newton's method to further mitigate violation of the
        // nonincreasing property caused by finite-precision arithmetic. This
        // loop is likely to only take one or two iterations.
        while (current < previous) {
          previous = current;
          current += .5 * ((div * div - (current - uH) * (current - uH) -
                            (current - uV) * (current - uV)) /
                           ((current - uH) + (current - uV)));
        }
        return current;
      } else {
        return current;
      }
    }
    return std::min(uH, uV) + div;
  };

  data_t solve(std::size_t i, std::size_t j) {
    auto const [uN, uS, uE, uW] = gather_neighbors(i, j);
    auto const uH = std::min(uE, uW), uV = std::min(uN, uS);
    return difference_scheme(f(i, j), dx, uH, uV);
  }

  std::array<std::tuple<bool, std::size_t, std::size_t>, 4>
  neighbors(std::size_t i, std::size_t j) {
    return {std::make_tuple(i, i - 1, j), std::make_tuple(i < ny - 1, i + 1, j),
            std::make_tuple(j, i, j - 1),
            std::make_tuple(j < nx - 1, i, j + 1)};
  };

  friend struct SelfVerifier;
  friend struct FileVerifier;
  /**
   * Verify results by re-apply operators on each nodes and observe the
   * differences.
   */
  struct SelfVerifier : Verifier {
    DenseSolver& p;

    explicit SelfVerifier(DenseSolver& p_) : p(p_) {}
    ~SelfVerifier() = default;

    data_t get(std::size_t id) const {
      auto const [i, j] = unravel_index(id);
      if (i == p.source_i && j == p.source_j)
        return 0.; // TODO: boundary condition
      auto const [uN, uS, uE, uW] = p.gather_neighbors(i, j);
      auto const uH = std::min(uE, uW), uV = std::min(uN, uS);
      return p.difference_scheme(p.f(i, j), dx, uH, uV);
    }

    void alert(std::size_t id, data_t old_val, data_t new_val,
               double error) const {
      auto const [i, j] = unravel_index(id);
      galois::gPrint("Error bound violated at cell ", id, " (", i, " ", j,
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

    inline data_t get(std::size_t id) const { return npy_data[id]; }

    void alert(std::size_t id, data_t old_val, data_t new_val,
               double error) const {
      auto const [i, j] = unravel_index(id);
      galois::gDebug("Results mismatch: nodeID=", id, " xy=(", i, ",", j,
                     ") exp_val=", old_val, " ref_val=", new_val,
                     " reldiff=", error, "\n");
    }
  };

public:
  DenseSolver() {
    if (input_npy.empty())
      GALOIS_DIE("Dense version must be fed with npy source.");
    if (dx != dy)
      GALOIS_DIE("Dense version only deals with square cells.");

    /*
     * setupGrids
     */
    cnpy::NpyArray npy = cnpy::npy_load(input_npy);
    if (npy.word_size != sizeof(float))
      GALOIS_DIE("Wrong data type.");
    if (npy.shape.size() != 2)
      GALOIS_DIE("Data should be 2-D.");
    speed = std::reinterpret_pointer_cast<float[]>(npy.data_holder);
    GALOIS_ASSERT(speed, "Failed to load speed data.");
    SetKnobs({npy.shape[1], npy.shape[0]}); // convert j-col,i-row to nx,ny
    u_buffer.create(NUM_CELLS);
    std::fill(u_buffer.begin(), u_buffer.end(),
              std::numeric_limits<double>::infinity());
  }

  // TODO: multi-source
  // Old input:
  // source_coordinates("d", llvm::cl::value_desc("i,j"),
  // llvm::cl::desc("Indices of the source point."));
  void determineBoundary() {
    switch (source_type) {
    case scatter: {
      // if (source_coordinates.empty()) {
      source_i = source_j = 0u;
      // } else {
      //   source_i = source_coordinates[0];
      //   source_j = source_coordinates[1];
      // }
    } break;
    case analytical: {
      // TODO: not implemented
      GALOIS_DIE("Multi-source not implemented.");
    } break;
    default:
      GALOIS_DIE("Unknown boundary type.");
    }
  }

  // TODO: multi-source
  void initBoundary() {
    u(source_i, source_j).store(0., std::memory_order_relaxed);
  }

  virtual void runAlgo() = 0;

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
    galois::GReduceMax<double> max_u;
    max_error.reset();
    max_u.reset();

    galois::do_all(
        galois::iterate(std::size_t(0u), NUM_CELLS),
        [&](std::size_t id) noexcept {
          auto const [i, j] = unravel_index(id);
          if (i == source_i && j == source_j)
            return;
          data_t const old_u = u(i, j).load(std::memory_order_relaxed);
          if (old_u == INF)
            GALOIS_DIE("Untouched Cell:", id);
          max_u.update(old_u);
          data_t const new_u = verifier->get(id);
          if (new_u != old_u) {
            data_t error = std::abs(new_u - old_u) / std::max(new_u, old_u);
            max_error.update(error);
            if (strict || error > tolerance) {
              verifier->alert(id, old_u, new_u, error);
              GALOIS_DIE("Failed correctness check");
            }
          }
        },
        galois::no_stats(), galois::loopname("DenseSanityCheck"));

    galois::gPrint("Max arrival time: ", max_u.reduce(), "\n");
    galois::runtime::reportStat_Single("(NULL)", "MaxError",
                                       max_error.reduce());
  }
};

/**
 * Solver implementation w/ CSR layout (AoS)
 */
class CSRSolver : public EikonalSolver {
protected:
  Graph graph;
  NodeListTy boundary;
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
  template <typename NodeData, typename It>
  std::pair<data_t, std::array<UpdateRequest, 4>>
  quickUpdate(NodeData& my_data, It edge_begin, [[maybe_unused]] It edge_end) {
    const auto f = my_data.speed;
    assert(dx == dy);
    auto h  = dx;
    GNode N = graph.getEdgeDst(*edge_begin),
          S = graph.getEdgeDst(*(++edge_begin)),
          E = graph.getEdgeDst(*(++edge_begin)),
          W = graph.getEdgeDst(*(++edge_begin));
    assert(++edge_begin == edge_end);
    data_t u_N = graph.getData(N).solution.load(std::memory_order_relaxed),
           u_S = graph.getData(S).solution.load(std::memory_order_relaxed),
           u_E = graph.getData(E).solution.load(std::memory_order_relaxed),
           u_W = graph.getData(W).solution.load(std::memory_order_relaxed);
    std::array<UpdateRequest, 4> adj_data{
        std::make_pair(u_N, N), std::make_pair(u_S, S), std::make_pair(u_E, E),
        std::make_pair(u_W, W)};
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
      return {std::min(u_H, u_V) + div, adj_data};
    }
    // Use this particular form of the differencing scheme to mitigate
    // precision loss from catastrophic cancellation inside the square root.
    // The loss of precision breaks the monotonicity guarantees of the
    // differencing operator. This mitigatest that issue somewhat.
    static double sqrt2 = std::sqrt(2);
    data_t s2div = sqrt2 * div, dif = u_H - u_V;
    return {.5 * (u_H + u_V + std::sqrt((s2div + dif) * (s2div - dif))),
            adj_data};
  }

  friend struct SelfVerifier;
  friend struct FileVerifier;
  /**
   * Verify results by re-apply operators on each nodes and observe the
   * differences.
   */
  struct SelfVerifier : Verifier {
    CSRSolver& p;

    explicit SelfVerifier(CSRSolver& p_) : p(p_) {}
    ~SelfVerifier() = default;

    data_t get(std::size_t node) const {
      return p
          .quickUpdate(p.graph.getData(node, UNPROTECTED),
                       p.graph.edge_begin(node, UNPROTECTED),
                       p.graph.edge_end(node, UNPROTECTED))
          .first;
    }

    void alert(std::size_t node, data_t old_val, data_t new_val,
               double error) const {
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

    inline data_t get(std::size_t node) const { return npy_data[node]; }

    void alert(std::size_t node, data_t old_val, data_t new_val,
               double error) const {
      auto [x, y] = id2xy(node);
      galois::gDebug("Results mismatch: nodeID=", node, " xy=(", x, ",", y,
                     ") exp_val=", old_val, " ref_val=", new_val,
                     " reldiff=", error, "\n");
    }
  };

public:
  explicit CSRSolver() { setupGrids(graph); }

  /**
   * Pick boundary nodes and assign them to the boundary list.
   */
  void determineBoundary() {
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
          galois::loopname("determineBoundary"));
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
        [&](const NodeListTy::value_type& node) noexcept {
          auto& curData = graph.getData(node, UNPROTECTED);
          curData.solution.store(BoundaryCondition(id2xy(node)),
                                 std::memory_order_relaxed);
        },
        galois::no_stats(), galois::loopname("initializeBoundary"));
  }

  /**
   * Start interface.
   */
  virtual void runAlgo() = 0;

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
          if (my_data.solution.load(std::memory_order_relaxed) ==
              0.) { // TODO: adapt to boundary condition or check with node id
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

    galois::runtime::reportStat_Single("(NULL)", "MaxError",
                                       max_error.reduce());
  }

  void exportResults() {
    if (!output_npy.empty())
      DumpToNpy(graph);
    else if (!output_csv.empty())
      DumpToCsv(graph);
  }
};

/**
 * Fast Marching Method
 */
template <typename LayoutTy, bool CONCURRENT, typename WorkListTy>
class FastMarchingMethod : public LayoutTy {};

/**
 * Dense impl. of FMM
 */
template <bool CONCURRENT, typename WorkListTy>
class FastMarchingMethod<DenseSolver, CONCURRENT, WorkListTy>
    : public DenseSolver {
  void exec(WorkListTy& initial);

public:
  template <typename... Args>
  FastMarchingMethod(Args&&... args)
      : DenseSolver(std::forward<Args>(args)...) {}

  /**
   * Start interface.
   */
  void runAlgo() {
    galois::InsertBag<UpdateRequest> initial;
    if (source_i)
      initial.push(UpdateRequest{0., ravel_index(source_i - 1, source_j)});
    if (source_i < ny - 1)
      initial.push(UpdateRequest{0., ravel_index(source_i + 1, source_j)});
    if (source_j)
      initial.push(UpdateRequest{0., ravel_index(source_i, source_j - 1)});
    if (source_j < nx - 1)
      initial.push(UpdateRequest{0., ravel_index(source_i, source_j + 1)});

    exec(initial);
  }
};

/**
 * Dense parallel execution of FMM
 */
template <>
void FastMarchingMethod<DenseSolver, true, ReqListTy>::exec(
    ReqListTy& initial) {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  auto indexer = [&](UpdateRequest item) noexcept -> std::size_t {
    return std::round(rounding_scale * item.first);
  };
  using PSchunk = galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>;
  using OBIM =
      galois::worklists::OrderedByIntegerMetric<decltype(indexer), PSchunk>;

  galois::for_each(
      galois::iterate(initial),
      [&](const UpdateRequest& work_item, auto& context) noexcept {
        auto const [i, j] = unravel_index(work_item.second);
        auto cur_u        = u(i, j).load(std::memory_order_relaxed);
        if (cur_u < work_item.first) {
#if (WORK_ITEM_STAT)
          emptyWork += 1;
#endif
          return;
        }
#if (WORK_ITEM_STAT)
        if (cur_u != INF)
          badWork += 1;
#endif
        data_t uN, uS, uE, uW, new_u;
        if (conservative) {
          do {
            auto u_neib   = gather_neighbors(i, j);
            uN            = u_neib[0];
            uS            = u_neib[1];
            uE            = u_neib[2];
            uW            = u_neib[3];
            auto const uH = std::min(uE, uW), uV = std::min(uN, uS);
            new_u = difference_scheme(f(i, j), dx, uH, uV);
            if (std::isnan(new_u))
              GALOIS_DIE(
                  "Differencing scheme returned NaN. This may result from "
                  "insufficient precision or from bad input.");
            if (new_u == cur_u)
              return;
          } while (!u(i, j).compare_exchange_weak(cur_u, new_u,
                                                  std::memory_order_relaxed));
        } else {
          // FIXME: ugly and repetitive code
          auto u_neib   = gather_neighbors(i, j);
          uN            = u_neib[0];
          uS            = u_neib[1];
          uE            = u_neib[2];
          uW            = u_neib[3];
          auto const uH = std::min(uE, uW), uV = std::min(uN, uS);
          new_u = difference_scheme(f(i, j), dx, uH, uV);
          if (std::isnan(new_u))
            GALOIS_DIE("Differencing scheme returned NaN. This may result from "
                       "insufficient precision or from bad input.");
          do {
            if (new_u >= cur_u)
              return;
          } while (!u(i, j).compare_exchange_weak(cur_u, new_u,
                                                  std::memory_order_relaxed));
        }

        if (i && uS > new_u)
          context.push(UpdateRequest{new_u, ravel_index(i - 1, j)});
        if (i < ny - 1 && uN > new_u)
          context.push(UpdateRequest{new_u, ravel_index(i + 1, j)});
        if (j && uW > new_u)
          context.push(UpdateRequest{new_u, ravel_index(i, j - 1)});
        if (j < nx - 1 && uE > new_u)
          context.push(UpdateRequest{new_u, ravel_index(i, j + 1)});
      },
      galois::loopname("DenseFMM"), galois::disable_conflict_detection(),
      galois::wl<OBIM>(indexer));

#if (WORK_ITEM_STAT)
  galois::runtime::reportStat_Single("DenseFMM", "EmptyWork",
                                     emptyWork.reduce());
  galois::runtime::reportStat_Single("DenseFMM", "BadWork", badWork.reduce());
#endif
}

// CSR
template <bool CONCURRENT, typename WorkListTy>
class FastMarchingMethod<CSRSolver, CONCURRENT, WorkListTy> : public CSRSolver {
  void exec(WorkListTy& wl);

public:
  template <typename... Args>
  FastMarchingMethod(Args&&... args) : CSRSolver(std::forward<Args>(args)...) {}

  void runAlgo() {
    WorkListTy initial;

    using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                           galois::StdForEach>::type;
    Loop loop;

    loop(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](GNode node) noexcept {
          auto& curData = graph.getData(node, UNPROTECTED);
          data_t u      = curData.solution.load(std::memory_order_relaxed);
          for (auto e : graph.edges(node, UNPROTECTED)) {
            auto dst = graph.getEdgeDst(e);
            if (dst >= NUM_CELLS)
              continue;
            auto& dst_data = graph.getData(dst, UNPROTECTED);
            if (dst_data.solution.load(std::memory_order_relaxed) == INF)
              pushWrap(initial, {u, dst});
          }
        },
        galois::loopname("FirstIteration"));

    if (initial.empty()) {
      galois::gPrint("No cell other than boundary nodes to be processed.\n");
      return;
#ifndef NDEBUG // Print work items for the first round
    } else {
      galois::gDebug("vvvvvvvv init band vvvvvvvv");
      for (auto [k, i] : initial) {
        auto [x, y] = id2xy(i);
        galois::gDebug(k, " - ", i, " (", x, " ", y, "): arrival_time=",
                       graph.getData(i, UNPROTECTED)
                           .solution.load(std::memory_order_relaxed));
      }
      galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
    }

    exec(initial);
  }
};

template <>
void FastMarchingMethod<CSRSolver, true, ReqListTy>::exec(ReqListTy& wl) {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  auto PullOp = [&](const auto& item, auto& wl) {
    // using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
    // typename ItemTy::second_type node;
    // std::tie(std::ignore, node) = item;
    auto [old_val, node] = item;
    galois::gDebug(old_val, " ", node);
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData  = graph.getData(node, UNPROTECTED);
    data_t cur_val = curData.solution.load(std::memory_order_relaxed);

    // Ignore stale repetitive work items
    if (cur_val < old_val) {
      return;
    }

    auto [new_val, adj_data] =
        quickUpdate(curData, graph.edge_begin(node, UNPROTECTED),
                    graph.edge_end(node, UNPROTECTED));

    do {
      if (new_val >= cur_val)
        return;
    } while (!curData.solution.compare_exchange_weak(
        cur_val, new_val, std::memory_order_relaxed));

    // Push neighbors
    for (auto& p : adj_data) {
      if (p.second >= NUM_CELLS)
        continue;
      // Given that the arrival time propagation is non-descending
      if (p.first > new_val) {
        wl.push(UpdateRequest{
            new_val, p.second}); // have to use new_val which is the
                                 // written value to avoid missing update
      }
    }
  };

  //! Run algo
  auto indexer = [&](UpdateRequest item) noexcept -> std::size_t {
    return std::round(rounding_scale * item.first);
  };
  using PSchunk = galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>;
  using OBIM =
      galois::worklists::OrderedByIntegerMetric<decltype(indexer), PSchunk>;

#ifdef GALOIS_ENABLE_VTUNE
  galois::runtime::profileVtune(
      [&]() {
#endif
        galois::for_each(galois::iterate(wl.begin(), wl.end()), PullOp,
                         galois::disable_conflict_detection(),
                         // galois::no_stats(),
                         galois::wl<OBIM>(indexer), galois::loopname("FMM"));
#ifdef GALOIS_ENABLE_VTUNE
      },
      "FMM_VTune");
#endif

#if (WORK_ITEM_STAT)
  galois::runtime::reportParam("Statistics", "EmptyWork", emptyWork.reduce());
  galois::runtime::reportParam("Statistics", "BadWork", badWork.reduce());
#endif
}

template <>
void FastMarchingMethod<CSRSolver, false, HeapTy>::exec(HeapTy& wl) {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  auto PushOp = [&](const auto& item, auto& wl) {
    // using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
    // typename ItemTy::second_type node;
    // std::tie(std::ignore, node) = item;
    auto [old_val, node] = item;
    galois::gDebug(old_val, " ", node);
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData        = graph.getData(node, UNPROTECTED);
    const data_t cur_val = curData.solution.load(std::memory_order_relaxed);

    // Ignore stale repetitive work items
    if (cur_val < old_val) {
      return;
    }

    // UpdateNeighbors
    for (auto e : graph.edges(node, UNPROTECTED)) {
      GNode dst = graph.getEdgeDst(e);

      if (dst >= NUM_CELLS) {
        continue;
      }
      auto& dstData = graph.getData(dst, UNPROTECTED);

      // Given that the arrival time propagation is non-descending
      data_t old_neighbor_val =
          dstData.solution.load(std::memory_order_relaxed);
      if (old_neighbor_val <= cur_val) {
#if (WORK_ITEM_STAT)
        emptyWork += 1;
#endif
        continue;
      }

      auto [sln_temp, adj_data] =
          quickUpdate(dstData, graph.edge_begin(dst, UNPROTECTED),
                      graph.edge_end(dst, UNPROTECTED));

      do {
        if (sln_temp >= old_neighbor_val){
#if (WORK_ITEM_STAT)
          emptyWork += 1;
#endif
          goto continue_outer;
        }
      } while (!dstData.solution.compare_exchange_weak(
          old_neighbor_val, sln_temp, std::memory_order_relaxed));
#if (WORK_ITEM_STAT)
      if (!std::isinf(old_neighbor_val))
        badWork += 1;
#endif
      wl.push(UpdateRequest{sln_temp, dst}, old_neighbor_val);
    continue_outer:;
    }
  };

  //! Run algo
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

#if (WORK_ITEM_STAT)
  galois::runtime::reportParam("Statistics", "EmptyWork", emptyWork.reduce());
  galois::runtime::reportParam("Statistics", "BadWork", badWork.reduce());
#endif
}

template<>
void FastMarchingMethod<CSRSolver, false, HeapTy>::runAlgo() {
  HeapTy initial;
  initial.push(std::pair<double, unsigned int>(0., 0u));

  exec(initial);
}

/**
 * Fast Iterative Method
 */
template <typename LayoutTy, bool CONCURRENT, typename WorkListTy>
class FastIterativeMethod : public LayoutTy {};

template <bool CONCURRENT, typename WorkListTy>
class FastIterativeMethod<DenseSolver, CONCURRENT, WorkListTy>
    : public DenseSolver {
  void exec(std::unique_ptr<WorkListTy>& wl);

public:
  template <typename... Args>
  FastIterativeMethod(Args&&... args)
      : DenseSolver(std::forward<Args>(args)...) {}

  void runAlgo() {
    std::unique_ptr<WorkListTy> initial(new WorkListTy());

    if (source_i)
      initial->push(UpdateRequest{0., ravel_index(source_i - 1, source_j)});
    if (source_i < ny - 1)
      initial->push(UpdateRequest{0., ravel_index(source_i + 1, source_j)});
    if (source_j)
      initial->push(UpdateRequest{0., ravel_index(source_i, source_j - 1)});
    if (source_j < nx - 1)
      initial->push(UpdateRequest{0., ravel_index(source_i, source_j + 1)});

    if (initial->empty()) {
      galois::gPrint("No cell other than boundary nodes to be processed.\n");
      return;
#ifndef NDEBUG // Print work items for the first round
    } else {
      galois::gDebug("vvvvvvvv init band vvvvvvvv");
      for (auto [k, id] : *initial) {
        auto const [i, j] = unravel_index(id);
        galois::gDebug(k, " - ", id, " (", i, " ", j, "): arrival_time=",
                       u(i, j).load(std::memory_order_relaxed));
      }
      galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
    }

    exec(initial);
  }
};

/**
 * Dense parallel execution of FIM
 */
template <>
void FastIterativeMethod<DenseSolver, true, ReqListTy>::exec(
    std::unique_ptr<ReqListTy>& wl) {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  // Jacobi iteration
  std::unique_ptr<ReqListTy> next(new ReqListTy());

  /**
   * Option 1: Allowing repetitive pushes w/o checking if it's in worklist
   */
  [[maybe_unused]] auto stateless_op = [&](const auto& work_item) {
    auto [old_u, id] = work_item;
    assert(id < NUM_CELLS && "Ghost Point!");
    auto const [i, j] = unravel_index(id);

    auto cur_u = u(i, j).load(std::memory_order_relaxed);
    if (cur_u < old_u) {
#if (WORK_ITEM_STAT)
      emptyWork += 1;
#endif
      return;
    }
    data_t const new_u = solve(i, j);
    if (std::isnan(new_u))
      GALOIS_DIE("Differencing scheme returned NaN. This may result from "
                 "insufficient precision or from bad input.");
    do {
      if (new_u > cur_u)
        return;
    } while (!u(i, j).compare_exchange_weak(cur_u, new_u,
                                            std::memory_order_relaxed));
#if (WORK_ITEM_STAT)
    if (cur_u != INF) {
      badWork += 1;
    }
#endif

    // "decreased" semantics implemented as "changed" due to
    // finite precision
    if (std::abs(cur_u - new_u) >= tolerance) { // not converged
      next->push(UpdateRequest{new_u, id});
      return;
    }

    for (auto [valid, ii, jj] : neighbors(i, j)) {
      if (!valid) // XXX: cannot identify those already in worklist
        continue;
      // TODO: similar code repeated
      auto cur_u         = u(ii, jj).load(std::memory_order_relaxed);
      data_t const new_u = solve(ii, jj);
      if (std::isnan(new_u))
        GALOIS_DIE("Differencing scheme returned NaN. This may "
                   "result from "
                   "insufficient precision or from bad input.");
      do {
        if (new_u > cur_u)
          goto fim_next_neighbor;
      } while (!u(ii, jj).compare_exchange_weak(cur_u, new_u,
                                                std::memory_order_relaxed));
#if (WORK_ITEM_STAT)
      if (cur_u != INF) {
        badWork += 1;
      }
#endif

      // "decreased" semantics implemented as "changed" due to
      // finite precision
      if (std::abs(cur_u - new_u)) {
        next->push(UpdateRequest{new_u, ravel_index(ii, jj)});
      }
    fim_next_neighbor:
      continue;
    }
  }; // end of stateless_op

  /**
   * Option 2: Maintaining state flag for checking if a node is in worklist
   */
  [[maybe_unused]] galois::DynamicBitSet isInWorkList;
  isInWorkList.resize(NUM_CELLS);
  isInWorkList.reset();
  galois::do_all(
      galois::iterate(wl->begin(), wl->end()),
      [&](const auto& work_item) {
        [[maybe_unused]] auto [prio, id] = work_item;
        isInWorkList.set(id);
        return;
      },
      // galois::steal(),
      // galois::no_stats(),
      // galois::chunk_size<CHUNK_SIZE>(),
      galois::loopname("DenseFIM_initFlags"));
  [[maybe_unused]] auto stateful_op = [&](const auto& work_item) {
    auto [old_u, id] = work_item;
    assert(id < NUM_CELLS && "Ghost Point!");
    if (!isInWorkList.reset(id))
      GALOIS_DIE("was not marked in list");
    auto const [i, j] = unravel_index(id);

    auto cur_u = u(i, j).load(std::memory_order_relaxed);
    if (cur_u < old_u) {
#if (WORK_ITEM_STAT)
      emptyWork += 1;
#endif
      galois::gWarn("Repeated item in worklist"); // No node locks
    }
    data_t const new_u = solve(i, j);
    if (std::isnan(new_u))
      GALOIS_DIE("Differencing scheme returned NaN. This may result from "
                 "insufficient precision or from bad input.");
    do {
      if (new_u > cur_u)
        return;
    } while (!u(i, j).compare_exchange_weak(cur_u, new_u,
                                            std::memory_order_relaxed));
#if (WORK_ITEM_STAT)
    if (cur_u != INF) {
      badWork += 1;
    }
#endif

    // "decreased" semantics implemented as "changed" due to
    // finite precision
    if (std::abs(cur_u - new_u) >= tolerance) { // not converged
      if (!isInWorkList.set(id))
        next->push(UpdateRequest{new_u, id});
      return;
    }

    for (auto [valid, ii, jj] : neighbors(i, j)) {
      if (!valid || isInWorkList.test(ravel_index(ii, jj)))
        continue;
      // TODO: similar code repeated
      auto cur_u         = u(ii, jj).load(std::memory_order_relaxed);
      data_t const new_u = solve(ii, jj);
      if (std::isnan(new_u))
        GALOIS_DIE("Differencing scheme returned NaN. This may "
                   "result from "
                   "insufficient precision or from bad input.");
      do {
        if (new_u > cur_u)
          goto fim_next_neighbor;
      } while (!u(ii, jj).compare_exchange_weak(cur_u, new_u,
                                                std::memory_order_relaxed));
#if (WORK_ITEM_STAT)
      if (cur_u != INF) {
        badWork += 1;
      }
#endif

      // "decreased" semantics implemented as "changed" due to
      // finite precision
      if (std::abs(cur_u - new_u)) {
        if (!isInWorkList.set(ravel_index(ii, jj)))
          next->push(UpdateRequest{new_u, ravel_index(ii, jj)});
      }
    fim_next_neighbor:
      continue;
    }
  }; // end of stateful_op

  //! Run algo
  std::size_t round = 0;
  while (!wl->empty()) {
    round++;
#ifdef GALOIS_ENABLE_VTUNE
    galois::runtime::profileVtune(
        [&]() {
#endif
          galois::do_all(galois::iterate(wl->begin(), wl->end()), stateless_op,
                         galois::steal(),
                         // galois::no_stats(),
                         galois::chunk_size<CHUNK_SIZE>(),
                         galois::loopname("DenseFIM"));
#ifdef GALOIS_ENABLE_VTUNE
        },
        "FIM_VTune");
#endif
    wl->clear();
    wl.swap(next);
  }

  galois::runtime::reportStat_Single("DenseFIM", "Rounds", round);
#if (WORK_ITEM_STAT)
  galois::runtime::reportStat_Single("DenseFIM", "EmptyWork",
                                     emptyWork.reduce());
  galois::runtime::reportStat_Single("DenseFIM", "BadWork", badWork.reduce());
#endif
}

template <bool CONCURRENT, typename WorkListTy>
class FastIterativeMethod<CSRSolver, CONCURRENT, WorkListTy>
    : public CSRSolver {
  void exec(WorkListTy* wl);

public:
  template <typename... Args>
  FastIterativeMethod(Args&&... args)
      : CSRSolver(std::forward<Args>(args)...) {}

  void runAlgo() {
    WorkListTy initial;

    using Loop = typename std::conditional<CONCURRENT, galois::DoAll,
                                           galois::StdForEach>::type;
    Loop loop;

    loop(
        galois::iterate(boundary.begin(), boundary.end()),
        [&](GNode node) noexcept {
          auto& curData = graph.getData(node, UNPROTECTED);
          data_t u      = curData.solution.load(std::memory_order_relaxed);
          for (auto e : graph.edges(node, UNPROTECTED)) {
            GNode dst = graph.getEdgeDst(e);
            if (dst >= NUM_CELLS)
              continue;
            auto& dst_data = graph.getData(dst, UNPROTECTED);
            // will have duplicates but doesn't matter
            if (dst_data.solution.load(std::memory_order_relaxed) == INF)
              pushWrap(initial, {u, dst});
          }
        },
        galois::loopname("FirstIteration"));

    if (initial.empty()) {
      galois::gPrint("No cell other than boundary nodes to be processed.\n");
      return;
#ifndef NDEBUG // Print work items for the first round
    } else {
      galois::gDebug("vvvvvvvv init band vvvvvvvv");
      for (auto [k, i] : initial) {
        auto [x, y] = id2xy(i);
        galois::gDebug(k, " - ", i, " (", x, " ", y, "): arrival_time=",
                       graph.getData(i, UNPROTECTED)
                           .solution.load(std::memory_order_relaxed));
      }
      galois::gDebug("^^^^^^^^ init band ^^^^^^^^");
#endif
    }

    exec(&initial);
  }
};

template <>
void FastIterativeMethod<CSRSolver, true, ReqListTy>::exec(ReqListTy* wl) {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  // Jacobi iteration
  ReqListTy* next = new ReqListTy();

  auto PushOp = [&](const auto& item) {
    // using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
    // typename ItemTy::second_type node;
    // std::tie(std::ignore, node) = item;
    auto [old_val, node] = item;
    galois::gDebug(old_val, " ", node);
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData        = graph.getData(node, UNPROTECTED);
    const data_t cur_val = curData.solution.load(std::memory_order_relaxed);

    // Ignore stale repetitive work items
    if (cur_val < old_val) {
      return;
    }

    auto [new_val, adj_data] =
        quickUpdate(curData, graph.edge_begin(node, UNPROTECTED),
                    graph.edge_end(node, UNPROTECTED));
    if (new_val != INF &&
        std::abs(new_val - galois::atomicMin(curData.solution, new_val)) >=
            tolerance) { // not converged
      galois::gDebug("not converged:", new_val, " ", node);
      next->push(UpdateRequest{new_val, node});
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
      if (old_neighbor_val <= cur_val) {
        continue;
      }

      auto [sln_temp, adj_data] =
          quickUpdate(dstData, graph.edge_begin(dst, UNPROTECTED),
                      graph.edge_end(dst, UNPROTECTED));

      do {
        if (sln_temp >= old_neighbor_val)
          goto continue_outer;
      } while (!dstData.solution.compare_exchange_weak(
          old_neighbor_val, sln_temp, std::memory_order_relaxed));
      assert(sln_temp < INF);
      next->push(UpdateRequest{sln_temp, dst});
    continue_outer:;
    }
  };

  //! Run algo
  using T      = typename ReqListTy::value_type;
  auto Indexer = [&](const T& item) {
    unsigned t = std::round(item.first * rounding_scale);
    // galois::gDebug(item.first, "\t", t, "\n");
    return t;
  };
  using PSchunk = galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>;
  using OBIM =
      galois::worklists::OrderedByIntegerMetric<decltype(Indexer), PSchunk>;

  while (!wl->empty()) {
#ifdef GALOIS_ENABLE_VTUNE
    galois::runtime::profileVtune(
        [&]() {
#endif
          galois::do_all(galois::iterate(wl->begin(), wl->end()), PushOp,
                         galois::steal(),
                         // galois::no_stats(),
                         galois::wl<OBIM>(Indexer), galois::loopname("FMM"));
#ifdef GALOIS_ENABLE_VTUNE
        },
        "FMM_VTune");
#endif
    wl->clear();
    std::swap(wl, next);
  }

  delete next;

#if (WORK_ITEM_STAT)
  galois::runtime::reportParam("Statistics", "EmptyWork", emptyWork.reduce());
  galois::runtime::reportParam("Statistics", "BadWork", badWork.reduce());
#endif
}

template <>
void FastIterativeMethod<CSRSolver, false, HeapTy>::exec(HeapTy* wl) {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  // Jacobi iteration
  HeapTy* next = new HeapTy();

  auto PushOp = [&](const auto& item) {
    // using ItemTy = std::remove_cv_t<std::remove_reference_t<decltype(item)>>;
    // typename ItemTy::second_type node;
    // std::tie(std::ignore, node) = item;
    auto [old_val, node] = item;
    galois::gDebug(old_val, " ", node);
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData        = graph.getData(node, UNPROTECTED);
    const data_t cur_val = curData.solution.load(std::memory_order_relaxed);

    // Ignore stale repetitive work items
    if (cur_val < old_val) {
      return;
    }

    auto [new_val, adj_data] =
        quickUpdate(curData, graph.edge_begin(node, UNPROTECTED),
                    graph.edge_end(node, UNPROTECTED));
    if (new_val != INF &&
        std::abs(new_val - galois::atomicMin(curData.solution, new_val)) >=
            tolerance) { // not converged
      galois::gDebug("not converged:", new_val, " ", node);
      pushWrap(*next, UpdateRequest{new_val, node}, old_val);
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
      if (old_neighbor_val <= cur_val) {
        continue;
      }

      auto [sln_temp, adj_data] =
          quickUpdate(dstData, graph.edge_begin(dst, UNPROTECTED),
                      graph.edge_end(dst, UNPROTECTED));

      do {
        if (sln_temp >= old_neighbor_val)
          goto continue_outer;
      } while (!dstData.solution.compare_exchange_weak(
          old_neighbor_val, sln_temp, std::memory_order_relaxed));
      assert(sln_temp < INF);
      pushWrap(*next, UpdateRequest{sln_temp, dst}, old_neighbor_val);
    continue_outer:;
    }
  };

  //! Run algo
  std::size_t num_iterations = 0;

  while (!wl->empty()) {

    PushOp(wl->pop());

    num_iterations++;

#ifndef NDEBUG
    sleep(1); // Debug pause
    galois::gDebug("\n********\n");
#endif

    wl->clear();
    std::swap(wl, next);
  }

  delete next;

  galois::gPrint("#iterarions: ", num_iterations, "\n");

#if (WORK_ITEM_STAT)
  galois::runtime::reportParam("Statistics", "EmptyWork", emptyWork.reduce());
  galois::runtime::reportParam("Statistics", "BadWork", badWork.reduce());
#endif
}

/**
 * Fast Sweeping Method
 */
template <typename LayoutTy, bool CONCURRENT, typename WorkListTy>
class FastSweepingMethod : public LayoutTy {};

template <bool CONCURRENT, typename WorkListTy>
class FastSweepingMethod<DenseSolver, CONCURRENT, WorkListTy>
    : public DenseSolver {
  void exec();

public:
  template <typename... Args>
  FastSweepingMethod(Args&&... args)
      : DenseSolver(std::forward<Args>(args)...) {}

  void runAlgo() { exec(); }
};

/**
 * Dense parallel execution of FSM
 */
template <>
void FastSweepingMethod<DenseSolver, true, ReqListTy>::exec() {
#if (WORK_ITEM_STAT)
  // Topology driven - no "empty" work
  galois::GAccumulator<std::size_t> badWork;
  badWork.reset();
#endif

  galois::GAccumulator<std::size_t> moreWork;

  //! Run algo
  std::size_t round = 0;
  do {
    round++;
    moreWork.reset();
#ifdef GALOIS_ENABLE_VTUNE
    galois::runtime::profileVtune(
        [&]() {
#endif
          // TODO: level should be signed but have to compare to unsigned
          // level=i+j from 0 to (nx-1)+(ny-1)
          for (std::size_t level = 0; level < (nx + ny - 1); level++) {
            galois::do_all(
                galois::iterate(
                    (level > nx - 1 ? level - nx + 1 : std::size_t(0u)),
                    std::min(std::size_t(ny), level + 1)),
                [&](const std::size_t& i) {
                  std::size_t const j = level - i;
                  galois::gDebug(level, " ", i, " ", j);

                  auto cur_u         = u(i, j).load(std::memory_order_relaxed);
                  data_t const new_u = solve(i, j);
                  if (std::isnan(new_u))
                    GALOIS_DIE("Differencing scheme returned NaN. This may "
                               "result from "
                               "insufficient precision or from bad input.");
                  do {
                    if (new_u >= cur_u)
                      return;
                  } while (!u(i, j).compare_exchange_weak(
                      cur_u, new_u, std::memory_order_relaxed));

#if (WORK_ITEM_STAT)
                  if (cur_u != INF)
                    badWork += 1;
#endif

                  if (std::abs(cur_u - new_u) >= tolerance) { // not converged
                    moreWork += 1;
                  }
                },
                galois::steal(),
                // galois::no_stats(),
                galois::chunk_size<CHUNK_SIZE>(),
                galois::loopname("DenseFSM_I"));
          }
          // level=i-j from (ny-1) to (1-nx)
          for (int64_t level = ny - 1; level > -(int64_t)nx; level--) {
            galois::do_all(
                galois::iterate(
                    (level > 0 ? std::size_t(level) : std::size_t(0u)),
                    std::min(std::size_t(ny), std::size_t(level + nx))),
                [&](const std::size_t& i) {
                  std::size_t const j = i - level;
                  galois::gDebug(level, " ", i, " ", j);

                  auto cur_u         = u(i, j).load(std::memory_order_relaxed);
                  data_t const new_u = solve(i, j);
                  if (std::isnan(new_u))
                    GALOIS_DIE("Differencing scheme returned NaN. This may "
                               "result from "
                               "insufficient precision or from bad input.");
                  do {
                    if (new_u >= cur_u)
                      return;
                  } while (!u(i, j).compare_exchange_weak(
                      cur_u, new_u, std::memory_order_relaxed));

#if (WORK_ITEM_STAT)
                  if (cur_u != INF)
                    badWork += 1;
#endif

                  if (std::abs(cur_u - new_u) >= tolerance) { // not converged
                    moreWork += 1;
                  }
                },
                galois::steal(),
                // galois::no_stats(),
                galois::chunk_size<CHUNK_SIZE>(),
                galois::loopname("DenseFSM_II"));
          }
          // level=i+j from (nx-1)+(ny-1) to 0
          for (std::size_t level = (nx + ny - 2); level + 1 != 0; level--) {
            galois::do_all(
                galois::iterate(
                    (level > nx - 1 ? level - nx + 1 : std::size_t(0u)),
                    std::min(std::size_t(ny), level + 1)),
                [&](const std::size_t& i) {
                  std::size_t const j = level - i;
                  galois::gDebug(level, " ", i, " ", j);

                  auto cur_u         = u(i, j).load(std::memory_order_relaxed);
                  data_t const new_u = solve(i, j);
                  if (std::isnan(new_u))
                    GALOIS_DIE("Differencing scheme returned NaN. This may "
                               "result from "
                               "insufficient precision or from bad input.");
                  do {
                    if (new_u >= cur_u)
                      return;
                  } while (!u(i, j).compare_exchange_weak(
                      cur_u, new_u, std::memory_order_relaxed));

#if (WORK_ITEM_STAT)
                  if (cur_u != INF)
                    badWork += 1;
#endif

                  if (std::abs(cur_u - new_u) >= tolerance) { // not converged
                    moreWork += 1;
                  }
                },
                galois::steal(),
                // galois::no_stats(),
                galois::chunk_size<CHUNK_SIZE>(),
                galois::loopname("DenseFSM_III"));
          }
          // level=i-j from (1-nx) to (ny-1)
          for (int64_t level = 1 - nx; level < (int64_t)ny; level++) {
            galois::do_all(
                galois::iterate(
                    (level > 0 ? std::size_t(level) : std::size_t(0u)),
                    std::min(std::size_t(ny), std::size_t(level + nx))),
                [&](const std::size_t& i) {
                  std::size_t const j = i - level;
                  galois::gDebug(level, " ", i, " ", j);

                  auto cur_u         = u(i, j).load(std::memory_order_relaxed);
                  data_t const new_u = solve(i, j);
                  if (std::isnan(new_u))
                    GALOIS_DIE("Differencing scheme returned NaN. This may "
                               "result from "
                               "insufficient precision or from bad input.");
                  do {
                    if (new_u >= cur_u)
                      return;
                  } while (!u(i, j).compare_exchange_weak(
                      cur_u, new_u, std::memory_order_relaxed));

#if (WORK_ITEM_STAT)
                  if (cur_u != INF)
                    badWork += 1;
#endif

                  if (std::abs(cur_u - new_u) >= tolerance) { // not converged
                    moreWork += 1;
                  }
                },
                galois::steal(),
                // galois::no_stats(),
                galois::chunk_size<CHUNK_SIZE>(),
                galois::loopname("DenseFSM_IV"));
          }
#ifdef GALOIS_ENABLE_VTUNE
        },
        "FSM_VTune");
#endif
  } while (moreWork.reduce());

  galois::runtime::reportStat_Single("DenseFSM", "Rounds", round);

#if (WORK_ITEM_STAT)
  galois::runtime::reportStat_Single("DenseFSM", "BadWork", badWork.reduce());
#endif
}

template <bool CONCURRENT, typename WorkListTy>
class FastSweepingMethod<CSRSolver, CONCURRENT, WorkListTy> : public CSRSolver {
  void exec();

public:
  template <typename... Args>
  FastSweepingMethod(Args&&... args) : CSRSolver(std::forward<Args>(args)...) {}

  void runAlgo() { exec(); }
};

template <>
void FastSweepingMethod<CSRSolver, true, ReqListTy>::exec() {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  galois::GAccumulator<std::size_t> didWork;

  auto PushOp = [&](const auto& node) {
    // using ItemTy =
    // std::remove_cv_t<std::remove_reference_t<decltype(item)>>; typename
    // ItemTy::second_type node; std::tie(std::ignore, node) = item; auto
    // [old_val, node] = item; galois::gDebug(old_val, " ", node);
    if (node >= NUM_CELLS)
      return;
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData = graph.getData(node, UNPROTECTED);
    // const data_t cur_val =
    // curData.solution.load(std::memory_order_relaxed);

    // Ignore stale repetitive work items
    /*
    if (cur_val < old_val) {
      return;
    }*/

    auto [new_val, adj_data] =
        quickUpdate(curData, graph.edge_begin(node, UNPROTECTED),
                    graph.edge_end(node, UNPROTECTED));

    if (galois::atomicMin(curData.solution, new_val) > new_val) {
      galois::gDebug("not converged:", new_val, " ", node);
      didWork += 1;
    }
  };

  //! Run algo
  do {
    didWork.reset();
#ifdef GALOIS_ENABLE_VTUNE
    galois::runtime::profileVtune(
        [&]() {
#endif
          galois::do_all(galois::iterate(graph.begin(), graph.end()), PushOp,
                         galois::steal(),
                         // galois::no_stats(),
                         galois::chunk_size<CHUNK_SIZE>(),
                         galois::loopname("FMM"));
#ifdef GALOIS_ENABLE_VTUNE
        },
        "FMM_VTune");
#endif
  } while (didWork.reduce());

#if (WORK_ITEM_STAT)
  galois::runtime::reportParam("Statistics", "EmptyWork", emptyWork.reduce());
  galois::runtime::reportParam("Statistics", "BadWork", badWork.reduce());
#endif
}

template <>
void FastSweepingMethod<CSRSolver, false, ReqListTy>::exec() {
#if (WORK_ITEM_STAT)
  galois::GAccumulator<std::size_t> emptyWork;
  galois::GAccumulator<std::size_t> badWork;
  emptyWork.reset();
  badWork.reset();
#endif

  galois::GAccumulator<std::size_t> didWork;

  auto PushOp = [&](const auto& node) {
    // using ItemTy =
    // std::remove_cv_t<std::remove_reference_t<decltype(item)>>; typename
    // ItemTy::second_type node; std::tie(std::ignore, node) = item; auto
    // [old_val, node] = item; galois::gDebug(old_val, " ", node);
    if (node >= NUM_CELLS)
      return;
    assert(node < NUM_CELLS && "Ghost Point!");
    auto& curData = graph.getData(node, UNPROTECTED);
    // const data_t cur_val =
    // curData.solution.load(std::memory_order_relaxed);

    // Ignore stale repetitive work items
    /*
    if (cur_val < old_val) {
      return;
    }*/

    auto [new_val, adj_data] =
        quickUpdate(curData, graph.edge_begin(node, UNPROTECTED),
                    graph.edge_end(node, UNPROTECTED));
    if (galois::atomicMin(curData.solution, new_val) > new_val) {
      galois::gDebug("not converged:", new_val, " ", node);
      didWork += 1;
    }
  };

  for (std::size_t i = 0; i < nx; i++) {
    galois::gDebug("1: Processing (", i, ") / (", nx, ", ", ny, ")");
    for (std::size_t j = 0; j < ny; j++)
      PushOp(ij2id({i, j}));
  }
  for (std::size_t i = nx; i > 0;) {
    galois::gDebug("2: Processing (", i, ") / (", nx, ", ", ny, ")");
    for (std::size_t j = 0; j < ny; j++)
      PushOp(ij2id({--i, j}));
  }
  for (std::size_t i = nx; i > 0;) {
    galois::gDebug("3: Processing (", i, ") / (", nx, ", ", ny, ")");
    for (std::size_t j = ny; j > 0;)
      PushOp(ij2id({--i, --j}));
  }
  for (std::size_t i = 0; i < nx; i++) {
    galois::gDebug("4: Processing (", i, ") / (", nx, ", ", ny, ")");
    for (std::size_t j = ny; j > 0;)
      PushOp(ij2id({i, --j}));
  }
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

#if (WORK_ITEM_STAT)
  galois::runtime::reportParam("Statistics", "EmptyWork", emptyWork.reduce());
  galois::runtime::reportParam("Statistics", "BadWork", badWork.reduce());
#endif
}

template <template <typename, bool, typename> class Solver, typename LayoutTy,
          bool CONCURRENT, typename Worklist, typename... Args>
void run(Args&&... args) {
  Solver<LayoutTy, CONCURRENT, Worklist> solver(std::forward<Args>(args)...);
  //! Boundary assignment and initialization
  // TODO better way for boundary settings?
  solver.determineBoundary();
  solver.initBoundary();

  //! Go!
  galois::StatTimer Tmain;
  Tmain.start();

  solver.runAlgo();

  Tmain.stop();

  solver.sanityCheck();
  solver.exportResults();
}

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  galois::gDebug(ALGO_NAMES[algo]);

  switch (algo) {
  case fmm:
    if (useDense) {
      if (useSerial)
        GALOIS_DIE("Serial Dense FMM not implemented yet.");
      else
        run<FastMarchingMethod, DenseSolver, true, ReqListTy>();
    } else {
      if (useSerial)
        run<FastMarchingMethod, CSRSolver, false, HeapTy>();
      else
        run<FastMarchingMethod, CSRSolver, true, ReqListTy>();
    }
    break;

  case fim:
    if (useDense) {
      if (useSerial)
        GALOIS_DIE("Serial Dense FIM not implemented yet.");
      else
        run<FastIterativeMethod, DenseSolver, true, ReqListTy>();
    } else {
      if (useSerial)
        GALOIS_DIE("Serial Sparse FIM not implemented yet.");
      else
        run<FastIterativeMethod, CSRSolver, true, ReqListTy>();
    }
    break;

  case fsm:
    if (useDense) {
      if (useSerial)
        GALOIS_DIE("Serial Dense FSM not implemented yet.");
      else
        run<FastSweepingMethod, DenseSolver, true, ReqListTy>();
    } else {
      if (useSerial)
        GALOIS_DIE("Serial Sparse FSM not implemented yet.");
      else
        run<FastSweepingMethod, CSRSolver, true, ReqListTy>();
    }
    break;

  default:
    std::abort();
  }

  return 0;
}
