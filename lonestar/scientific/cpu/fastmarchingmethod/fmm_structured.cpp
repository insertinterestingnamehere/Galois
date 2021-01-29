#include <array>
#include <functional>
#include <limits>

#include "numpy/cnpy.h"

#include <llvm/Support/CommandLine.h>

#include <galois/Bag.h>
#include <galois/LargeArray.h>
#include <galois/runtime/Executor_ForEach.h>

#include "Lonestar/BoilerPlate.h"

static char const* name = "Fast Marching Method";
static char const* desc =
    "Eikonal equation solver "
    "(https://en.wikipedia.org/wiki/Fast_marching_method)";
static char const* url = "";

static llvm::cl::opt<std::string> input_npy(
    "inpy", llvm::cl::value_desc("path-to-file"),
    llvm::cl::desc(
        "Use npy file (dtype=float32) as input speed map. NOTE: This will "
        "determine the size on each dimensions"),
    llvm::cl::init(""));

// TODO: support different dx and dy.
static llvm::cl::opt<float> h_input{
    "h", llvm::cl::desc("Distance between discrete samples in the input grid."),
    llvm::cl::init(1.f)};

#define CONSERVATIVE_WORK_ITEMS false
#define STRICT_MONOTONICITY false

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

// Handle comma separated values in inputs
template <typename NumTy, int MAX_SIZE = 0>
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
    if (MAX_SIZE && Val.size() > MAX_SIZE)
      return O.error(ArgName + "=" + ArgValue + ": expect no more than " +
                     std::to_string(MAX_SIZE) + " numbers but get " +
                     std::to_string(Val.size()));
    return false;
  }
};

static llvm::cl::opt<std::vector<std::size_t>, false,
                     NumVecParser<std::size_t, 2>>
    source_coordinates("d", llvm::cl::value_desc("i,j"),
                       llvm::cl::desc("Indices of the source point."));

static llvm::cl::opt<float> rounding_scale{
    "rounding_scale", llvm::cl::desc("Scale to use for roundoff."),
    llvm::cl::init(1.f)};

static std::size_t ny, nx;

auto ravel_index = [](std::size_t i, std::size_t j) noexcept -> std::size_t {
  return i * nx + j;
};

// TODO: use fastmod for this since it's a repeated operation.
auto unravel_index = [](std::size_t id) noexcept -> std::array<std::size_t, 2> {
  return {id / nx, id % nx};
};

struct work_t {
  double priority;
  std::size_t id;
};

int main(int argc, char** argv) noexcept {
  galois::SharedMemSys galois_system;
  LonestarStart(argc, argv, name, desc, url, nullptr);

  /*
   * setupGrids
   */
  cnpy::NpyArray npy = cnpy::npy_load(input_npy);
  if (npy.word_size != sizeof(float))
    GALOIS_DIE("Wrong data type.");
  if (npy.shape.size() != 2)
    GALOIS_DIE("Data should be 2-D.");
  std::shared_ptr<float[]> speed =
      std::reinterpret_pointer_cast<float[]>(npy.data_holder);
  GALOIS_ASSERT(speed, "Failed to load speed data.");
  ny = npy.shape[0], nx = npy.shape[1];
  // Use a lambda for indexing into the speed function.
  auto f = [&](size_t i, size_t j) noexcept -> float {
    assert(i < ny && j < nx && "Out of bounds access.");
    return std::cref(speed[i * nx + j]);
  };

  galois::LargeArray<std::atomic<double>> u_buffer;
  u_buffer.create(ny * nx);
  std::fill(u_buffer.begin(), u_buffer.end(),
            std::numeric_limits<double>::infinity());
  // Also use a lambda for indexing into the solution.
  auto u = [&](size_t i, size_t j) noexcept -> std::atomic<double>& {
    assert(i < ny && j < nx && "Out of bounds access.");
    std::atomic<double>& ret = u_buffer[i * nx + j];
    return ret;
  };

  /*
   * assignBoundary
   */
  std::size_t source_i, source_j;
  if (source_coordinates.empty()) {
    source_i = source_j = 0u;
  } else {
    source_i = source_coordinates[0];
    source_j = source_coordinates[1];
  }

  galois::InsertBag<work_t> initial;
  if (source_i)
    initial.push(work_t({0.f, ravel_index(source_i - 1, source_j)}));
  if (source_i < ny - 1)
    initial.push(work_t({0.f, ravel_index(source_i + 1, source_j)}));
  if (source_j)
    initial.push(work_t({0.f, ravel_index(source_i, source_j - 1)}));
  if (source_j < nx - 1)
    initial.push(work_t({0.f, ravel_index(source_i, source_j + 1)}));

  /*
   * initBoundary
   */
  u(source_i, source_j).store(0.f, std::memory_order_relaxed);

  /*
   * runAlgo
   */
  float scale  = rounding_scale;
  auto indexer = [&](work_t item) noexcept -> std::size_t {
    return std::round(scale * item.priority);
  };
  using PSchunk = galois::worklists::PerSocketChunkFIFO<128>;
  using OBIM =
      galois::worklists::OrderedByIntegerMetric<decltype(indexer), PSchunk>;

  auto gather_neighbors = [&](std::size_t i,
                              std::size_t j) noexcept -> std::array<double, 4> {
    double uN = std::numeric_limits<double>::infinity(), uS = uN, uE = uS,
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

  float h                = h_input;
  auto difference_scheme = [&](double f, double h, double uH,
                               double uV) noexcept -> double {
    double div = h / f, dif = uH - uV;
    if (!std::isinf(uH) && !std::isinf(uV) && std::abs(dif) < div) {
      // This variant of the differencing scheme is set up to avoid catastrophic
      // cancellation. The resulting precision loss can cause violations of the
      // nonincreasing property of the operator.
      double current  = .5 * (uH + uV + std::sqrt(2 * div * div - dif * dif));
      double previous = std::numeric_limits<double>::infinity();
      if constexpr (STRICT_MONOTONICITY) {
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

  galois::for_each(
      galois::iterate(initial),
      [&](work_t work_item, auto& context) noexcept {
        auto const [i, j] = unravel_index(work_item.id);
        double uN, uS, uE, uW, previous, new_u;
        if constexpr (!CONSERVATIVE_WORK_ITEMS) {
          previous = u(i, j).load(std::memory_order_relaxed);
          if (previous < work_item.priority)
            return;
          auto const neighbor_vals = gather_neighbors(i, j);
          uN                       = neighbor_vals[0];
          uS                       = neighbor_vals[1];
          uE                       = neighbor_vals[2];
          uW                       = neighbor_vals[3];
          auto const uH = std::min(uE, uW), uV = std::min(uN, uS);
          new_u = difference_scheme(f(i, j), h, uH, uV);
          if (std::isnan(new_u))
            GALOIS_DIE("Differencing scheme returned NaN. This may result from "
                       "insufficient precision or from bad input.");
          do {
            if (new_u >= previous)
              return;
          } while (!u(i, j).compare_exchange_weak(previous, new_u,
                                                  std::memory_order_relaxed));
        } else {
          do {
            previous        = u(i, j).load(std::memory_order_relaxed);
            auto const vals = gather_neighbors(i, j);
            uN              = vals[0];
            uS              = vals[1];
            uE              = vals[2];
            uW              = vals[3];
            double const uH = std::min(uE, uW);
            double const uV = std::min(uN, uS);
            new_u           = difference_scheme(f(i, j), h, uH, uV);
            if (std::isnan(new_u))
              GALOIS_DIE("Differencing scheme returned NaN.");
            if (new_u == previous)
              return;
          } while (!u(i, j).compare_exchange_weak(previous, new_u,
                                                  std::memory_order_relaxed));
        }
        if (i && uS > new_u)
          context.push(work_t({new_u, ravel_index(i - 1, j)}));
        if (i < ny - 1 && uN > new_u)
          context.push(work_t({new_u, ravel_index(i + 1, j)}));
        if (j && uW > new_u)
          context.push(work_t({new_u, ravel_index(i, j - 1)}));
        if (j < nx - 1 && uE > new_u)
          context.push(work_t({new_u, ravel_index(i, j + 1)}));
      },
      galois::loopname("fmm"), galois::disable_conflict_detection(),
      galois::wl<OBIM>(indexer));

  /*
   * sanityCheck
   */
  // Tolerance for non-strict differencing operator.
  double constexpr tolerance = CONSERVATIVE_WORK_ITEMS ? 0. : 1E-14;
  // double constexpr tolerance = 0.;

  galois::do_all(
      galois::iterate(std::size_t(0u), nx * ny),
      [&](std::size_t id) noexcept {
        auto const [i, j] = unravel_index(id);
        if (i == source_i && j == source_j)
          return;
        auto const [uN, uS, uE, uW] = gather_neighbors(i, j);
        auto const uH = std::min(uE, uW), uV = std::min(uN, uS);
        double const new_u = difference_scheme(f(i, j), h, uH, uV);
        double const old_u = u(i, j).load(std::memory_order_relaxed);
        if (new_u != old_u) {
          if (STRICT_MONOTONICITY ||
              (std::abs(new_u - old_u) > tolerance * std::max(new_u, old_u))) {
            std::cout << new_u << " " << old_u << " " << new_u - old_u
                      << std::endl;
            GALOIS_DIE("Failed correctness check");
          }
        }
      },
      galois::loopname("Check correctness"));
}
