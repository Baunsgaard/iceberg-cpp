/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Microbenchmark for `ForEachPositionDelete`, modeled on the Java
// `TestRoaringPositionBitmapPerf` (`benchmarkSetRangeVsSetLoop`) style:
// per-call wall time, simple before/after/speedup output.
//
// Skipped by default. To run:
//   ICEBERG_RUN_BENCHMARKS=1 ctest -R position_delete_range_consumer_benchmark -V
//
// Each scenario times two paths over the same `positions` input:
//   Before = for (pos : positions) index.Delete(pos);
//   After  = ForEachPositionDelete(positions, index);
//
// The index is constructed inside the timed region (one fresh index per
// data file load in production). The reported number is the average
// wall time of a single call across measure iterations.

#include "iceberg/deletes/position_delete_range_consumer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "iceberg/deletes/position_delete_index.h"

namespace iceberg {
namespace {

bool BenchmarksEnabled() {
  const char* env = std::getenv("ICEBERG_RUN_BENCHMARKS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

constexpr int64_t kPositionCount = 5'000'000;
// Amortizes a single call (~80ns) for tiny-N scenarios so the wall time
// rises above `steady_clock` resolution; we still report per-call time.
constexpr int kTinyInnerIterations = 500'000;
constexpr int kWarmupIterations = 3;
constexpr int kMeasureIterations = 5;

struct Scenario {
  std::string name;
  std::vector<int64_t> positions;
  int inner_iterations = 1;
};

std::string FormatTime(double ns) {
  char buf[64];
  if (ns < 1000.0) {
    std::snprintf(buf, sizeof(buf), "%.1f ns", ns);
  } else if (ns < 1e6) {
    std::snprintf(buf, sizeof(buf), "%.1f us", ns / 1e3);
  } else {
    std::snprintf(buf, sizeof(buf), "%.2f ms", ns / 1e6);
  }
  return buf;
}

std::vector<int64_t> BuildAlternating(int64_t count) {
  std::vector<int64_t> positions;
  positions.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    positions.push_back(i * 2);
  }
  return positions;
}

std::vector<int64_t> BuildRuns(int64_t count, int64_t run_length, int64_t gap) {
  std::vector<int64_t> positions;
  positions.reserve(count);
  int64_t pos = 0;
  while (static_cast<int64_t>(positions.size()) < count) {
    int64_t run_end = std::min(pos + run_length,
                               pos + count - static_cast<int64_t>(positions.size()));
    for (int64_t p = pos; p < run_end; ++p) {
      positions.push_back(p);
    }
    pos = run_end + gap;
  }
  return positions;
}

std::vector<int64_t> BuildContiguous(int64_t count) {
  std::vector<int64_t> positions;
  positions.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    positions.push_back(i);
  }
  return positions;
}

// Returns exactly `count` distinct positions in [0, domain), sorted. Uses
// rejection sampling for sparse half; complement (sample excluded) for
// dense half to keep high-density cases cheap.
std::vector<int64_t> SampleUniqueSorted(int64_t count, int64_t domain, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<int64_t> positions;
  positions.reserve(count);
  std::uniform_int_distribution<int64_t> dist(0, domain - 1);

  if (count * 2 <= domain) {
    std::unordered_set<int64_t> chosen;
    chosen.reserve(static_cast<size_t>(count));
    while (static_cast<int64_t>(chosen.size()) < count) {
      chosen.insert(dist(rng));
    }
    positions.assign(chosen.begin(), chosen.end());
  } else {
    const int64_t exclude_count = domain - count;
    std::unordered_set<int64_t> excluded;
    excluded.reserve(static_cast<size_t>(exclude_count));
    while (static_cast<int64_t>(excluded.size()) < exclude_count) {
      excluded.insert(dist(rng));
    }
    for (int64_t i = 0; i < domain; ++i) {
      if (!excluded.contains(i)) {
        positions.push_back(i);
      }
    }
  }

  std::sort(positions.begin(), positions.end());
  return positions;
}

std::vector<int64_t> BuildSparse(int64_t count, double density, uint32_t seed) {
  const int64_t domain = static_cast<int64_t>(static_cast<double>(count) / density);
  return SampleUniqueSorted(count, domain, seed);
}

std::vector<Scenario> BuildScenarios() {
  std::vector<Scenario> scenarios;
  scenarios.push_back({"single_position", {42}, kTinyInnerIterations});
  scenarios.push_back({"tiny_8_scattered",
                       {3, 17, 42, 43, 44, 1'000, 5'000, 100'000},
                       kTinyInnerIterations});
  scenarios.push_back({"tiny_8_contiguous",
                       {0, 1, 2, 3, 4, 5, 6, 7},
                       kTinyInnerIterations});
  scenarios.push_back({"alternating_5M", BuildAlternating(kPositionCount)});
  scenarios.push_back({"short_runs_4_gap_4_5M",
                       BuildRuns(kPositionCount, /*run_length=*/4, /*gap=*/4)});
  scenarios.push_back({"medium_runs_64_gap_16_5M",
                       BuildRuns(kPositionCount, /*run_length=*/64, /*gap=*/16)});
  scenarios.push_back({"contiguous_5M", BuildContiguous(kPositionCount)});
  scenarios.push_back({"sparse_5pct_5M",
                       BuildSparse(kPositionCount, /*density=*/0.05, /*seed=*/1)});
  scenarios.push_back({"sparse_50pct_5M",
                       BuildSparse(kPositionCount, /*density=*/0.50, /*seed=*/2)});
  scenarios.push_back({"sparse_95pct_5M",
                       BuildSparse(kPositionCount, /*density=*/0.95, /*seed=*/3)});
  return scenarios;
}

double MeasureBaseline(const std::vector<int64_t>& positions, int iterations, int inner) {
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < inner; ++k) {
      PositionDeleteIndex index;
      for (int64_t pos : positions) {
        index.Delete(pos);
      }
      EXPECT_FALSE(index.IsEmpty());
    }
    auto t1 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }
  return static_cast<double>(total_ns) / iterations / inner;
}

double MeasureForEach(const std::vector<int64_t>& positions, int iterations, int inner) {
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < inner; ++k) {
      PositionDeleteIndex index;
      ForEachPositionDelete(std::span<const int64_t>(positions), index);
      EXPECT_FALSE(index.IsEmpty());
    }
    auto t1 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }
  return static_cast<double>(total_ns) / iterations / inner;
}

TEST(PositionDeleteRangeConsumerBenchmark, MeasureBeforeAndAfter) {
  if (!BenchmarksEnabled()) {
    GTEST_SKIP() << "set ICEBERG_RUN_BENCHMARKS=1 to run this benchmark";
  }

  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  C++ ForEachPositionDelete Benchmark\n";
  std::cout << "  Before = loop of Delete(pos) | After = ForEachPositionDelete()\n";
  std::cout << "  Iterations: warmup=" << kWarmupIterations
            << ", measure=" << kMeasureIterations << "\n";
  std::cout << "============================================================\n";

  for (const auto& s : BuildScenarios()) {
    ASSERT_FALSE(s.positions.empty()) << s.name << " has no positions";
    ASSERT_TRUE(std::is_sorted(s.positions.begin(), s.positions.end()))
        << s.name << " positions are not sorted";
    ASSERT_EQ(std::adjacent_find(s.positions.begin(), s.positions.end()),
              s.positions.end())
        << s.name << " positions contain duplicates";

    const int inner = s.inner_iterations;
    MeasureBaseline(s.positions, kWarmupIterations, inner);
    MeasureForEach(s.positions, kWarmupIterations, inner);

    double before_ns = MeasureBaseline(s.positions, kMeasureIterations, inner);
    double after_ns = MeasureForEach(s.positions, kMeasureIterations, inner);
    double speedup = after_ns > 0 ? before_ns / after_ns : 0.0;

    std::cout << "\n--- " << s.name << " (" << s.positions.size() << " positions) ---\n";
    std::cout << "  Before (Delete loop):  " << FormatTime(before_ns) << "\n";
    std::cout << "  After  (ForEach):      " << FormatTime(after_ns) << "\n";
    std::cout << "  Speedup:               " << std::fixed << std::setprecision(1)
              << speedup << "x\n";
    std::cout << std::flush;
  }

  std::cout << "\n============================================================\n"
            << std::flush;
}

}  // namespace
}  // namespace iceberg
