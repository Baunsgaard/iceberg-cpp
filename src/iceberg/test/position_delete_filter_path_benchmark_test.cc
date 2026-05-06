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

// Microbenchmark for the filter-path inside DeleteLoader::LoadPositionDelete:
// when a position delete file does NOT have the `referenced_data_file`
// writer hint, we must check each row's `file_path` column to decide which
// positions belong to the data file we're loading deletes for.
//
// Two approaches are compared on identical inputs:
//
//   PerPosition    -- if row matches, call `index.Delete(pos)` directly.
//                     No staging buffer. Mirrors the pre-PR loader logic.
//   BufferedForEach -- if row matches, push_back into `std::vector`,
//                     then call `ForEachPositionDelete(buffer, index)`.
//                     The current loader logic.
//
// Plus a fast-path baseline (PureForEach over a span of matched-only
// positions) for reference -- this is what the hint-set path runs.
//
// Scenarios vary BOTH match selectivity (% of rows belonging to our data
// file) AND the layout of those matches (contiguous blocks vs scattered),
// to span the realistic shapes of multi-target delete files. Compaction-
// produced files sort by (file_path, pos) -> contiguous blocks. Pathological
// interleaving is included as a stress case.
//
// Skipped by default. To run:
//   ICEBERG_RUN_BENCHMARKS=1 ctest -R position_delete_filter_path_benchmark -V

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
#include <vector>

#include <gtest/gtest.h>

#include "iceberg/deletes/position_delete_index.h"

namespace iceberg {
namespace {

bool BenchmarksEnabled() {
  const char* env = std::getenv("ICEBERG_RUN_BENCHMARKS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

constexpr int64_t kRowCount = 5'000'000;
constexpr int kWarmupIterations = 2;
constexpr int kMeasureIterations = 4;

struct Scenario {
  std::string name;
  // One slot per row. `positions[i]` is the value that would be inserted if
  // `matches[i]` is true. Keeping both arrays the same length mirrors the
  // shape of the Arrow batch the loader iterates over.
  std::vector<int64_t> positions;
  std::vector<uint8_t> matches;  // 1 = belongs to our data file
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

// `total_rows` rows, of which the first `match_count` form one contiguous
// block of matches starting at row 0; the rest are non-matching rows.
// Mirrors a multi-target delete file sorted by (file_path, pos) where our
// data file sorts first. `positions[i]` is i across the whole array, so the
// matched positions form a contiguous run [0, match_count).
Scenario BuildContiguousBlockScenario(std::string name, int64_t total_rows,
                                      int64_t match_count) {
  Scenario s{std::move(name), {}, {}};
  s.positions.reserve(total_rows);
  s.matches.reserve(total_rows);
  for (int64_t i = 0; i < total_rows; ++i) {
    s.positions.push_back(i);
    s.matches.push_back(i < match_count ? 1 : 0);
  }
  return s;
}

// Matches scattered uniformly: every Nth row matches, where N is chosen so
// that match_ratio of rows match. Positions are sequential. The matched
// SUBSET ends up as 0, N, 2N, ... -- still sorted but with fixed stride.
// This is the worst case for run-coalescing on the matched output.
Scenario BuildScatteredScenario(std::string name, int64_t total_rows,
                                double match_ratio, uint32_t seed) {
  Scenario s{std::move(name), {}, {}};
  s.positions.reserve(total_rows);
  s.matches.reserve(total_rows);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int64_t i = 0; i < total_rows; ++i) {
    s.positions.push_back(i);
    s.matches.push_back(dist(rng) < match_ratio ? 1 : 0);
  }
  return s;
}

std::vector<Scenario> BuildScenarios() {
  std::vector<Scenario> scenarios;
  // Contiguous-block scenarios: realistic multi-target file shape.
  scenarios.push_back(BuildContiguousBlockScenario(
      "100pct_match_contiguous", kRowCount, kRowCount));
  scenarios.push_back(
      BuildContiguousBlockScenario("50pct_match_block", kRowCount, kRowCount / 2));
  scenarios.push_back(
      BuildContiguousBlockScenario("25pct_match_block", kRowCount, kRowCount / 4));
  scenarios.push_back(
      BuildContiguousBlockScenario("10pct_match_block", kRowCount, kRowCount / 10));
  scenarios.push_back(
      BuildContiguousBlockScenario("1pct_match_block", kRowCount, kRowCount / 100));
  // Scattered: stress case, no run coalescing possible on the matched subset.
  scenarios.push_back(
      BuildScatteredScenario("50pct_match_scattered", kRowCount, 0.50, /*seed=*/1));
  scenarios.push_back(
      BuildScatteredScenario("10pct_match_scattered", kRowCount, 0.10, /*seed=*/2));
  scenarios.push_back(
      BuildScatteredScenario("1pct_match_scattered", kRowCount, 0.01, /*seed=*/3));
  return scenarios;
}

int64_t CountMatches(const std::vector<uint8_t>& matches) {
  return std::count(matches.begin(), matches.end(), uint8_t{1});
}

// Approach A: pre-PR loader logic. Inline `index.Delete(pos)` for each
// matched row; no staging buffer.
double MeasurePerPosition(const Scenario& s, int iterations) {
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    PositionDeleteIndex index;
    const int64_t n = static_cast<int64_t>(s.positions.size());
    for (int64_t k = 0; k < n; ++k) {
      if (s.matches[k]) {
        index.Delete(s.positions[k]);
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    EXPECT_FALSE(index.IsEmpty());
  }
  return static_cast<double>(total_ns) / iterations;
}

// Approach B: current loader logic. Filter into a staging vector, then
// ForEachPositionDelete the whole buffer.
double MeasureBufferedForEach(const Scenario& s, int iterations) {
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    PositionDeleteIndex index;
    std::vector<int64_t> buf;
    buf.reserve(s.positions.size());
    const int64_t n = static_cast<int64_t>(s.positions.size());
    for (int64_t k = 0; k < n; ++k) {
      if (s.matches[k]) {
        buf.push_back(s.positions[k]);
      }
    }
    ForEachPositionDelete(buf, index);
    auto t1 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    EXPECT_FALSE(index.IsEmpty());
  }
  return static_cast<double>(total_ns) / iterations;
}

// Reference: fast-path call when the writer hint matches. The matched
// subset is pre-extracted (no filter cost) and handed straight to
// ForEachPositionDelete. Lower bound on what BufferedForEach can achieve.
double MeasurePureForEach(const Scenario& s, int iterations) {
  std::vector<int64_t> matched_only;
  matched_only.reserve(s.positions.size());
  for (size_t k = 0; k < s.positions.size(); ++k) {
    if (s.matches[k]) matched_only.push_back(s.positions[k]);
  }
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    PositionDeleteIndex index;
    ForEachPositionDelete(matched_only, index);
    auto t1 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    EXPECT_FALSE(index.IsEmpty());
  }
  return static_cast<double>(total_ns) / iterations;
}

TEST(PositionDeleteFilterPathBenchmark, CompareApproaches) {
  if (!BenchmarksEnabled()) {
    GTEST_SKIP() << "set ICEBERG_RUN_BENCHMARKS=1 to run this benchmark";
  }

  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << "  Filter-path benchmark: per-position vs buffered+ForEach\n";
  std::cout << "  Total rows per scenario: " << kRowCount << "\n";
  std::cout << "  Iterations: warmup=" << kWarmupIterations
            << ", measure=" << kMeasureIterations << "\n";
  std::cout << "================================================================\n";

  for (const auto& s : BuildScenarios()) {
    const int64_t matched = CountMatches(s.matches);

    // Warmup
    MeasurePerPosition(s, kWarmupIterations);
    MeasureBufferedForEach(s, kWarmupIterations);
    MeasurePureForEach(s, kWarmupIterations);

    double per_pos_ns = MeasurePerPosition(s, kMeasureIterations);
    double buffered_ns = MeasureBufferedForEach(s, kMeasureIterations);
    double pure_ns = MeasurePureForEach(s, kMeasureIterations);

    double speedup = buffered_ns > 0 ? per_pos_ns / buffered_ns : 0.0;
    double filter_overhead = pure_ns > 0 ? buffered_ns / pure_ns : 0.0;

    std::cout << "\n--- " << s.name << " (" << matched << " of " << s.positions.size()
              << " matched) ---\n";
    std::cout << "  PerPosition (proposed):    " << FormatTime(per_pos_ns) << "\n";
    std::cout << "  BufferedForEach (current): " << FormatTime(buffered_ns) << "\n";
    std::cout << "  PureForEach (fast-path):   " << FormatTime(pure_ns) << "\n";
    std::cout << "  Buffered vs PerPosition:   " << std::fixed << std::setprecision(2)
              << speedup << "x faster\n";
    std::cout << "  Filter overhead vs fast:   " << std::fixed << std::setprecision(2)
              << filter_overhead << "x slower than no-filter\n";
    std::cout << std::flush;
  }

  std::cout << "\n================================================================\n"
            << std::flush;
}

}  // namespace
}  // namespace iceberg
