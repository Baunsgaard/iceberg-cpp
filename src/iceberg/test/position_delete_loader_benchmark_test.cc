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

// End-to-end benchmark for `DeleteLoader::LoadPositionDeletes`. Measures
// the full path: Parquet decode + Arrow row iteration + bitmap inserts.
//
// The baseline replicates the pre-PR loader logic (per-position
// `index.Delete(pos)` inside the Arrow batch loop); the "after" path
// calls `LoadPositionDeletes`, which buffers per-batch and dispatches
// through `ForEachPositionDelete`.
//
// Skipped by default. To run:
//   ICEBERG_RUN_BENCHMARKS=1 ctest -R position_delete_loader_benchmark -V

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "iceberg/arrow/arrow_fs_file_io_internal.h"
#include "iceberg/arrow_c_data_guard_internal.h"
#include "iceberg/data/delete_loader.h"
#include "iceberg/data/position_delete_writer.h"
#include "iceberg/deletes/position_delete_index.h"
#include "iceberg/file_format.h"
#include "iceberg/file_reader.h"
#include "iceberg/manifest/manifest_entry.h"
#include "iceberg/metadata_columns.h"
#include "iceberg/parquet/parquet_register.h"
#include "iceberg/partition_spec.h"
#include "iceberg/row/arrow_array_wrapper.h"
#include "iceberg/row/partition_values.h"
#include "iceberg/schema.h"
#include "iceberg/schema_field.h"
#include "iceberg/type.h"
#include "iceberg/util/macros.h"

namespace iceberg {
namespace {

bool BenchmarksEnabled() {
  const char* env = std::getenv("ICEBERG_RUN_BENCHMARKS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

constexpr int64_t kPositionCount = 500'000;
constexpr int kWarmupIterations = 5;
constexpr int kMeasureIterations = 15;

std::string FormatTime(double ns) {
  char buf[64];
  if (ns < 1e3) {
    std::snprintf(buf, sizeof(buf), "%.1f ns", ns);
  } else if (ns < 1e6) {
    std::snprintf(buf, sizeof(buf), "%.1f us", ns / 1e3);
  } else {
    std::snprintf(buf, sizeof(buf), "%.2f ms", ns / 1e6);
  }
  return buf;
}

struct Scenario {
  std::string name;
  std::vector<int64_t> positions;
};

std::vector<int64_t> BuildContiguous(int64_t count) {
  std::vector<int64_t> positions;
  positions.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    positions.push_back(i);
  }
  return positions;
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

std::vector<int64_t> SampleUniqueSorted(int64_t count, int64_t domain, uint32_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, domain - 1);
  std::vector<int64_t> positions;
  positions.reserve(count);

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
  scenarios.push_back({"contiguous_500K", BuildContiguous(kPositionCount)});
  scenarios.push_back({"runs_64_gap_16_500K",
                       BuildRuns(kPositionCount, /*run_length=*/64, /*gap=*/16)});
  scenarios.push_back({"runs_4_gap_4_500K",
                       BuildRuns(kPositionCount, /*run_length=*/4, /*gap=*/4)});
  scenarios.push_back({"alternating_500K", BuildAlternating(kPositionCount)});
  scenarios.push_back(
      {"sparse_50pct_500K", BuildSparse(kPositionCount, 0.50, /*seed=*/2)});
  scenarios.push_back(
      {"sparse_5pct_500K", BuildSparse(kPositionCount, 0.05, /*seed=*/3)});
  return scenarios;
}

class DeleteLoaderBenchmark : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { parquet::RegisterAll(); }

  void SetUp() override {
    file_io_ = arrow::ArrowFileSystemFileIO::MakeMockFileIO();
    schema_ = std::make_shared<Schema>(
        std::vector<SchemaField>{SchemaField::MakeRequired(1, "id", int32())});
    partition_spec_ = PartitionSpec::Unpartitioned();
    loader_ = std::make_unique<DeleteLoader>(file_io_);
  }

  std::shared_ptr<DataFile> WritePositionDeletes(
      const std::string& path, const std::string& data_file_path,
      const std::vector<int64_t>& positions) {
    PositionDeleteWriterOptions options{
        .path = path,
        .schema = schema_,
        .spec = partition_spec_,
        .partition = PartitionValues{},
        .format = FileFormatType::kParquet,
        .io = file_io_,
        .flush_threshold = 100'000,
        .properties = {{"write.parquet.compression-codec", "uncompressed"}},
    };
    auto writer = PositionDeleteWriter::Make(options).value();
    for (int64_t pos : positions) {
      ICEBERG_THROW_NOT_OK(writer->WriteDelete(data_file_path, pos));
    }
    ICEBERG_THROW_NOT_OK(writer->Close());
    return writer->Metadata().value().data_files[0];
  }

  std::shared_ptr<FileIO> file_io_;
  std::shared_ptr<Schema> schema_;
  std::shared_ptr<PartitionSpec> partition_spec_;
  std::unique_ptr<DeleteLoader> loader_;
};

std::shared_ptr<Schema> PosDeleteSchema() {
  return std::make_shared<Schema>(std::vector<SchemaField>{
      MetadataColumns::kDeleteFilePath,
      MetadataColumns::kDeleteFilePos,
  });
}

// Replicates the pre-PR loader logic: open the delete file, iterate each
// batch row-by-row, and call `index.Delete(pos)` for each matching row.
Status LoadPositionDeleteBaseline(const std::shared_ptr<FileIO>& io,
                                  const DataFile& file, PositionDeleteIndex& index,
                                  std::string_view data_file_path) {
  ReaderOptions options{
      .path = file.file_path,
      .length = static_cast<size_t>(file.file_size_in_bytes),
      .io = io,
      .projection = PosDeleteSchema(),
  };
  ICEBERG_ASSIGN_OR_RAISE(auto reader,
                          ReaderFactoryRegistry::Open(file.file_format, options));

  ICEBERG_ASSIGN_OR_RAISE(auto arrow_schema, reader->Schema());
  internal::ArrowSchemaGuard schema_guard(&arrow_schema);

  while (true) {
    ICEBERG_ASSIGN_OR_RAISE(auto batch_opt, reader->Next());
    if (!batch_opt.has_value()) break;

    auto& batch = batch_opt.value();
    internal::ArrowArrayGuard batch_guard(&batch);

    ICEBERG_ASSIGN_OR_RAISE(
        auto row, ArrowArrayStructLike::Make(arrow_schema, batch, /*row_index=*/0));

    for (int64_t i = 0; i < batch.length; ++i) {
      if (i > 0) {
        ICEBERG_RETURN_UNEXPECTED(row->Reset(i));
      }
      ICEBERG_ASSIGN_OR_RAISE(auto path_scalar, row->GetField(0));
      auto path = std::get<std::string_view>(path_scalar);
      if (path == data_file_path) {
        ICEBERG_ASSIGN_OR_RAISE(auto pos_scalar, row->GetField(1));
        index.Delete(std::get<int64_t>(pos_scalar));
      }
    }
  }

  return reader->Close();
}

double MeasureBaseline(const std::shared_ptr<FileIO>& io, const DataFile& file,
                       std::string_view data_file_path, int iterations) {
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    PositionDeleteIndex index;
    auto status = LoadPositionDeleteBaseline(io, file, index, data_file_path);
    auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(status.has_value());
    EXPECT_FALSE(index.IsEmpty());
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }
  return static_cast<double>(total_ns) / iterations;
}

double MeasureLoader(DeleteLoader& loader,
                     const std::vector<std::shared_ptr<DataFile>>& files,
                     std::string_view data_file_path, int iterations) {
  long long total_ns = 0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    auto result = loader.LoadPositionDeletes(files, data_file_path);
    auto t1 = std::chrono::steady_clock::now();
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(result->IsEmpty());
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }
  return static_cast<double>(total_ns) / iterations;
}

TEST_F(DeleteLoaderBenchmark, MeasureBeforeAndAfter) {
  if (!BenchmarksEnabled()) {
    GTEST_SKIP() << "set ICEBERG_RUN_BENCHMARKS=1 to run this benchmark";
  }

  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  DeleteLoader::LoadPositionDeletes End-to-End Benchmark\n";
  std::cout << "  Before = per-position index.Delete() in batch loop\n";
  std::cout << "  After  = per-batch ForEachPositionDelete()\n";
  std::cout << "  Iterations: warmup=" << kWarmupIterations
            << ", measure=" << kMeasureIterations << "\n";
  std::cout << "============================================================\n";

  const std::string data_file_path = "data.parquet";
  int scenario_index = 0;
  for (const auto& s : BuildScenarios()) {
    const std::string path = "pos_deletes_bench_" + std::to_string(scenario_index++) +
                             ".parquet";
    auto delete_file = WritePositionDeletes(path, data_file_path, s.positions);
    std::vector<std::shared_ptr<DataFile>> files = {delete_file};

    MeasureBaseline(file_io_, *delete_file, data_file_path, kWarmupIterations);
    MeasureLoader(*loader_, files, data_file_path, kWarmupIterations);

    double before_ns =
        MeasureBaseline(file_io_, *delete_file, data_file_path, kMeasureIterations);
    double after_ns = MeasureLoader(*loader_, files, data_file_path, kMeasureIterations);
    double speedup = after_ns > 0 ? before_ns / after_ns : 0.0;

    std::cout << "\n--- " << s.name << " (" << s.positions.size() << " positions) ---\n";
    std::cout << "  Before (Delete loop):  " << FormatTime(before_ns) << "\n";
    std::cout << "  After  (LoadPositionDeletes): " << FormatTime(after_ns) << "\n";
    std::cout << "  Speedup:               " << std::fixed << std::setprecision(2)
              << speedup << "x\n";
    std::cout << std::flush;
  }

  std::cout << "\n============================================================\n"
            << std::flush;
}

}  // namespace
}  // namespace iceberg
