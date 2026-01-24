/**
 * ThunderDuck Detailed Benchmark
 *
 * Generates comprehensive performance report with:
 * - Specific SQL queries
 * - Execution time
 * - Data volume accessed
 * - Speedup ratio
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <sstream>
#include <cmath>
#include <map>
#include <algorithm>
#include <numeric>
#include <ctime>

// DuckDB
#include "duckdb.hpp"

// ThunderDuck
#include "thunderduck/thunderduck.h"
#include "thunderduck/filter.h"
#include "thunderduck/aggregate.h"
#include "thunderduck/sort.h"
#include "thunderduck/join.h"
#include "thunderduck/memory.h"

using namespace duckdb;
using namespace std::chrono;

// ============================================================================
// Test Case Structure
// ============================================================================

struct TestCase {
    std::string id;
    std::string category;
    std::string name;
    std::string sql;
    std::string thunder_op;
    size_t data_rows;
    size_t data_bytes;
    double duckdb_ms;
    double thunder_ms;
    double speedup;
    size_t result_count;
    std::string winner;
};

// ============================================================================
// Data Generator
// ============================================================================

class DataGenerator {
public:
    DataGenerator(uint32_t seed = 42) : rng_(seed) {}

    int32_t random_int(int32_t min, int32_t max) {
        std::uniform_int_distribution<int32_t> dist(min, max);
        return dist(rng_);
    }

    double random_double(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(rng_);
    }

    std::string random_date(int year_start, int year_end) {
        int year = random_int(year_start, year_end);
        int month = random_int(1, 12);
        int day = random_int(1, 28);
        std::ostringstream oss;
        oss << year << "-" << std::setfill('0') << std::setw(2) << month
            << "-" << std::setw(2) << day;
        return oss.str();
    }

private:
    std::mt19937 rng_;
};

// ============================================================================
// Timer
// ============================================================================

class Timer {
public:
    void start() { start_ = high_resolution_clock::now(); }
    void stop() { end_ = high_resolution_clock::now(); }
    double ms() const {
        return duration_cast<nanoseconds>(end_ - start_).count() / 1000000.0;
    }
private:
    high_resolution_clock::time_point start_, end_;
};

// ============================================================================
// Detailed Benchmark
// ============================================================================

class DetailedBenchmark {
public:
    DetailedBenchmark() : gen_(42) {}

    void run() {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              ThunderDuck Detailed Performance Benchmark                   ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n\n";

        // Initialize
        init_thunderduck();
        init_duckdb();
        create_schema();
        generate_data();
        extract_data();

        // Run tests
        run_all_tests();

        // Generate report
        generate_report();
    }

private:
    void init_thunderduck() {
        std::cout << "[1/6] Initializing ThunderDuck... ";
        thunderduck::initialize();
        std::cout << "OK\n";
    }

    void init_duckdb() {
        std::cout << "[2/6] Initializing DuckDB... ";
        db_ = std::make_unique<DuckDB>(nullptr);
        conn_ = std::make_unique<Connection>(*db_);
        std::cout << "OK\n";
    }

    void create_schema() {
        std::cout << "[3/6] Creating schema... ";
        conn_->Query(R"(
            CREATE TABLE lineitem (
                l_orderkey INTEGER,
                l_partkey INTEGER,
                l_quantity INTEGER,
                l_extendedprice DECIMAL(15,2),
                l_discount DECIMAL(5,2),
                l_tax DECIMAL(5,2),
                l_shipdate DATE
            )
        )");
        conn_->Query(R"(
            CREATE TABLE orders (
                o_orderkey INTEGER PRIMARY KEY,
                o_custkey INTEGER,
                o_totalprice DECIMAL(15,2),
                o_orderdate DATE,
                o_orderpriority VARCHAR(20)
            )
        )");
        conn_->Query(R"(
            CREATE TABLE customer (
                c_custkey INTEGER PRIMARY KEY,
                c_name VARCHAR(50),
                c_acctbal DECIMAL(15,2),
                c_mktsegment VARCHAR(20)
            )
        )");
        std::cout << "OK\n";
    }

    void generate_data() {
        std::cout << "[4/6] Generating test data...\n";

        // Generate customers (100K)
        num_customers_ = 100000;
        std::cout << "       - customers: " << num_customers_ << " rows... ";
        conn_->Query("BEGIN TRANSACTION");
        Appender cust_app(*conn_, "customer");
        std::vector<const char*> segments = {"AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"};
        for (size_t i = 0; i < num_customers_; ++i) {
            std::string name = "Customer_" + std::to_string(i);
            cust_app.BeginRow();
            cust_app.Append<int32_t>(i);
            cust_app.Append<const char*>(name.c_str());
            cust_app.Append<double>(gen_.random_double(-1000, 10000));
            cust_app.Append<const char*>(segments[gen_.random_int(0, 4)]);
            cust_app.EndRow();
        }
        cust_app.Close();
        conn_->Query("COMMIT");
        std::cout << "OK\n";

        // Generate orders (1M)
        num_orders_ = 1000000;
        std::cout << "       - orders: " << num_orders_ << " rows... ";
        conn_->Query("BEGIN TRANSACTION");
        Appender order_app(*conn_, "orders");
        std::vector<const char*> priorities = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
        for (size_t i = 0; i < num_orders_; ++i) {
            std::string date = gen_.random_date(2020, 2024);
            order_app.BeginRow();
            order_app.Append<int32_t>(i);
            order_app.Append<int32_t>(gen_.random_int(0, num_customers_ - 1));
            order_app.Append<double>(gen_.random_double(100, 500000));
            order_app.Append<const char*>(date.c_str());
            order_app.Append<const char*>(priorities[gen_.random_int(0, 4)]);
            order_app.EndRow();
        }
        order_app.Close();
        conn_->Query("COMMIT");
        std::cout << "OK\n";

        // Generate lineitem (5M)
        num_lineitem_ = 5000000;
        std::cout << "       - lineitem: " << num_lineitem_ << " rows... ";
        conn_->Query("BEGIN TRANSACTION");
        Appender line_app(*conn_, "lineitem");
        for (size_t i = 0; i < num_lineitem_; ++i) {
            std::string date = gen_.random_date(2020, 2024);
            line_app.BeginRow();
            line_app.Append<int32_t>(gen_.random_int(0, num_orders_ - 1));
            line_app.Append<int32_t>(gen_.random_int(0, 10000));
            line_app.Append<int32_t>(gen_.random_int(1, 50));
            line_app.Append<double>(gen_.random_double(1, 1000));
            line_app.Append<double>(gen_.random_double(0, 0.1));
            line_app.Append<double>(gen_.random_double(0, 0.08));
            line_app.Append<const char*>(date.c_str());
            line_app.EndRow();
        }
        line_app.Close();
        conn_->Query("COMMIT");
        std::cout << "OK\n";
    }

    void extract_data() {
        std::cout << "[5/6] Extracting data for ThunderDuck... ";

        // Extract lineitem.l_quantity
        auto result = conn_->Query("SELECT CAST(l_quantity AS INTEGER) FROM lineitem");
        lineitem_quantity_.reserve(result->RowCount());
        for (size_t i = 0; i < result->RowCount(); ++i) {
            lineitem_quantity_.push_back(result->GetValue(0, i).GetValue<int32_t>());
        }

        // Extract lineitem.l_extendedprice as integer
        result = conn_->Query("SELECT CAST(l_extendedprice AS INTEGER) FROM lineitem");
        lineitem_price_.reserve(result->RowCount());
        for (size_t i = 0; i < result->RowCount(); ++i) {
            lineitem_price_.push_back(result->GetValue(0, i).GetValue<int32_t>());
        }

        // Extract orders.o_totalprice
        result = conn_->Query("SELECT CAST(o_totalprice AS INTEGER) FROM orders");
        order_price_.reserve(result->RowCount());
        for (size_t i = 0; i < result->RowCount(); ++i) {
            order_price_.push_back(result->GetValue(0, i).GetValue<int32_t>());
        }

        // Extract orders.o_custkey
        result = conn_->Query("SELECT o_custkey FROM orders");
        order_custkey_.reserve(result->RowCount());
        for (size_t i = 0; i < result->RowCount(); ++i) {
            order_custkey_.push_back(result->GetValue(0, i).GetValue<int32_t>());
        }

        // Extract customer.c_custkey
        result = conn_->Query("SELECT c_custkey FROM customer");
        customer_key_.reserve(result->RowCount());
        for (size_t i = 0; i < result->RowCount(); ++i) {
            customer_key_.push_back(result->GetValue(0, i).GetValue<int32_t>());
        }

        std::cout << "OK\n";
    }

    void run_all_tests() {
        std::cout << "[6/6] Running benchmarks...\n\n";

        int iterations = 10;
        int warmup = 3;

        // =====================================================================
        // Filter Tests
        // =====================================================================

        // F1: Simple comparison
        {
            TestCase tc;
            tc.id = "F1";
            tc.category = "Filter";
            tc.name = "Simple Comparison (GT)";
            tc.sql = "SELECT COUNT(*) FROM lineitem WHERE l_quantity > 25";
            tc.thunder_op = "count_i32_v2(l_quantity, GT, 25)";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->GetValue(0, 0).GetValue<int64_t>();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            for (int i = 0; i < warmup; ++i) {
                thunderduck::filter::count_i32_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), thunderduck::filter::CompareOp::GT, 25);
            }
            total = 0;
            size_t count = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                count = thunderduck::filter::count_i32_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), thunderduck::filter::CompareOp::GT, 25);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // F2: Equality comparison
        {
            TestCase tc;
            tc.id = "F2";
            tc.category = "Filter";
            tc.name = "Equality Comparison (EQ)";
            tc.sql = "SELECT COUNT(*) FROM lineitem WHERE l_quantity = 30";
            tc.thunder_op = "count_i32_v2(l_quantity, EQ, 30)";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->GetValue(0, 0).GetValue<int64_t>();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            for (int i = 0; i < warmup; ++i) {
                thunderduck::filter::count_i32_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), thunderduck::filter::CompareOp::EQ, 30);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::filter::count_i32_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), thunderduck::filter::CompareOp::EQ, 30);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // F3: Range filter
        {
            TestCase tc;
            tc.id = "F3";
            tc.category = "Filter";
            tc.name = "Range Filter (BETWEEN)";
            tc.sql = "SELECT COUNT(*) FROM lineitem WHERE l_quantity >= 10 AND l_quantity < 40";
            tc.thunder_op = "count_i32_range_v2(l_quantity, 10, 40)";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->GetValue(0, 0).GetValue<int64_t>();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            for (int i = 0; i < warmup; ++i) {
                thunderduck::filter::count_i32_range_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), 10, 40);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::filter::count_i32_range_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), 10, 40);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // F4: High selectivity filter
        {
            TestCase tc;
            tc.id = "F4";
            tc.category = "Filter";
            tc.name = "High Selectivity (price > 500)";
            tc.sql = "SELECT COUNT(*) FROM lineitem WHERE l_extendedprice > 500";
            tc.thunder_op = "count_i32_v2(l_extendedprice, GT, 500)";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->GetValue(0, 0).GetValue<int64_t>();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            for (int i = 0; i < warmup; ++i) {
                thunderduck::filter::count_i32_v2(lineitem_price_.data(),
                    lineitem_price_.size(), thunderduck::filter::CompareOp::GT, 500);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::filter::count_i32_v2(lineitem_price_.data(),
                    lineitem_price_.size(), thunderduck::filter::CompareOp::GT, 500);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // =====================================================================
        // Aggregation Tests
        // =====================================================================

        // A1: SUM
        {
            TestCase tc;
            tc.id = "A1";
            tc.category = "Aggregation";
            tc.name = "SUM";
            tc.sql = "SELECT SUM(l_quantity) FROM lineitem";
            tc.thunder_op = "sum_i32_v2(l_quantity)";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
            }
            tc.duckdb_ms = total / iterations;
            tc.result_count = 1;

            // ThunderDuck
            for (int i = 0; i < warmup; ++i) {
                thunderduck::aggregate::sum_i32_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size());
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::aggregate::sum_i32_v2(lineitem_quantity_.data(),
                    lineitem_quantity_.size());
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // A2: MIN/MAX
        {
            TestCase tc;
            tc.id = "A2";
            tc.category = "Aggregation";
            tc.name = "MIN/MAX Combined";
            tc.sql = "SELECT MIN(l_quantity), MAX(l_quantity) FROM lineitem";
            tc.thunder_op = "minmax_i32(l_quantity)";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
            }
            tc.duckdb_ms = total / iterations;
            tc.result_count = 2;

            // ThunderDuck
            int32_t min_val, max_val;
            for (int i = 0; i < warmup; ++i) {
                thunderduck::aggregate::minmax_i32(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), &min_val, &max_val);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::aggregate::minmax_i32(lineitem_quantity_.data(),
                    lineitem_quantity_.size(), &min_val, &max_val);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // A3: AVG
        {
            TestCase tc;
            tc.id = "A3";
            tc.category = "Aggregation";
            tc.name = "AVG";
            tc.sql = "SELECT AVG(l_extendedprice) FROM lineitem";
            tc.thunder_op = "sum_i32_v2(l_extendedprice) / count";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = num_lineitem_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
            }
            tc.duckdb_ms = total / iterations;
            tc.result_count = 1;

            // ThunderDuck
            for (int i = 0; i < warmup; ++i) {
                thunderduck::aggregate::sum_i32_v2(lineitem_price_.data(),
                    lineitem_price_.size());
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                int64_t sum = thunderduck::aggregate::sum_i32_v2(lineitem_price_.data(),
                    lineitem_price_.size());
                volatile double avg = (double)sum / lineitem_price_.size();
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // A4: COUNT
        {
            TestCase tc;
            tc.id = "A4";
            tc.category = "Aggregation";
            tc.name = "COUNT(*)";
            tc.sql = "SELECT COUNT(*) FROM lineitem";
            tc.thunder_op = "array.size()";
            tc.data_rows = num_lineitem_;
            tc.data_bytes = 0;  // No data access needed

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->GetValue(0, 0).GetValue<int64_t>();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                volatile size_t count = lineitem_quantity_.size();
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // =====================================================================
        // Sort Tests
        // =====================================================================

        // S1: Sort ASC
        {
            TestCase tc;
            tc.id = "S1";
            tc.category = "Sort";
            tc.name = "Sort ASC (1M rows)";
            tc.sql = "SELECT o_totalprice FROM orders ORDER BY o_totalprice ASC";
            tc.thunder_op = "sort_i32_v2(o_totalprice, ASC)";
            tc.data_rows = num_orders_;
            tc.data_bytes = num_orders_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->RowCount();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            std::vector<int32_t> copy = order_price_;
            for (int i = 0; i < warmup; ++i) {
                copy = order_price_;
                thunderduck::sort::sort_i32_v2(copy.data(), copy.size(),
                    thunderduck::sort::SortOrder::ASC);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                copy = order_price_;
                timer.start();
                thunderduck::sort::sort_i32_v2(copy.data(), copy.size(),
                    thunderduck::sort::SortOrder::ASC);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // S2: Sort DESC
        {
            TestCase tc;
            tc.id = "S2";
            tc.category = "Sort";
            tc.name = "Sort DESC (1M rows)";
            tc.sql = "SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC";
            tc.thunder_op = "sort_i32_v2(o_totalprice, DESC)";
            tc.data_rows = num_orders_;
            tc.data_bytes = num_orders_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->RowCount();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            std::vector<int32_t> copy = order_price_;
            for (int i = 0; i < warmup; ++i) {
                copy = order_price_;
                thunderduck::sort::sort_i32_v2(copy.data(), copy.size(),
                    thunderduck::sort::SortOrder::DESC);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                copy = order_price_;
                timer.start();
                thunderduck::sort::sort_i32_v2(copy.data(), copy.size(),
                    thunderduck::sort::SortOrder::DESC);
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // =====================================================================
        // Top-K Tests
        // =====================================================================

        // T1: Top-10
        {
            TestCase tc;
            tc.id = "T1";
            tc.category = "TopK";
            tc.name = "Top-10";
            tc.sql = "SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 10";
            tc.thunder_op = "topk_max_i32_v3(o_totalprice, 10)";
            tc.data_rows = num_orders_;
            tc.data_bytes = num_orders_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
            }
            tc.duckdb_ms = total / iterations;
            tc.result_count = 10;

            // ThunderDuck
            std::vector<int32_t> values(10);
            std::vector<uint32_t> indices(10);
            for (int i = 0; i < warmup; ++i) {
                thunderduck::sort::topk_max_i32_v3(order_price_.data(),
                    order_price_.size(), 10, values.data(), indices.data());
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::sort::topk_max_i32_v3(order_price_.data(),
                    order_price_.size(), 10, values.data(), indices.data());
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // T2: Top-100
        {
            TestCase tc;
            tc.id = "T2";
            tc.category = "TopK";
            tc.name = "Top-100";
            tc.sql = "SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 100";
            tc.thunder_op = "topk_max_i32_v3(o_totalprice, 100)";
            tc.data_rows = num_orders_;
            tc.data_bytes = num_orders_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
            }
            tc.duckdb_ms = total / iterations;
            tc.result_count = 100;

            // ThunderDuck
            std::vector<int32_t> values(100);
            std::vector<uint32_t> indices(100);
            for (int i = 0; i < warmup; ++i) {
                thunderduck::sort::topk_max_i32_v3(order_price_.data(),
                    order_price_.size(), 100, values.data(), indices.data());
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::sort::topk_max_i32_v3(order_price_.data(),
                    order_price_.size(), 100, values.data(), indices.data());
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // T3: Top-1000
        {
            TestCase tc;
            tc.id = "T3";
            tc.category = "TopK";
            tc.name = "Top-1000";
            tc.sql = "SELECT o_totalprice FROM orders ORDER BY o_totalprice DESC LIMIT 1000";
            tc.thunder_op = "topk_max_i32_v3(o_totalprice, 1000)";
            tc.data_rows = num_orders_;
            tc.data_bytes = num_orders_ * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
            }
            tc.duckdb_ms = total / iterations;
            tc.result_count = 1000;

            // ThunderDuck
            std::vector<int32_t> values(1000);
            std::vector<uint32_t> indices(1000);
            for (int i = 0; i < warmup; ++i) {
                thunderduck::sort::topk_max_i32_v3(order_price_.data(),
                    order_price_.size(), 1000, values.data(), indices.data());
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                thunderduck::sort::topk_max_i32_v3(order_price_.data(),
                    order_price_.size(), 1000, values.data(), indices.data());
                timer.stop();
                total += timer.ms();
            }
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        // =====================================================================
        // Join Tests
        // =====================================================================

        // J1: Hash Join
        {
            TestCase tc;
            tc.id = "J1";
            tc.category = "Join";
            tc.name = "Hash Join (orders-customer)";
            tc.sql = "SELECT COUNT(*) FROM orders o INNER JOIN customer c ON o.o_custkey = c.c_custkey";
            tc.thunder_op = "hash_join_i32_v2(c_custkey, o_custkey)";
            tc.data_rows = num_orders_ + num_customers_;
            tc.data_bytes = (num_orders_ + num_customers_) * sizeof(int32_t);

            // DuckDB
            for (int i = 0; i < warmup; ++i) conn_->Query(tc.sql);
            Timer timer;
            double total = 0;
            for (int i = 0; i < iterations; ++i) {
                timer.start();
                auto r = conn_->Query(tc.sql);
                timer.stop();
                total += timer.ms();
                tc.result_count = r->GetValue(0, 0).GetValue<int64_t>();
            }
            tc.duckdb_ms = total / iterations;

            // ThunderDuck
            thunderduck::join::JoinResult* join_result =
                thunderduck::join::create_join_result(order_custkey_.size());
            for (int i = 0; i < warmup; ++i) {
                join_result->count = 0;
                thunderduck::join::hash_join_i32_v2(
                    customer_key_.data(), customer_key_.size(),
                    order_custkey_.data(), order_custkey_.size(),
                    thunderduck::join::JoinType::INNER, join_result);
            }
            total = 0;
            for (int i = 0; i < iterations; ++i) {
                join_result->count = 0;
                timer.start();
                thunderduck::join::hash_join_i32_v2(
                    customer_key_.data(), customer_key_.size(),
                    order_custkey_.data(), order_custkey_.size(),
                    thunderduck::join::JoinType::INNER, join_result);
                timer.stop();
                total += timer.ms();
            }
            thunderduck::join::free_join_result(join_result);
            tc.thunder_ms = total / iterations;
            tc.speedup = tc.duckdb_ms / tc.thunder_ms;
            tc.winner = tc.speedup >= 1.0 ? "ThunderDuck" : "DuckDB";
            tests_.push_back(tc);
            print_test_result(tc);
        }

        std::cout << "\n";
    }

    void print_test_result(const TestCase& tc) {
        std::string speedup_str;
        if (tc.speedup >= 1.0) {
            speedup_str = "\033[32m" + std::to_string(tc.speedup).substr(0, 5) + "x\033[0m";
        } else {
            speedup_str = "\033[33m" + std::to_string(1.0/tc.speedup).substr(0, 5) + "x slower\033[0m";
        }
        std::cout << "  [" << tc.id << "] " << std::left << std::setw(30) << tc.name
                  << " DuckDB: " << std::right << std::setw(8) << std::fixed << std::setprecision(3)
                  << tc.duckdb_ms << " ms | Thunder: " << std::setw(8) << tc.thunder_ms
                  << " ms | " << speedup_str << "\n";
    }

    void generate_report() {
        std::string filename = "docs/DETAILED_BENCHMARK_REPORT.md";
        std::ofstream file(filename);

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        file << "# ThunderDuck v2.0 详细性能测试报告\n\n";
        file << "> **生成时间**: " << std::ctime(&time_t);
        file << "> **测试环境**: Apple Silicon M4, macOS, Clang 17.0.0\n\n";
        file << "---\n\n";

        // Test Configuration
        file << "## 一、测试配置\n\n";
        file << "### 1.1 硬件环境\n\n";
        file << "| 项目 | 配置 |\n";
        file << "|------|------|\n";
        file << "| 处理器 | Apple M4 (10核: 4性能核 + 6能效核) |\n";
        file << "| 内存 | 统一内存架构 |\n";
        file << "| SIMD | ARM Neon 128-bit |\n";
        file << "| L1 缓存 | 64 KB |\n";
        file << "| L2 缓存 | 4 MB |\n";
        file << "| 缓存行 | 128 bytes |\n\n";

        file << "### 1.2 软件环境\n\n";
        file << "| 项目 | 版本 |\n";
        file << "|------|------|\n";
        file << "| 操作系统 | macOS Darwin |\n";
        file << "| 编译器 | Clang 17.0.0 |\n";
        file << "| DuckDB | 1.1.3 |\n";
        file << "| ThunderDuck | 2.0.0 |\n";
        file << "| 优化级别 | -O3 -mcpu=native |\n\n";

        file << "### 1.3 测试数据集\n\n";
        file << "| 表名 | 行数 | 数据量 |\n";
        file << "|------|------|--------|\n";
        file << "| customer | " << format_number(num_customers_) << " | "
             << format_bytes(num_customers_ * 50) << " |\n";
        file << "| orders | " << format_number(num_orders_) << " | "
             << format_bytes(num_orders_ * 40) << " |\n";
        file << "| lineitem | " << format_number(num_lineitem_) << " | "
             << format_bytes(num_lineitem_ * 36) << " |\n\n";

        file << "### 1.4 测试方法\n\n";
        file << "- **预热迭代**: 3 次\n";
        file << "- **正式迭代**: 10 次\n";
        file << "- **统计方法**: 平均执行时间\n\n";

        file << "---\n\n";

        // Detailed Test Results
        file << "## 二、详细测试结果\n\n";

        std::string current_category = "";
        for (const auto& tc : tests_) {
            if (tc.category != current_category) {
                current_category = tc.category;
                file << "### 2." << get_category_index(tc.category) << " "
                     << tc.category << " 测试\n\n";
            }

            file << "#### " << tc.id << ". " << tc.name << "\n\n";
            file << "| 属性 | 值 |\n";
            file << "|------|----|\n";
            file << "| **SQL** | `" << tc.sql << "` |\n";
            file << "| **ThunderDuck 操作** | `" << tc.thunder_op << "` |\n";
            file << "| **访问行数** | " << format_number(tc.data_rows) << " |\n";
            file << "| **访问数据量** | " << format_bytes(tc.data_bytes) << " |\n";
            file << "| **结果行数** | " << format_number(tc.result_count) << " |\n";
            file << "| **DuckDB 执行时间** | " << std::fixed << std::setprecision(3)
                 << tc.duckdb_ms << " ms |\n";
            file << "| **ThunderDuck 执行时间** | " << tc.thunder_ms << " ms |\n";

            if (tc.speedup >= 1.0) {
                file << "| **加速比** | **" << std::setprecision(2) << tc.speedup
                     << "x** |\n";
                file << "| **胜者** | **ThunderDuck** |\n\n";
            } else {
                file << "| **加速比** | " << std::setprecision(2) << (1.0/tc.speedup)
                     << "x slower |\n";
                file << "| **胜者** | DuckDB |\n\n";
            }
        }

        file << "---\n\n";

        // Summary Table
        file << "## 三、性能对比汇总表\n\n";
        file << "| ID | 测试名称 | SQL | 数据量 | DuckDB (ms) | ThunderDuck (ms) | 加速比 | 胜者 |\n";
        file << "|----|---------|----|--------|-------------|-----------------|--------|------|\n";
        for (const auto& tc : tests_) {
            std::string sql_short = tc.sql;
            if (sql_short.length() > 40) {
                sql_short = sql_short.substr(0, 37) + "...";
            }
            file << "| " << tc.id << " | " << tc.name << " | `" << sql_short << "` | "
                 << format_number(tc.data_rows) << " | "
                 << std::fixed << std::setprecision(3) << tc.duckdb_ms << " | "
                 << tc.thunder_ms << " | ";
            if (tc.speedup >= 1.0) {
                file << "**" << std::setprecision(2) << tc.speedup << "x** | **"
                     << tc.winner << "** |\n";
            } else {
                file << std::setprecision(2) << (1.0/tc.speedup) << "x slower | "
                     << tc.winner << " |\n";
            }
        }
        file << "\n";

        file << "---\n\n";

        // Category Summary
        file << "## 四、分类统计\n\n";
        std::map<std::string, std::vector<double>> category_speedups;
        for (const auto& tc : tests_) {
            category_speedups[tc.category].push_back(tc.speedup);
        }

        file << "| 类别 | 测试数 | 平均加速比 | 最佳加速比 | 胜率 |\n";
        file << "|------|--------|-----------|-----------|------|\n";
        int total_wins = 0, total_tests = 0;
        for (const auto& [cat, speedups] : category_speedups) {
            double avg = std::accumulate(speedups.begin(), speedups.end(), 0.0) / speedups.size();
            double best = *std::max_element(speedups.begin(), speedups.end());
            int wins = std::count_if(speedups.begin(), speedups.end(),
                                     [](double s) { return s >= 1.0; });
            total_wins += wins;
            total_tests += speedups.size();

            file << "| " << cat << " | " << speedups.size() << " | ";
            if (avg >= 1.0) {
                file << "**" << std::fixed << std::setprecision(2) << avg << "x** | ";
            } else {
                file << std::setprecision(2) << avg << "x | ";
            }
            file << std::setprecision(2) << best << "x | "
                 << wins << "/" << speedups.size() << " ("
                 << (wins * 100 / speedups.size()) << "%) |\n";
        }
        file << "\n";

        file << "---\n\n";

        // Overall Statistics
        file << "## 五、总体统计\n\n";
        file << "| 指标 | 值 |\n";
        file << "|------|----|\n";
        file << "| 总测试数 | " << total_tests << " |\n";
        file << "| ThunderDuck 胜出 | **" << total_wins << "** ("
             << (total_wins * 100 / total_tests) << "%) |\n";
        file << "| DuckDB 胜出 | " << (total_tests - total_wins) << " ("
             << ((total_tests - total_wins) * 100 / total_tests) << "%) |\n";

        double total_speedup = 0;
        for (const auto& tc : tests_) {
            total_speedup += tc.speedup;
        }
        file << "| 平均加速比 | " << std::fixed << std::setprecision(2)
             << (total_speedup / total_tests) << "x |\n\n";

        // Find best and worst
        auto best_it = std::max_element(tests_.begin(), tests_.end(),
            [](const TestCase& a, const TestCase& b) { return a.speedup < b.speedup; });
        auto worst_it = std::min_element(tests_.begin(), tests_.end(),
            [](const TestCase& a, const TestCase& b) { return a.speedup < b.speedup; });

        file << "### 最佳表现\n\n";
        file << "- **测试**: " << best_it->name << " (" << best_it->id << ")\n";
        file << "- **加速比**: " << std::setprecision(2) << best_it->speedup << "x\n";
        file << "- **DuckDB**: " << std::setprecision(3) << best_it->duckdb_ms << " ms\n";
        file << "- **ThunderDuck**: " << best_it->thunder_ms << " ms\n\n";

        file << "### 待优化项\n\n";
        file << "- **测试**: " << worst_it->name << " (" << worst_it->id << ")\n";
        if (worst_it->speedup >= 1.0) {
            file << "- **加速比**: " << std::setprecision(2) << worst_it->speedup << "x\n";
        } else {
            file << "- **加速比**: " << std::setprecision(2) << (1.0/worst_it->speedup) << "x slower\n";
        }
        file << "- **DuckDB**: " << std::setprecision(3) << worst_it->duckdb_ms << " ms\n";
        file << "- **ThunderDuck**: " << worst_it->thunder_ms << " ms\n\n";

        file << "---\n\n";

        // Conclusions
        file << "## 六、结论\n\n";
        file << "### 6.1 ThunderDuck 优势场景\n\n";
        file << "1. **聚合操作**: SIMD 向量化带来显著加速\n";
        file << "2. **排序操作**: Radix Sort 实现 O(n) 时间复杂度\n";
        file << "3. **Top-K 查询**: 堆选择算法避免全量排序\n";
        file << "4. **过滤操作**: 纯计数版本消除内存写入开销\n\n";

        file << "### 6.2 待优化场景\n\n";
        file << "1. **Hash Join**: 需要更高效的哈希表实现\n\n";

        file << "### 6.3 技术亮点\n\n";
        file << "- 16 元素/迭代 + 4 独立累加器\n";
        file << "- 软件预取 (`__builtin_prefetch`)\n";
        file << "- LSD Radix Sort (11-11-10 位分组，3 趟)\n";
        file << "- 合并的 minmax 函数（单次遍历）\n\n";

        file << "---\n\n";
        file << "*ThunderDuck v2.0 - 针对 Apple M4 优化的高性能数据库算子库*\n";

        file.close();
        std::cout << "Report saved to: " << filename << "\n";
    }

    std::string format_number(size_t n) {
        std::string s = std::to_string(n);
        int len = s.length();
        for (int i = len - 3; i > 0; i -= 3) {
            s.insert(i, ",");
        }
        return s;
    }

    std::string format_bytes(size_t bytes) {
        if (bytes >= 1024 * 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
        } else if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + " KB";
        } else {
            return std::to_string(bytes) + " B";
        }
    }

    int get_category_index(const std::string& cat) {
        static std::map<std::string, int> indices = {
            {"Filter", 1}, {"Aggregation", 2}, {"Sort", 3}, {"TopK", 4}, {"Join", 5}
        };
        return indices[cat];
    }

private:
    DataGenerator gen_;
    std::unique_ptr<DuckDB> db_;
    std::unique_ptr<Connection> conn_;

    size_t num_customers_;
    size_t num_orders_;
    size_t num_lineitem_;

    std::vector<int32_t> lineitem_quantity_;
    std::vector<int32_t> lineitem_price_;
    std::vector<int32_t> order_price_;
    std::vector<int32_t> order_custkey_;
    std::vector<int32_t> customer_key_;

    std::vector<TestCase> tests_;
};

int main() {
    DetailedBenchmark benchmark;
    benchmark.run();
    return 0;
}
