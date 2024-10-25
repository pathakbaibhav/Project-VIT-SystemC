#ifndef GEMM_TB_H
#define GEMM_TB_H
#define DEV_BOUND(BASE, LEN) \
    {                        \
        BASE, BASE + LEN     \
    }

#include "darknet_dims.hh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string.h>

// Macros that define testbench generation parameters
#define SEED 2345467345
#define MAT_VAL_LOWER_LIM 0
#define MAT_VAL_UPPER_LIM (1 << 8) - 1
#define SMALL_MAT_DIM_LOWER_LIM 1
#define SMALL_MAT_DIM_UPPER_LIM 200
#define LARGE_MAT_DIM_LOWER_LIM 200
#define LARGE_MAT_DIM_UPPER_LIM 512

// Status bar macros
#define PBSTR "||||||||||"
#define PBWIDTH 10

using namespace std::placeholders;

using DimGenerator = void(uint32_t &, uint32_t &, uint32_t &, uint32_t);

class Gemm_tb
{
private:
    std::default_random_engine eng;                        // Random value generation engine
    std::uniform_real_distribution<double> mat_val_dist;   // Uniform distribution used to generate matrix values
    std::uniform_int_distribution<int> small_mat_dim_dist; // Uniform distribution used to generate small matrices that fit in ACC
    std::uniform_int_distribution<int> large_mat_dim_dist; // Uniform distribution used to generate large matrices that will require tiling

    /// @brief Populates matrix vectors a and b with random values from the mat_val_dist of the class
    /// @param a A matrix
    /// @param b B matrix
    void generate_test_case(std::vector<uint32_t> &a, std::vector<uint32_t> &b)
    {
        std::generate(a.begin(), a.end(), [this]()
                      { return (uint32_t)mat_val_dist(eng); });
        std::generate(b.begin(), b.end(), [this]()
                      { return (uint32_t)mat_val_dist(eng); });
    }

    /// @brief used to populate matrix M, N, K
    /// @param dist distribution used to populate M, N, K values
    /// @param test_no testcase number (unused) remains present to simplify interface to all generators used by the example tb call
    void generate_mat_dims(std::uniform_int_distribution<int> dist, uint32_t &M, uint32_t &N, uint32_t &K, uint32_t test_no)
    {
        M = dist(eng);
        N = dist(eng);
        K = dist(eng);
    }

    /// @brief Loads matrix dims from darknet header
    /// @param M 
    /// @param N 
    /// @param K 
    /// @param test_no 
    void load_mat_dims_from_darknet(uint32_t &M, uint32_t &N, uint32_t &K, uint32_t test_no)
    {
        uint32_t test_case_idx = 0;
        for (test_case_idx = 0; test_case_idx < darknet_dims.size(); test_case_idx += 1)
        {
            if (test_no >= darknet_dims[test_case_idx].at("start") && \
                test_no <= darknet_dims[test_case_idx].at("stop"))
            {
                break;
            }
        }
        M = darknet_dims[test_case_idx].at("M");
        N = darknet_dims[test_case_idx].at("N");
        K = darknet_dims[test_case_idx].at("K");
    }

    /// @brief Simple progress bar from
    /// https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
    void printProgress(double percentage)
    {
        int val = (int)(percentage * 100);
        int lpad = (int)(percentage * PBWIDTH);
        int rpad = PBWIDTH - lpad;
        printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
        fflush(stdout);
    }

    /** @brief Gemm_nn extracted from Darknet but in 32bit fixed point.
     *  completes the C = ALPHA * A * B + C matrix operation, and the output C is
     *  also stored in rows (all rows are combined into one row)
     * @param MA, the number of lines in C (not transposed)
     * @param NB, the number of columns in C (not as a device)
     * @param KA's column number, C's row number (not transposed)
     * @param ALPHA coefficient
     * @param A input matrix (one-dimensional array format)
     * @param lda A number of columns (not transposed)
     * @param B input matrix (one-dimensional array format)
     * @param ldb B's number of columns (not transposed)
     * @param C input matrix (one-dimensional array format)
     * @param ldc C column number (not transposed)
     */
    void gemm_nn(uint32_t M, uint32_t N, uint32_t K, uint32_t ALPHA,
                 uint32_t *A, uint32_t lda,
                 uint32_t *B, uint32_t ldb,
                 uint32_t *C, uint32_t ldc)
    {
        uint32_t i, j, k;
        for (i = 0; i < M; ++i)
        {
            for (k = 0; k < K; ++k)
            {
                register uint32_t A_PART = ALPHA * A[i * lda + k];
                for (j = 0; j < N; ++j)
                {
                    C[i * ldc + j] += A_PART * B[k * ldb + j];
                }
            }
        }
    }

public:
    Gemm_tb() : eng(static_cast<long unsigned int>(SEED)),
                mat_val_dist(MAT_VAL_LOWER_LIM, MAT_VAL_UPPER_LIM),
                small_mat_dim_dist(SMALL_MAT_DIM_LOWER_LIM, SMALL_MAT_DIM_UPPER_LIM),
                large_mat_dim_dist(LARGE_MAT_DIM_LOWER_LIM, LARGE_MAT_DIM_UPPER_LIM)
    {
    }

    /// @brief This function is the Gemm_tb object's test case runner. Use it to test out your accelerator against different matrix dimension scenarios
    /// @param dims_generator defines the matrix dims generator function used during test case creation
    /// @param acc_driver defines the sw entry point to your accelerator to compute a matrix multiplication. acc_driver uses the same function signature as gemm_nn
    /// @param test_case_count the number of test cases generate by the runner
    template <typename MatrixDimGenerator, typename AccDriverFunc>
    void gemm_testcase_runner(MatrixDimGenerator dims_generator, AccDriverFunc acc_driver, uint32_t test_case_count)
    {
        uint32_t M, N, K, LDA, LDB, LDC;

        for (uint32_t test_no = 0; test_no < test_case_count; test_no += 1)
        {
            dims_generator(M, N, K, test_no);

            LDA = K;
            LDB = N;
            LDC = N;
            std::vector<uint32_t> a(M * K);
            std::vector<uint32_t> b(K * N);
            generate_test_case(a, b);

            uint32_t *expected = (uint32_t *)malloc(sizeof(uint32_t) * M * N);
            uint32_t *c = (uint32_t *)malloc(sizeof(uint32_t) * M * N);
            memset(expected, 0, sizeof(uint32_t) * M * N);
            memset(c, 0, sizeof(uint32_t) * M * N);

            // Generate expected
            gemm_nn(M, N, K, 1, a.data(), LDA, b.data(), LDB, expected, LDC);

            // Run accelerator
            acc_driver(M, N, K, 1, a.data(), LDA, b.data(), LDB, c, LDC);

            // Compare results
            if (memcmp(c, expected, sizeof(uint32_t) * M * N) != 0)
            {
                printf(" Test case %d failed! Matrix dimensions: {M:%d, N:%d, K:%d}\n", test_no, M, N, K);
                exit(-1);
            }
            printf(" Test case %d Matrix dimensions: {M:%d, N:%d, K:%d}", test_no + 1, M, N, K);

            free(c);
            free(expected);

            printProgress((double)(test_no + 1) / test_case_count);
        }
        printf("\n");
    }

    void small_mat_generator(uint32_t &M, uint32_t &N, uint32_t &K, uint32_t test_no)
    {
        generate_mat_dims(small_mat_dim_dist, M, N, K, test_no);
    }

    void large_mat_generator(uint32_t &M, uint32_t &N, uint32_t &K, uint32_t test_no)
    {
        generate_mat_dims(large_mat_dim_dist, M, N, K, test_no);
    }

    void darknet_mat_generator(uint32_t &M, uint32_t &N, uint32_t &K, uint32_t test_no)
    {
        load_mat_dims_from_darknet(M, N, K, test_no);
    }
};

#endif // GEMM_TB_H
