#include <iostream>
#include "../../common/gemm_tb.hh"


// TODO replace with own implementation in SW that calls GEMM ACC
void acc_driver(uint32_t M, uint32_t N, uint32_t K, uint32_t ALPHA,
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

int main(int argc, char *argv[])
{
	Gemm_tb tb;

    std::cout << "Running small matrix tests" << std::endl;
    tb.gemm_testcase_runner(
        std::bind(&Gemm_tb::small_mat_generator, tb, _1, _2, _3, _4),
        acc_driver,
        1000
    );

    std::cout << "Running large matrix tests" << std::endl;
    tb.gemm_testcase_runner(
        std::bind(&Gemm_tb::large_mat_generator, tb, _1, _2, _3, _4),
        acc_driver,
        500
    );

    std::cout << "Running darknet matrix tests" << std::endl;
    tb.gemm_testcase_runner(
        std::bind(&Gemm_tb::darknet_mat_generator, tb, _1, _2, _3, _4),
        acc_driver,
        3694
    );

	return 0; 
}
