#include "swEmu.h"

#define MMR_CSR 0x00
#define MMR_VA 0x04
#define MMR_TRACE 0xC4

/// @brief Interrupt Service Routine for debug
void SWEmu::isr1()
{
    // there is a startup intertrupt for no good reason,
    // ignore it.
    if (sc_time_stamp() < sc_time(1, SC_PS))
    {
        return;
    }
    cout << "ISR1 called at " << sc_time_stamp() << endl;
}

/// @brief Interrupt Service Routine for debug
void SWEmu::isr2()
{
    // there is a startup intertrupt for no good reason,
    // ignore it.
    if (sc_time_stamp() < sc_time(1, SC_PS))
    {
        return;
    }
    cout << "ISR2  called at " << sc_time_stamp() << endl;
}

/* old testbench for vector processor a reference how
   to use SWEmu for testbenching */
void SWEmu::vector_processor_tb()
{
    
    // Allocate sw buffers
    uint32_t expected_output[16 * 3];
    uint32_t sw_buf[16 * 3];

    // Memory map device ptrs
    uint8_t *ptr = (uint8_t *)sw_emu_mmap(0x49000000, 0x100 - 1);
    uint32_t *dev_buf = (uint32_t *)(&ptr[MMR_VA]);
    uint32_t *dev_csr = (uint32_t *)(&ptr[MMR_CSR]);
    uint32_t *dev_trace = (uint32_t *)(&ptr[MMR_TRACE]);

    printf("Running vector processor tests...\n");
    // Get trace
    reg_read(dev_trace);

    // Initialize sw buffers
    for (int i = 0; i < 16; i++)
    {
        sw_buf[i] = i;
        sw_buf[i + 16] = i;
        sw_buf[i + 16 * 2] = 0;
    }
    for (int i = 0; i < 16; i++)
    {
        ((uint32_t *)expected_output)[i] = sw_buf[i];
        ((uint32_t *)expected_output)[i + 16] = sw_buf[16 + i];
        ((uint32_t *)expected_output)[i + 16 * 2] = sw_buf[i] + sw_buf[16 + i];
    }

    // Transfer sw buffers to dev buffers
    sw_emu_memcpy(dev_buf, sw_buf, 16 * 3 * sizeof(uint32_t));

    // Start add operation
    reg_write(dev_csr, 0x1);

    // Wait for operation to finish (no need to poll, we know timing is accurate in SystemC)
    usleep(5 * 1000 + 1);

    // Check to see if CSR indicates that we're done
    if (!(reg_read(dev_csr) == 0x0))
    {
        perror("Device operation not concluded\n");
        exit(-1);
    }

    // Copy contents back to sw_buf
    sw_emu_memcpy(sw_buf, dev_buf, 16 * 3 * sizeof(uint32_t));

    // Compare results
    if (memcmp(expected_output, sw_buf, 16 * 3 * sizeof(uint32_t)) != 0)
    {
        perror("Expected output not found when transferring data back from device buffers\n");
        exit(-1);
    }

    // One last trace call for good luck
    reg_read(dev_trace);

    printf("vector processor test success!\n");
}


// This is a global pointer to SWEmu for use within the acc_driver. 
// Not really the right way of encapsulation, but for F22 iteration we use 
// it to learn more about how we use SWEmu and  how to aid the transtion 
// between SWEmu and QEMU execution. 
SWEmu *pSwEmu; 


    /** @brief Gemm_nn replacement call for running HW accelerated gemm_nn. 
     *  Should completes the C = ALPHA * A * B + C matrix operation, and the output C is
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
    void acc_driver(uint32_t M, uint32_t N, uint32_t K, uint32_t ALPHA,
                uint32_t *A, uint32_t lda,
                uint32_t *B, uint32_t ldb,
                uint32_t *C, uint32_t ldc)
{
    //TODO interface with ACC
    // This is the SW implementation of gemm_nn as a starting point. 
    // Change this implemenation  to using the GEMM accelerator. 
    // Use reference to SWEmu class for communication and other primitives. 
    // See SWEmu::vector_processor_tb() as a reference in how to use it. 
    // Howver, as acc_driver is a stand alone function (not part of SWEmu), 
    // it needs to reference to the SWEmu instance
    // So a call 
    //   usleep(10)
    // turns into 
    // pSwEmu->usleep(10);

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

/// @brief GEMM  Testbench
void SWEmu::gemm_processor_tb()
{
	Gemm_tb tb; // instanciate GEMM testbench driver

    // store pointer to SWEmu instance globally for use in acc_driver 
    // (see comment at the global variable about temporary nature)
    pSwEmu = this;

    // small should work without tiling
    std::cout << "Running small matrix tests" << std::endl;
    tb.gemm_testcase_runner(
        std::bind(&Gemm_tb::small_mat_generator, tb, _1, _2, _3, _4),
        acc_driver,
        1000   // number of test cases (random sets of GEMM calls, reduce for developmen)
    );

    // This requries tiling. Disable until tiling is implemented
    std::cout << "Running large matrix tests" << std::endl;
    tb.gemm_testcase_runner(
        std::bind(&Gemm_tb::large_mat_generator, tb, _1, _2, _3, _4),
        acc_driver,
        500
    );

    // Usind actual dimensions (and repetition) that darknet would call.     
    // this should reduce the amount of surprises when moving to darknet. 
    std::cout << "Running darknet matrix tests" << std::endl;
    tb.gemm_testcase_runner(
        std::bind(&Gemm_tb::darknet_mat_generator, tb, _1, _2, _3, _4),
        acc_driver,
        3694
    );
}


void SWEmu::thread_process()
{
    // call vector processor testbench (can be disabled)
    vector_processor_tb();
    // call gemm processor testbench 
    gemm_processor_tb();
}
