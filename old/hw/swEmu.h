#ifndef MYMASTER_H
#define MYMASTER_H
#define DEV_BOUND(BASE, LEN) \
    {                        \
        BASE, BASE + LEN     \
    }

#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/tlm_quantumkeeper.h"
#include <errno.h>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <functional>
#include "../common/gemm_tb.hh"

#define DEV_BOUND(BASE, LEN) \
    {                        \
        BASE, BASE + LEN     \
    }

using namespace sc_core;
using namespace std;
using namespace placeholders;

///@brief Software Emulation
///       Host compiled simulation of SW, much faster than QEMU to test out
///       basic interactions
class SWEmu : public sc_module
{
public:
    tlm_utils::simple_initiator_socket<SWEmu> socket; // master interface to access bus
    sc_in<bool> irq1, irq2;                           // irq ports into SWEmu
    tlm::tlm_generic_payload *trans;                  // payload for transactions
    tlm_utils::tlm_quantumkeeper m_qk;                // Quantum keeper for temporal decoupling
    sc_time timeOffset;                               // local time offset to systemc time
    uint32_t data;                                    // Internal data buffer used by initiator with generic payload
    unsigned long long baseAddr;                      // device base address
    unsigned int retVal = 0;                          // return value for simulation
    std::vector<std::pair<uint64_t, uint64_t>> dev_bounds;

    SWEmu(sc_module_name name) : sc_module(name),
                                 socket("socket")
    {
        SC_THREAD(thread_process);

        // emulate interrupt service routine being called when
        // the irq rises
        SC_METHOD(isr1);
        sensitive << irq1.pos();

        SC_METHOD(isr2);
        sensitive << irq2.pos();

        // reset timeOffset within quantum (all processes use one global quantum)
        m_qk.reset();

        // allocate payload we will use it for all transactions
        // there is only one transaction at any given time
        trans = new tlm::tlm_generic_payload;
        trans->set_data_length(4);
        trans->set_streaming_width(4); // = data_length to indicate no streaming
        trans->set_byte_enable_ptr(0); // 0 indicates unused
        trans->set_dmi_allowed(false); // Mandatory initial value
    }

    /// @brief write a word onto socket using fault transaction
    /// @param offset target address offset in bytes
    /// @param value to write
    uint32_t transaction(uint32_t *addr, uint32_t *value)
    {
        timeOffset = m_qk.get_local_time();

        // use common transaction (with present sizes ... )
        // all transactions are 32 bit

        if (value != NULL)
        {
            trans->set_command(tlm::TLM_WRITE_COMMAND);
            data = *value;
        }
        else
        {
            trans->set_command(tlm::TLM_READ_COMMAND);
            data = 0;
        }
        // No DMI, so use blocking transport interface
        trans->set_address((uint64_t)addr);
        trans->set_data_ptr((uint8_t *)(&data));
        trans->set_response_status(tlm::TLM_INCOMPLETE_RESPONSE); // Mandatory initial value

        socket->b_transport(*trans, timeOffset);

        if (trans->is_response_error())
            SC_REPORT_ERROR("TLM-2", trans->get_response_string().c_str());

        m_qk.set(timeOffset); // update the time with what came back from target
        if (m_qk.need_sync())
            m_qk.sync();

        return data;
    }

    /// @brief check if pointer is to a physical address (or virtual in our memory)
    /// @param addr to check
    /// @return true if physical address reserved by swemu_mmap()
    bool is_device(uint32_t addr)
    {
        for (const auto &bounds : dev_bounds)
        {
            if (addr >= bounds.first && addr < bounds.second)
            {
                return true;
            }
        }
        return false;
    }

    /// @brief  Read 32bit from
    /// @param addr pointer to physical address (with offset)
    /// @return value read
    uint32_t reg_read(uint32_t *addr)
    {
        if (!is_device((uint64_t)addr))
        {
            perror("Invalid reg read requested, addr does not specify a device. Did you forget to mmap this device?\n");
            exit(-1);
        }
        return transaction(addr, NULL);
    }

    /// @brief Write 32bit to
    /// @param addr  pointer to physical address (with offset)
    /// @param value to write
    void reg_write(uint32_t *addr, uint32_t value)
    {
        if (!is_device((uint64_t)addr))
        {
            perror("Invalid reg read requested, addr does not specify a device. Did you forget to mmap this device?\n");
            exit(-1);
        }
        transaction(addr, &value);
    }

    /// @brief Memcopy wich can use SWEmu phys addr pointer from sw_emu_mmap
    /// @param dst   address to write to
    /// @param src   address to read from
    /// @param size  size in bytes 
    void sw_emu_memcpy(void *dst, void *src, size_t size)
    {
        bool srcDev = is_device((uint64_t)src);  // is src a phys device?
        bool dstDev = is_device((uint64_t)dst);  // is dst a phys device?

        // each transaction is 32 bit, how many do we need?
        unsigned int nTrans = size/sizeof(uint32_t);

        if (!srcDev && !dstDev) {
            // both in virtual use regular memcpy
            memcpy(dst, src, size);    
        } else if (!srcDev && dstDev) {
            // dst in phys but src in virtual
            // write through transactions
            for (size_t i = 0; i < nTrans; i++)
            {
                transaction( ((uint32_t*)dst) + i , ((uint32_t*)src) + i );
            }
        } else if (srcDev && !dstDev) {
            // src phys to dest virt 
            // copy via read transactions
            for (size_t i = 0; i < nTrans; i++)
            {
                *(((uint32_t*)dst) + i) = transaction(((uint32_t*)src) + i, NULL);
            }
        } else {
            // device to device copy, not sure we need it but it can be done ... 
            uint32_t buf;             
            for (size_t i = 0; i < nTrans; i++)
            {
                // read from device into buffer
                buf = transaction(((uint32_t*)src) + i, NULL);
                // write from buffer into device 
                transaction( ((uint32_t*)dst) + i , &buf );
            }
        }
    }

    /// @brief Get pointer to physcial address and range
    ///        returned pointer cannot be read from or written to  directly
    ///        (this will cause seg fault to indicate wrong access)
    ///         Only use this pointer with sw_emu_memcpy, reg_write, reg_read
    /// @param address start address (physical)
    /// @param range   range
    /// @return pointer to SWEmu physcial memory do not access directly but through
    ///         sw_emu_memcpy, reg_write, reg_read
    void *sw_emu_mmap(uint64_t address, uint64_t range)
    {
        // Reserve a virtual address range on the simulation host that has the same address as
        // the physical address of the target. By doing so, we can use the returned address just
        // like on the SW side (and perform pointer arith).
        // Reserving avoids that the host process uses the same address range by chance.
        // If address range is already in use, mmap should fail.
        // Assumption/Limiation: only no two SWEmus can talk to the same address range, but
        // but for this we could just pick a random range (actual virtual address does not matter)

        /// This prevents any potential conflict between virtual memory allocated by the host kernel
        /// and device memory in SystemC. Any attempt to access this reserved memory will trigger a
        /// seg fault. Data from devices can only be accessed after a memcpy to a buffer allocated on the host.
        /// https://stackoverflow.com/questions/2782628/any-way-to-reserve-but-not-commit-memory-in-linux
        size_t page_size = sysconf(_SC_PAGESIZE); /// get page size
        if (range > page_size)
        {
            page_size = (size_t)ceil((double)range / page_size) * page_size;
        }
        // mmap to /dev/zero (not /dev/mem) as we don't actually want to read or write anything using this pointer
        int fd = open("/dev/zero", O_RDWR);
        if (fd < 1)
        {
            perror("Failed to open file descriptor to /dev/zero\n");
            exit(errno);
        }
        // use PROT_NONE to avoid any access
        void *pDev = mmap((void *)address, page_size, PROT_NONE, MAP_SHARED, fd, (address & ~(page_size - 1)));
        if (pDev == (void *)-1)
        {
            perror("Failed to reserve device virtual address range\n");
            exit(errno);
        }

        // Add device bounds to SWEmu so that it figure out where it should move data to
        dev_bounds.push_back(DEV_BOUND(address, range));
        return (void *)address;
    }

    /// @brief increase local time and sync if needed
    /// @param incTime in micro seconds
    void usleep(unsigned int usec)
    {
        m_qk.inc(sc_time(usec, SC_US)); // delay for usec
        if (m_qk.need_sync())           // sync quantum if needed
            m_qk.sync();
    }

    unsigned int retVal_get() { 
        return retVal; 
    }

    /// @brief Interrupt Service Routine, called on rising IRQ
    void isr1();
    /// @brief Interrupt Service Routine, called on rising IRQ
    void isr2();

    /// @brief user process (just one)
    void thread_process();

    /// @brief existing vector processor testbench
    void vector_processor_tb();

    /// @brief existing vector processor testbench
    void gemm_processor_tb();

    SC_HAS_PROCESS(SWEmu);
};

#endif // MYMASTER_H
