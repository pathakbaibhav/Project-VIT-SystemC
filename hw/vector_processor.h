

/** @brief Vector Processor (derived from Xilinx DebugDev)
 * 
 */


/* address space definition (offsets within target) */
#define MMR_CSR 0x00
#define MMR_VA 0x04
#define MMR_VB 0x44
#define MMR_VC 0x84
#define MMR_TRACE 0xC4

/* number of elements in vector */
#define V_LEN 16

/* offset within own array in word size */
#define VA 0
#define VB (VA + V_LEN)
#define VC (VB + V_LEN)


#define WORD_SIZE 4 
// end of vector MMR Space
#define MMR_V_END (MMR_VC + V_LEN * WORD_SIZE)

class vector_processor
	: public sc_core::sc_module
{
public:
	tlm_utils::simple_target_socket<vector_processor> socket;

	sc_out<bool> irq;

	vector_processor(sc_core::sc_module_name name);
	virtual void b_transport(tlm::tlm_generic_payload &trans,
							 sc_time &delay);
	virtual unsigned int transport_dbg(tlm::tlm_generic_payload &trans);

	SC_HAS_PROCESS(vector_processor);

private:
	// thread to do the processing
	void thread_process();
	sc_signal<unsigned int> mmr_csr; // request of new CSR value
	sc_signal<unsigned int> mmr_csr_resp; // response of processor for actual value
	sc_event start;
	unsigned int mmr_v[V_LEN*3];
};