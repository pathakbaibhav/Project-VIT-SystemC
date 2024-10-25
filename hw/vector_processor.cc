/*
 * Vector Processor 
 */

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include <inttypes.h>

#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

using namespace sc_core;
using namespace std;

#include "vector_processor.h"
#include <sys/types.h>
#include <time.h>

#ifdef __DEBUG__
#define DEBUG_PRINT(...)              \
	do                                \
	{                                 \
		fprintf(stderr, __VA_ARGS__); \
	} while (false)
#else
#define DEBUG_PRINT(...) \
	do                   \
	{                    \
	} while (false)
#endif


// constructor
vector_processor::vector_processor(sc_module_name name)
	: sc_module(name), socket("socket"), mmr_csr("mmr_csr"), mmr_csr_resp("mmr_csr_resp")
{
	// register blocking and debug transport
	socket.register_b_transport(this, &vector_processor::b_transport);
	socket.register_transport_dbg(this, &vector_processor::transport_dbg);

	// initialize vector array to 0
	memset(mmr_v, 0, sizeof(uint8_t) * V_LEN * WORD_SIZE);

	// register processing thread
	SC_THREAD(thread_process);
}

void vector_processor::thread_process()
{

	// get the event on when mmr_csr changes
	const sc_event &ChangedEvent = mmr_csr.value_changed_event();

	mmr_csr_resp = 0; // start out indicating idle 

	while (true)
	{
		wait(ChangedEvent); // wait for new request

		// Compute vector add in 0 time at the end of the time sequence 
		switch (mmr_csr) {
			case 0: 
				// no need to do anything, this is probably just the confirmation
				// of going back to idle 
				mmr_csr_resp = mmr_csr;
				break;
			case 1: 	
				mmr_csr_resp = mmr_csr; // respond: working on it
				// Compute latency
				wait(5, SC_MS);
				for (int idx = 0; idx < V_LEN; idx++) {
					mmr_v[VC + idx] = mmr_v[VA+ idx] + mmr_v[VB + idx];
				}
				break;
			case 2: 	
				mmr_csr_resp = mmr_csr; // respond: working on it
				// Compute latency
				wait(5, SC_MS);
				for (int idx = 0; idx < V_LEN; idx++) {
					mmr_v[VC + idx] = mmr_v[VA+ idx] - mmr_v[VB + idx];
				}
				break;
			case 3: 	
				mmr_csr_resp = mmr_csr; // respond: working on it
				// Compute latency
				wait(5, SC_MS);
				for (int idx = 0; idx < V_LEN; idx++) {
					mmr_v[VC + idx] = mmr_v[VA+ idx] * mmr_v[VB + idx];
				}
				break;
			case 4: 	
				mmr_csr_resp = mmr_csr; // respond: working on it
				// Compute latency
				wait(5, SC_MS);
				for (int idx = 0; idx < V_LEN; idx++) {
					mmr_v[VC + idx] = mmr_v[VA+ idx] / mmr_v[VB + idx];
				}
				break;
			case 5: 	
				mmr_csr_resp = mmr_csr; // respond: working on it
				// Compute latency
				wait(5, SC_MS);
				for (int idx = 0; idx < V_LEN; idx++) {
					mmr_v[VC + idx] = mmr_v[VA+ idx] * mmr_v[VB + idx] + mmr_v[VC + idx];
				}
				break;
			default: 
				// should not happen as we do have the check above 
				std::cout << "invalid operation: " << mmr_csr << endl;
				mmr_csr_resp = 0; // set our response to 0, this does not work
		}
		mmr_csr_resp = 0; // indicate we are done and wait for new work
	}
}

// called when a TLM transaction arrives for this target
void vector_processor::b_transport(tlm::tlm_generic_payload &trans, sc_time &delay)
{
	tlm::tlm_command cmd = trans.get_command();
	sc_dt::uint64 addr = trans.get_address();
	unsigned char *data = trans.get_data_ptr();

	unsigned int len = trans.get_data_length();
	unsigned char *byt = trans.get_byte_enable_ptr();
	unsigned int wid = trans.get_streaming_width();

	// transactions with separate byte lanes are not supported
	if (byt != 0) {
		trans.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
		return;
	}

	// bursts not supported
	if (len > 4 || wid < len) {
		trans.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
		return;
	}
	// besides that, let everything pass 
	// note: even an access to a non existing MMR passes
	trans.set_response_status(tlm::TLM_OK_RESPONSE);

	// Annotate that this target needs 1us to think 
	// about how to answer an MMR request (not processing)
	// This delay is on top of transport delay (which the iconnect should model).
	delay += sc_time(1, SC_US);

	// force to catch up any quantum delay offset (to make it easier for now)
	wait(delay);
	delay = sc_time(0, SC_US);

	// compute current time (incl. any quantum offset if no sync above)
	sc_time now = sc_time_stamp() + delay;

	// handle reads commands
	if (cmd == tlm::TLM_READ_COMMAND) {
		static sc_time old_ts = SC_ZERO_TIME, diff;
		uint32_t v = 0;

		// special handling for reading from vectors 
		if( (addr >= MMR_VA) && (addr < MMR_V_END)){
			// compute offset within array by 
			//   convert to void* so that we can have byte offset calculation
			//   subtract base address of vectors (MMR_VA)
			memcpy(data, ((unsigned char *)mmr_v)+(addr-MMR_VA), len);
		} else {
			switch (addr)
			{
			case MMR_CSR:
				// return current status as per processor
				v = mmr_csr_resp;
				// sync up our own CSR state with the same value, so that a future write 
				// will be detected as a change
				mmr_csr = mmr_csr_resp;
				break;
			case MMR_TRACE:
				diff = now - old_ts; // diff to last TRACE read call
				v = now.to_seconds() * 1000 * 1000 * 1000; // ns 
				cout << "TRACE: "
					<< " " << now << " diff=" << diff << "\n";
				old_ts = now;
				break;
			default:
				break;
			}
			memcpy(data, &v, len);
			// trace reads
			DEBUG_PRINT("%10s,  read[%02x] len %d = %d\n",now.to_string().c_str(), (unsigned int)  addr, len, v);
		}

	// handle write commands
	} else if (cmd == tlm::TLM_WRITE_COMMAND) {
		static sc_time old_ts = SC_ZERO_TIME, diff;
		// trace writes 
		DEBUG_PRINT("%10s, write[%02x] len %d = %d\n",now.to_string().c_str(), (unsigned int)addr, len, *(unsigned int*)data);
		// special handling for reading from vectors 
		if( (addr >= MMR_VA) && (addr < MMR_V_END)){
			// compute offset within array by 
			//   convert to void* so that we can have byte offset calculation
			//   subtract base address of vectors (MMR_VA)
			memcpy(((unsigned char*)mmr_v)+(addr-MMR_VA), data, len);
		} else {
			switch (addr) {
			case MMR_CSR:
				// write out the requested new state
				mmr_csr = *(unsigned int*) data; 
				break;
			case MMR_TRACE:
				diff = now - old_ts; // diff to last TRACE write call
				cout << "TRACE: "
					<< " "
					<< hex << *(uint32_t *)data
					<< ", " << now << " diff=" << diff << "\n";
				old_ts = now;
			default:
				break;
			}
		}
	} else {
		// no other commands supported
		trans.set_response_status(tlm::TLM_COMMAND_ERROR_RESPONSE);
	}
}

unsigned int vector_processor::transport_dbg(tlm::tlm_generic_payload &trans)
{
	unsigned int len = trans.get_data_length();
	return len;
}