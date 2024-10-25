#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/// base address of debug device (physical)
#define SYSTEMC_DEVICE_ADDR (0x49000000ULL)

#define MMR_CSR 0x00 
#define MMR_VA  0x04
#define MMR_VB  0x44
#define MMR_VC  0x84  
#define MMR_TRACE  0xC4


// number of elements in vector 
#define N_ELEM 16


#define REG_WRITE(base, offset, val)
#define REG_READ(base, offset)  


int main(int argc, char *argv[])
{
	int fd;       /// file descriptor to phys mem
	volatile char *pDev;   /// pointer to base address of device (mapped into user's virtual mem)
	unsigned page_size=sysconf(_SC_PAGESIZE);  /// get page size 

	//open device file representing physical memory 
	fd=open("/dev/mem",O_RDWR);
	if(fd<1) {
		perror(argv[0]);
		exit(-1);
	}


	/// get a pointer in process' virtual memory that points to the physcial address of the device
	pDev= (char*)mmap(NULL,page_size,PROT_READ|PROT_WRITE,MAP_SHARED,fd,(SYSTEMC_DEVICE_ADDR & ~(page_size-1)));
	
	// for ease of addressing define pointers to the individual MMRs
    volatile unsigned int* dev_VA =    (unsigned int*)(&pDev[MMR_VA]);
    volatile unsigned int* dev_VB =    (unsigned int*)(&pDev[MMR_VB]);
    volatile unsigned int* dev_VC =    (unsigned int*)(&pDev[MMR_VC]);
    volatile unsigned int* dev_CSR =   (unsigned int*)(&pDev[MMR_CSR]);
    volatile unsigned int* dev_trace = (unsigned int*)(&pDev[MMR_TRACE]);


	 // write to the trace register so say we are starting
	*dev_trace = 1; 

	// poll until idle
	unsigned int timeout = 100; 
	while(*dev_CSR > 0) {
		timeout--;
		if (timeout<=0){
			printf("Timout waiting for idle\n");
			exit(1);
		}
		usleep(1000);
	}

	*dev_trace = 2; // fill 

    // write input vectors into acc
    for(int i = 0; i < N_ELEM ; i++)
    {	// write to MMRs on device side C is cool [] automatically derefereces
        dev_VA[i] = i;
        dev_VB[i] = i;
        dev_VC[i] = 0;
    }

	*dev_trace = 3; // kick off 

    // Start add operation
	*dev_CSR = 0x1;

	// poll until idle
	timeout = 100; 
	while(*dev_CSR > 0) {
		timeout--;
		if (timeout<=0){
			printf("Timout waiting for idle\n");
			exit(1);
		}
		usleep(1000);
	}

	*dev_trace = 4; // SW sees idle 

    // compare result output 
    for(int i = 0; i < N_ELEM ; i++)
    {
		unsigned int exp, res; 		
		
		// for sanity also compare input 
		exp = i; 
		res = dev_VA[i];
		if ( exp != res ) {
			printf("Invalid VA[%d] %d != %d\n",  i, exp , res );
		}

		res = dev_VB[i];
		if ( exp != res ) {
			printf("Invalid VB[%d] %d != %d\n",  i, exp , res );
		}

		exp = i+i; 
		res = dev_VC[i];
		if(exp != res) {
			printf("Invalid result for VC[%d] %d != %d\n",  i, exp , res );
			exit(1);
		}
    }
	
	printf("All tests passed. Yay!\n");

	return 0; 
}
