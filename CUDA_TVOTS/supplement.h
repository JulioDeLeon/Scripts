#ifndef SUPPLEMENT_H
#define SUPPLEMENT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <sstream>
#include <pcap.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>

#include <cpptest.h>
#include <cpptest-textoutput.h>
#include <ctime>
#include <vector>

#include <openssl/sha.h>
using namespace std;
/*
0-12 uninteresting
13-14 length of "event data" (data+signature, I think)
15 uninteresting
16-end  data+signature (data might be variable length, signature length is variable, but calculatable)

signature part (relative to the signature beginning)
0-7 tvots timestamp
8-11 tvots sequence number
12-15 tvots epoch ID
16-(end of signature) (k+1) 20-byte secrets

For our specific pcap file:
data is 4 bytes, so:
16-19 data
20-27 tvots timestamp
28-31 tvots seq. num
32-35 tvots epoch ID
36-55 "salt" secret (only changes when the epoch ID changes)
56-75 secret
76-96 secret
 .
 .
 .
total of k=13 plain (not salt) secrets

*/

#define DEBUG 1
#define TARGET_NUMBER_PACKETS 500
#define MD_LENGTH 20
#define TARGET_PACKET_LENGTH 348
#define K_SECRET_POSITION 56
#define NUMBER_K_SECRETS 13
#define K_SECRET_LENGTH 20
#define DATA_LENGTH 4
#define DATA_POSITION 16
#define TIMESTAMP_LENGTH 8
#define TIMESTAMP_POSITION 20
#define SEQUENCE_NUMBER_LENGTH 4
#define SEQUENCE_NUMBER_POSITION 28
#define EPOCH_ID_LENGTH 4
#define EPOCH_ID_POSITION 32
#define SALT_SECRET_LENGTH 20
#define SALT_SECRET_POSITION 36
#define u_long unsigned long
#define u_int unsigned int 
#define INTS_PER_MD 5

class TVOTS_Packet {
  public:
    TVOTS_Packet(const u_char* packet);
    ~TVOTS_Packet();

    u_char* getEventData();
    u_char* getTimeStamp();
    u_char* getSequenceNumber();
    u_char* getEpochID();
    u_char* getSaltSecret();
    u_char** getKSecrets();
  private:
    u_char* event_data;
    u_char* time_stamp;
    u_char* sequence_number;
    u_char* epoch_ID;
    u_char* salt_secret;
    u_char** k_secrets;
};

typedef struct btimes {
  float mallocTime;
  float hashTime;
  float findTime;
  float memcpyHTDTime;
  float memcpyDTHTime;
  float totalTime;
} btimes;

/*
 *  For handlePackets, Since this is reading off a file, the given captured will be the whole of the data with no regard for the epoch period. For the future, the captured and pool variables should be made thread safe (This may be wrong and the whole application could be done on one thread).
 *  If epoch period were being taken into account, the pool of secrets should be flushed after each epoch.
 *
 *  This function should return true if a hash collision/successful birthday attack has been detected. 
 *  NOTICE: the pool variable may need to be contained in the function 
 */
__global__ void
findMatch(u_char* hashed, size_t hashedPitch, int hashedSize, u_char* pool, size_t poolPitch, int poolSize, int hashLength, u_int* result);

__global__ void
findMatchV2(u_char* hashed, size_t hashedPitch, u_int hashedSize, u_int* pool, u_int poolSize, u_int bValue, u_int max, u_int* result);

__host__ __device__ int
memcmpDevice(const void* s1, const void* s2, size_t n);

__host__ __device__ void
memcpyDevice(void* dest, const void* src, size_t n);

extern "C" bool 
handleData(u_char* dataPool, int dataLength, int dataPoolSize, u_char* secretPool,
  int spoolSize, u_int* ret, struct btimes* times);

extern "C" bool
handleDataV2(u_char* dataPool, u_int dataLength, u_int dataPoolSize, u_int* secretPool,
  u_int spoolSize, u_int bValue, u_int max, u_int* ret, struct btimes* times);

extern "C" bool
vectorAdditionExample(const int argc, const char **argv, float *arrA, float *arrB, float *output, int len);

void printPacketInformation(struct pcap_pkthdr *header, const u_char *packetData);

//SHA1 headings 
__host__ __device__ void 
sha1Init(u_long* total, u_long* state, u_char* buff);

__host__ __device__ void
sha1ProcessBlock(u_long* total, u_long* stae, u_char* buff, u_char data[64]);

__host__ __device__ void
sha1Update(u_long* total, u_long* state, u_char* buff, u_char* input, int len);

__host__ __device__ void
sha1Finish(u_long* total, u_long* state, u_char* buff, u_char* output);

__host__ __device__ void
sha1Device(u_char* input, int len, u_char* output);

__host__ __device__ void
setBit(u_int*x, u_int val);

__host__ __device__ void
clearBit(u_int*x, u_int val);

__host__ __device__ int
isSet(u_int* x, u_int val);

__host__ __device__ void
toggleBit(u_int*, u_int val);

__host__ __device__ int
getNthIdentifier(u_int* buff, int nth, int bValue, int length);

extern "C" bool
sha1Kernel(u_char* src, int len, u_char* dest);

extern "C" bool
sha1Kernel2D(u_char* src, int width, int height, u_char* dst, struct btimes *times);

/*
 * 32-bit integer manipulation macros (big endian)
  */
#ifndef GET_UINT32_BE
#define GET_UINT32_BE(n,b,i)\
{\
  (n) = ( (unsigned long) (b)[(i) ] << 24 )\
      | ( (unsigned long) (b)[(i) + 1] << 16 )\
      | ( (unsigned long) (b)[(i) + 2] <<  8 )\
      | ( (unsigned long) (b)[(i) + 3]       );\
}
#endif

#ifndef RETURN_UINT32_BE
#define RETURN_UINT32_BE(b,i)\
(\
  ( (unsigned long) (b)[(i) ] << 24 )\
  | ( (unsigned long) (b)[(i) + 1] << 16 )\
  | ( (unsigned long) (b)[(i) + 2] <<  8 )\
  | ( (unsigned long) (b)[(i) + 3]       )\
)
#endif


#ifndef GET_UINT32_BE_GPU
#define GET_UINT32_BE_GPU(n,b,i)\
{\
    (n) = ( (unsigned long) (b)[(i) + 3] << 24 )\
        | ( (unsigned long) (b)[(i) + 2] << 16 )\
        | ( (unsigned long) (b)[(i) + 1] <<  8 )\
        | ( (unsigned long) (b)[(i) ]       );\
}
#endif


#ifndef PUT_UINT32_BE
#define PUT_UINT32_BE(n,b,i)\
{\
    (b)[(i)    ] = (unsigned char) ( (n) >> 24 ); \
    (b)[(i) + 1] = (unsigned char) ( (n) >> 16 ); \
    (b)[(i) + 2] = (unsigned char) ( (n) >>  8 ); \
    (b)[(i) + 3] = (unsigned char) ( (n)       ); \
}
#endif


#define TRUNCLONG(x)  (x)
/* Circular rotation to the right for 32 bit word */
#define ROTATER(x,n)  (((x) >> (n)) | ((x) << (32 - (n))))
/* Shift to the right */
#define SHIFTR(x,n)   ((x) >> (n))

/* Little-Endian to Big-Endian for 32 bit word */
#define LETOBE32(i) (((i) & 0xff) << 24) + (((i) & 0xff00) << 8) + (((i) & 0xff0000) >> 8) + (((i) >> 24) & 0xff)
/* Return number of 0 bytes to pad */
#define padding_256(len)  (((len) & 0x3f) < 56) ? (56 - ((len) & 0x3f)) : (120 - ((len) & 0x3f))
#endif
