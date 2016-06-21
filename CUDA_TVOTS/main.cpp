#include "supplement.h"
using namespace std;


void quickVectorExample();

int 
main(int argc, char** argv){
  findCudaDevice(argc, (const char**) argv);

  //Pcap initializations
  const char* fileName = (argv[1])? (const char*) argv[1] : "tvots_gs_packets.pcapng";
  
#ifdef DEBUG
  printf("Will open: %s \n", fileName);
#endif
    
  pcap_t* test_handle;
  char* dev;
  //char test[5] = "eth0";
  struct pcap_pkthdr header;
  const u_char* packet;
  int pktcnt = 0;
  int secretsSeen = 0;
  int errBuffSize = 1024;
  char* errBuff = new char[errBuffSize];
  u_char** dataBuff = new u_char*[TARGET_NUMBER_PACKETS];
  u_char** secretBuff = new u_char*[TARGET_NUMBER_PACKETS * (1 + NUMBER_K_SECRETS)];
  pcap_t* pcapH = pcap_open_offline(fileName, errBuff);
  if(pcapH == NULL){
    printf("Failed to open %s\n", fileName);
    exit(1);
  }
 
  /* Open the session in promiscuous mode 
  test_handle = pcap_open_live(test, BUFSIZ, 1, 1000, errbuff);
  if (test_handle == NULL) {
    fprintf(stderr, "Couldn't open device %s: %s\n", dev, errbuff);
    return(2);
  }
  */

#ifdef DEBUG
  printf("Entering PCAP Example\n");
#endif
    
  packet = pcap_next(pcapH, &header);
  while (packet && pktcnt < TARGET_NUMBER_PACKETS) {
    if (header.len != TARGET_PACKET_LENGTH) {     
      packet = pcap_next(pcapH, &header);
      continue;
    }

    //repurpose the following code to be static in class 
    dataBuff[pktcnt] = new u_char[DATA_LENGTH];
    memcpy((char*)dataBuff[pktcnt], packet + DATA_POSITION, DATA_LENGTH);
    for(int i = 0; i < NUMBER_K_SECRETS; i++){
      secretBuff[secretsSeen] = new u_char[K_SECRET_LENGTH];
      memcpy((char*)secretBuff[secretsSeen], packet + K_SECRET_POSITION + (i * K_SECRET_LENGTH), K_SECRET_LENGTH);
      secretsSeen++;
    }
    
    
    ++pktcnt;
    packet = pcap_next(pcapH, &header);
  }
  printf("[%d packets captured]\n", pktcnt);
  
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  //TODO: Drop handle packets in the segement of code here
  //bool succ = handleData(dataBuff, pktcnt, secretBuff, secretsSeen. NULL);
  
  sdkStopTimer(&timer);
  printf("[Processing time: %f (ms)]\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  delete[] dataBuff;
  delete[] secretBuff;
  delete[] errBuff;
  pcap_close(pcapH);
  //printf("[%s]\n", (succ)?"PASSED":"FAILED");
  checkCudaErrors(cudaDeviceReset());
  return 0;
}



void quickVectorExample(){
  //----------------------------------------------------------------------------
  //Vector Example-        	
#ifdef DEBUG  
  printf("Entering Vector Example\n");
#endif
  int numElem = 5000000;
  size_t memSize = numElem * sizeof(float);
  printf("[Vector Addition of %d  elements]\n", numElem);

  float* arrA = (float*) malloc(memSize);
  float* arrB = (float*) malloc(memSize);
  float* output = (float*) malloc(memSize);

  for(int i = 0; i < numElem; ++i){
    arrA[i] = rand()/(float)RAND_MAX;
    arrB[i] = rand()/(float)RAND_MAX;
  }
  
  bool succ = vectorAdditionExample(0,0, arrA, arrB, output, numElem);
 
  free(arrA);
  free(arrB);
  free(output);
  //-----------------------------------------------------------------------------
  //end vector example
}
