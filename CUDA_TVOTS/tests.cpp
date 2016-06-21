#include "supplement.h"
using namespace std;
//http://cpptest.sourceforge.net/tutorial.html

void initBtimes(struct btimes *x){
  x->mallocTime = 0.0;
  x->memcpyDTHTime = 0.0;
  x->memcpyHTDTime = 0.0;
  x->hashTime = 0.0;
  x->findTime = 0.0;
  x->totalTime = 0.0;
}

string GetHexRepresentation(const unsigned char * Bytes, size_t Length){
    std::ostringstream os;
    os.fill('0');
    os<<std::hex;
    for(const unsigned char * ptr=Bytes;ptr<Bytes+Length;ptr++)
        os<<std::setw(2)<<(unsigned int)*ptr;
    return os.str();
}

class CudaKernelTestSuite : public Test::Suite {
public:
  CudaKernelTestSuite(){
//    TEST_ADD(CudaKernelTestSuite::first_test)
//    TEST_ADD(CudaKernelTestSuite::PacketClassDataTest)
//    TEST_ADD(CudaKernelTestSuite::GpuMemcmpTest)
//    TEST_ADD(CudaKernelTestSuite::GpuMemcpyTest)
//    TEST_ADD(CudaKernelTestSuite::SHA1GpuTest)
 //   TEST_ADD(CudaKernelTestSuite::SHA1GpuTest2D)
//    TEST_ADD(CudaKernelTestSuite::SHA1GpuTest2DComplex)
//    TEST_ADD(CudaKernelTestSuite::SHA1CPUTest)
//    TEST_ADD(CudaKernelTestSuite::GPU2dHashFindBench)
//    TEST_ADD(CudaKernelTestSuite::GPU2dHashFind)
//    TEST_ADD(CudaKernelTestSuite::MockAttack)
	  TEST_ADD(CudaKernelTestSuite::GetIdentifierTest)

  }
private:  
  void first_test();
  void PacketClassDataTest();
  void SHA1GpuTest();
  void SHA1GpuTest2D();
  void SHA1GpuTest2DComplex();
  void GpuMemcmpTest();
  void GpuMemcpyTest();
  void SHA1CPUTest();
  void GPU2dHashFind();
  void GPU2dHashFindBench();
  void GetIdentifierTest();
  void MockAttack();
};

void CudaKernelTestSuite::first_test(){
  clock_t start, end;
  double elapsed;
  start = time(NULL);
  // test timing of functions here
  end = time(NULL);
  elapsed = difftime(start, end);
  TEST_ASSERT(1 == 1); //CHnage this assert to check if functions have past some time benchmark
}

void CudaKernelTestSuite::SHA1CPUTest(){
    
  u_char hash1[20] = {};
  u_char hash2[20] = {};
  u_char t1[] = "abc";
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";
  u_char t2[] =  "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  u_char e2[] = "\x84\x98\x3e\x44\x1c\x3b\xd2\x6e\xba\xae\x4a\xa1\xf9\x51\x29\xe5\xe5\x46\x70\xf1";
  
  
  SHA1(t1, strlen((const char*) t1), hash1);
  sha1Device(t1, strlen((const char*)t1), hash2); 
  TEST_ASSERT(0 == memcmp(e1, hash1, 20)); 
  TEST_ASSERT(0 == memcmp(e1, hash2, 20)); 
  
  SHA1(t2, strlen((const char*) t2), hash1);
  sha1Device(t2, strlen((const char*) t2), hash2); 
  TEST_ASSERT(0 == memcmp(e2, hash1, 20)); 
  TEST_ASSERT(0 == memcmp(e2, hash2, 20)); 
  
}

void CudaKernelTestSuite::SHA1GpuTest2D(){
  int target = 10;
  u_char t1[] = "abc";
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

  u_char data1[target * (strlen((const char*) t1))];
  memset(data1, 0, (target* strlen((const char*) t1)));
  u_char out1[target * MD_LENGTH];
  memset(out1, 0, (target*MD_LENGTH*sizeof(u_char)));

  //fill input
  u_char* temp = data1;
  int tlen = strlen((const char*) t1);
  for(int i = 0; i < target; i++){
    memcpy(temp + (i*tlen),  t1, tlen);
  }
  u_char hash[20];
  SHA1(data1, tlen, hash);

  TEST_ASSERT(0 == memcmp(hash, e1, MD_LENGTH));
  //do the hashing of data set 1
   u_char hash2[20];
  sha1Kernel(data1, tlen, hash2);

  TEST_ASSERT(0 == memcmp(hash2, e1, MD_LENGTH));


  sha1Kernel2D(data1, tlen, target, out1, NULL);
  u_char* tempRep = new u_char[20];
  for(int i = 0; i < target; i++){
	  memcpy(tempRep, out1 + (i * MD_LENGTH), MD_LENGTH);

	  TEST_ASSERT(0 == memcmp(tempRep, e1, MD_LENGTH))
  }


  delete[] tempRep;
  //compare the results of hash 1 with expected
  //do the hashing of data set 2


  TEST_ASSERT(0==0);
}

void CudaKernelTestSuite::SHA1GpuTest2DComplex(){
  int target = 10000;
  int maxRange = 1000;
  struct btimes gpuTimes;
  struct btimes cpuTimes;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  u_char* data = new u_char[target * sizeof(int)];
  memset(data, 0, (target* sizeof(int)));
  u_char* check = new u_char[target * MD_LENGTH];
  memset(check, 0, (target*MD_LENGTH*sizeof(u_char)));
  u_char* out = new u_char[target * MD_LENGTH];
  memset(out, 0, (target*MD_LENGTH*sizeof(u_char)));
  initBtimes(&cpuTimes);
  initBtimes(&gpuTimes);  
  int* temp = (int*)data;
  u_char* temp2 = check;
  u_char* temp3 = out;

  for(int i = 0; i < target; i++){
	  //memcpy(temp + (i*tlen),  t1, tlen);
	  //This time fill data array with random values;
	  int ranNum = 1 + (rand() % maxRange);
	  temp[i] = ranNum;
  }

  sdkStartTimer(&timer);
  //hash all the random values into a checking array;
  for(int i = 0; i < target; i++){
	  SHA1(data + (i*sizeof(int)), sizeof(int), check + (i*MD_LENGTH));
  }
  sdkStopTimer(&timer);
  cpuTimes.hashTime = sdkGetTimerValue(&timer);
  sdkResetTimer(&timer);

  //printing array
  for(int i = 0; i < target; i++){
	  u_char tempMsg[MD_LENGTH];
	  memcpy(tempMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);
	  //printf("CPU[%d] => [%s]\n", temp[i], GetHexRepresentation(tempMsg, MD_LENGTH).c_str() );
  }
  //hash the random values on the GPU
  sha1Kernel2D(data, sizeof(int), target, out, &gpuTimes);

  for(int i = 0; i < target; i++){
	  u_char tempMsg[MD_LENGTH];
	  memcpy(tempMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);
          //printf("GPU[%d] => [%s]\n", temp[i], GetHexRepresentation(tempMsg, MD_LENGTH).c_str() );
  }

  //checking array
 // //printf("Checking correctness\n")
  for(int i = 0; i < target; i++){
	  u_char correctMsg[MD_LENGTH]; //we assume any sha1 hash produced by openssl is the target we want
	  memcpy(correctMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);

	  u_char gpuMsg[MD_LENGTH];
	  memcpy(gpuMsg, temp2 +(i*MD_LENGTH), MD_LENGTH);
	  //printf("[%d] => [%s]\n", temp[i], GetHexRepresentation(tempMsg, MD_LENGTH).c_str() );
	  TEST_ASSERT(0 == memcmp(correctMsg, gpuMsg, MD_LENGTH));
  }
  cpuTimes.totalTime = cpuTimes.hashTime + cpuTimes.findTime; 
  gpuTimes.totalTime = gpuTimes.hashTime + gpuTimes.findTime + gpuTimes.mallocTime + gpuTimes.memcpyDTHTime + gpuTimes.memcpyHTDTime;

  printf("Test %s:\n", __PRETTY_FUNCTION__);
  printf("Target size: %d\n", target);
  printf("CPU hash time\t\t[%f ms]\n", cpuTimes.hashTime);
  printf("CPU total time\t\t[%f ms]\n", cpuTimes.totalTime);
  printf("GPU malloc time\t\t[%f ms]\n", gpuTimes.mallocTime);
  printf("GPU memcpyHTD time\t[%f ms]\n", gpuTimes.memcpyHTDTime);
  printf("GPU hash time\t\t[%f ms]\n", gpuTimes.hashTime);
  printf("GPU memcpyDTH time\t[%f ms]\n", gpuTimes.memcpyDTHTime);
  printf("GPU total time\t\t[%f ms]\n", gpuTimes.totalTime);
  delete[] data;
  delete[] out;
  delete[] check;
  sdkDeleteTimer(&timer);
  cudaDeviceReset();
}

void CudaKernelTestSuite::PacketClassDataTest(){
  char filename[] = "tvots_gs_packets.pcapng";
  const u_char* packet;
  char* tmp;
  int errBuffSize = 1024;
  struct pcap_pkthdr header;
  char* errBuff = new char[errBuffSize];
  pcap_t* pcapH = pcap_open_offline(filename, errBuff);
  int check_target = 1;
  int pktcnt = 0;

  TEST_ASSERT(pcapH != NULL);
  do{
    packet = pcap_next(pcapH, &header);
    if(header.len != TARGET_PACKET_LENGTH) continue;
    if(!packet) continue;
    tmp = (char*) packet;
    TVOTS_Packet* captured = new TVOTS_Packet(packet);
    //check fields of the new packet to the original.
    
    //data
    TEST_ASSERT(0 ==  memcmp(packet + DATA_POSITION, captured->getEventData(), DATA_LENGTH));
    //timestamp
    TEST_ASSERT(0 == memcmp(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    //epoch
    TEST_ASSERT(0 == memcmp(packet + EPOCH_ID_POSITION, captured->getEpochID(), EPOCH_ID_LENGTH));
    //secret
    TEST_ASSERT(0 == memcmp(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    
    //ksecrets
    u_char** k_secrets = captured->getKSecrets();
    for(int i = 0; i < NUMBER_K_SECRETS; i++){
      TEST_ASSERT(0 == memcmp(k_secrets[i], packet + K_SECRET_POSITION + (i * K_SECRET_LENGTH), K_SECRET_LENGTH));  
    }

    pktcnt++;
  }while(packet && pktcnt < check_target);
  
  delete errBuff;
}

void CudaKernelTestSuite::SHA1GpuTest(){
  u_char t1[] = "abc";
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";
  u_char t2[] = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"; 
  u_char e2[] = "\x84\x98\x3e\x44\x1c\x3b\xd2\x6e\xba\xae\x4a\xa1\xf9\x51\x29\xe5\xe5\x46\x70\xf1";
  
  u_char hash1[20];
  u_char hash2[20];
  

  //openssl is used for reference to test correctness of the inhouse sha1 implemenations;
  SHA1(t1, strlen((const char*) t1), hash1);
  TEST_ASSERT(0 == memcmp(hash1, e1, 20));
  //gpu hash1  
  TEST_ASSERT(sha1Kernel(t1, strlen((const char*) t1), hash2));
  TEST_ASSERT(0 == memcmp(hash1, hash2, 20));

  SHA1(t2, strlen((const char*) t2), hash1);
  TEST_ASSERT(0 == memcmp(hash1, e2, 20));
  //gpu hash 2
  TEST_ASSERT(sha1Kernel(t2, strlen((const char*) t2), hash2));
  TEST_ASSERT(0 == memcmp(hash1, hash2, 20));
  cudaDeviceReset();
  
}

void CudaKernelTestSuite::GpuMemcmpTest(){
//remote comment from insight
  char filename[] = "tvots_gs_packets.pcapng";
  const u_char* packet;
  char* tmp;
  int errBuffSize = 1024;
  struct pcap_pkthdr header;
  char* errBuff = new char[errBuffSize];
  pcap_t* pcapH = pcap_open_offline(filename, errBuff);
  int check_target = 1;
  int pktcnt = 0;

  TEST_ASSERT(pcapH != NULL);
  do{
    packet = pcap_next(pcapH, &header);
    if(header.len != TARGET_PACKET_LENGTH) continue;
    if(!packet) continue;
    tmp = (char*) packet;
    TVOTS_Packet* captured = new TVOTS_Packet(packet);
    //check fields of the new packet to the original.
    
    //data
    TEST_ASSERT(0 ==  memcmpDevice(packet + DATA_POSITION, captured->getEventData(), DATA_LENGTH));
    //timestamp
    TEST_ASSERT(0 == memcmpDevice(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    //epoch
    TEST_ASSERT(0 == memcmpDevice(packet + EPOCH_ID_POSITION, captured->getEpochID(), EPOCH_ID_LENGTH));
    //secret
    TEST_ASSERT(0 == memcmpDevice(packet + TIMESTAMP_POSITION, captured->getTimeStamp(), TIMESTAMP_LENGTH));
    
    //ksecrets
    u_char** k_secrets = captured->getKSecrets();
    for(int i = 0; i < NUMBER_K_SECRETS; i++){
      TEST_ASSERT(0 == memcmpDevice(k_secrets[i], packet + K_SECRET_POSITION + (i * K_SECRET_LENGTH), K_SECRET_LENGTH));  
    }

    //improper data size test
    TEST_ASSERT( 0 != memcmpDevice(captured->getEventData(), "a", DATA_LENGTH));
    TEST_ASSERT( 0 != memcmpDevice(captured->getEventData(), "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", DATA_LENGTH));

    pktcnt++;
  }while(packet && pktcnt < check_target);
  
  delete errBuff;
}

void CudaKernelTestSuite::GpuMemcpyTest(){
  char t1[] = "012345678901234567890";
  char* d1 = new char[strlen(t1)];

  memcpyDevice(d1, t1, strlen(t1));
  TEST_ASSERT(0 == memcmp(t1, d1, strlen(t1)));
  TEST_ASSERT(0 != memcmp(d1, "123123235234", strlen(d1)));
}


void CudaKernelTestSuite::GPU2dHashFind(){
  int target = 1024;
  int secretMultiple = 14; //This number is based off a ratio of number of secrets to data points
  int maxRange = 1000;
  struct btimes gpuTimes;
  struct btimes cpuTimes;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  srand(time(NULL));
  u_char *data, *gpuH, *cpuH, *pool;
  u_int cpuResult, gpuResult;
  printf("Test %s:\n", __PRETTY_FUNCTION__);
  initBtimes(&gpuTimes);
  initBtimes(&cpuTimes);

  data = new u_char[target * sizeof(int)];
  memset(data, 0, (target* sizeof(int)));
  gpuH = new u_char[target * MD_LENGTH];
  memset(gpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
  cpuH = new u_char[target * MD_LENGTH];
  memset(cpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
  pool = new u_char[secretMultiple*target * MD_LENGTH];
  memset(pool, 0, (secretMultiple*target*MD_LENGTH*sizeof(u_char)));
  cpuResult = 0;
  gpuResult = 0;

  int ranDataIndex = (rand() % target); //This will be a random index which our hard example t1 will be inserted in our data pool
  int ranCheckIndex =  (rand() % (target*secretMultiple)); //This will be the random index which out hard check e1 will be inserted in our secrets
  u_char t1[] = "abc";
  int tlen = strlen((const char*) t1);
  u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

  //init data  and poolData
  int* temp = (int*)data;
  for(int i = 0; i < target/sizeof(int); i++){
    //memcpy(temp + (i*tlen),  t1, tlen);
    //This time fill data array with random values;      
    int ranNum = 1 + (rand() % maxRange);
    temp[i] = ranNum;
  }
  memcpy(data + (ranDataIndex * tlen), t1, tlen);

  temp = (int*)pool;
  for(int i = 0; i < (target*secretMultiple)/sizeof(int); i++){
    //memcpy(temp + (i*tlen),  t1, tlen);
    //This time fill data array with random values;
    int ranNum = 1 + (rand() % maxRange);
    temp[i] = ranNum;
  }
  memcpy(pool + (ranCheckIndex*MD_LENGTH), e1, MD_LENGTH);
  //hash the data on cpu
  sdkStartTimer(&timer);
  for(int i = 0; i < target; i++){
    SHA1(data + (i*tlen), tlen, cpuH + (i*MD_LENGTH));
  }
  sdkStopTimer(&timer);
  cpuTimes.hashTime += sdkGetTimerValue(&timer);
  sdkResetTimer(&timer);

  //find hash in pool cpu SIDE
  sdkStartTimer(&timer);
  for(int i = 0; i < target; i++){
    for(int j = 0; j < target * secretMultiple; j++){
      if(0 == memcmpDevice((const void*) (cpuH + (i * MD_LENGTH)), (const void*) (pool + (j*MD_LENGTH)), MD_LENGTH)){
        //cpuResult &= 1 << i;  this does not work on large data sets
        cpuResult = i;
      }
    }
  }
  sdkStopTimer(&timer);
  cpuTimes.findTime += sdkGetTimerValue(&timer);
  sdkResetTimer(&timer);
  cpuTimes.totalTime += cpuTimes.findTime + cpuTimes.hashTime;

  //hash data on GPU and find match
  handleData(data, tlen, target, pool, target*secretMultiple, &gpuResult, &gpuTimes);
  printf("Target size: %d\n", target);
  printf("CPU hash time\t\t[%f ms]\n", cpuTimes.hashTime);
  printf("CPU find time\t\t[%f ms]\n", cpuTimes.findTime);
  printf("CPU total time\t\t[%f ms]\n", cpuTimes.totalTime);
  printf("Result from cpu computation: %d\n", cpuResult);
  printf("GPU malloc time\t\t[%f ms]\n", gpuTimes.mallocTime);
  printf("GPU memcpyHTD time\t[%f ms]\n", gpuTimes.memcpyHTDTime);
  printf("GPU hash time\t\t[%f ms]\n", gpuTimes.hashTime);
  printf("GPU find time\t\t[%f ms]\n", gpuTimes.findTime);
  printf("GPU memcpyDTH time\t[%f ms]\n", gpuTimes.memcpyDTHTime);
  printf("GPU total time\t\t[%f ms]\n", gpuTimes.totalTime);
  printf("Result from gpu computation: %d\n", gpuResult);
  TEST_ASSERT(gpuResult == cpuResult);
  delete[] data;
  delete[] cpuH;
  delete[] gpuH;
  delete[] pool;
  sdkDeleteTimer(&timer);
  cudaDeviceReset();
  TEST_ASSERT( 0 == 0);
}

/*void CudaKernelTestSuite::GPU2dHashFindBench(){
  int target = 1024;
  int secretMultiple = 14; //Thi number is based off a ratio of number of secrets to data points
  int sampleSize = 10;
  int maxRange = 1000;
  int iterations = 1;
  struct btimes tempGpuTimes;
  struct btimes gpuTimes;
  struct btimes cpuTimes;
  struct btimes avgCpuTimes;
  struct btimes avgGpuTimes;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  srand(time(NULL));
  u_char *data, *gpuH, *cpuH, *pool;
  u_int cpuResult, gpuResult;
  printf("Test %s:\n", __PRETTY_FUNCTION__);
  for(int x = 0; x < iterations; x++){ 
    initBtimes(&gpuTimes);
    initBtimes(&cpuTimes);
    initBtimes(&avgGpuTimes);
    initBtimes(&avgCpuTimes);
    initBtimes(&tempGpuTimes); 
    for(int y = 0; y < sampleSize; y++){
      data = new u_char[target * sizeof(int)];
      memset(data, 0, (target* sizeof(int)));
      gpuH = new u_char[target * MD_LENGTH];
      memset(gpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
      cpuH = new u_char[target * MD_LENGTH];
      memset(cpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
      pool = new u_char[secretMultiple*target * MD_LENGTH];
      memset(pool, 0, (secretMultiple*target*MD_LENGTH*sizeof(u_char)));
      cpuResult = 0;
      gpuResult = 0;

      int ranDataIndex =  (rand() % target); //This will be a random index which our hard example t1 will be inserted in our data pool
      int ranCheckIndex =  (rand() % (target*secretMultiple)); //This will be the random index which out hard check e1 will be inserted in our secrets
      u_char t1[] = "abc";
      int tlen = strlen((const char*) t1);
      u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

      //init data  and poolData
      int* temp = (int*)data;
      for(int i = 0; i < target/sizeof(int); i++){
        //memcpy(temp + (i*tlen),  t1, tlen);
        //This time fill data array with random values;      
        int ranNum = 1 + (rand() % maxRange);
        temp[i] = ranNum;
      }
      memcpy(data + (ranDataIndex * tlen), t1, tlen);

      temp = (int*)pool;
      for(int i = 0; i < (target*secretMultiple)/sizeof(int); i++){
        //memcpy(temp + (i*tlen),  t1, tlen);
        //This time fill data array with random values;
        int ranNum = 1 + (rand() % maxRange);
        temp[i] = ranNum;
      }
      memcpy(pool + (ranCheckIndex*MD_LENGTH), e1, MD_LENGTH);
      //hash the data on cpu
      sdkStartTimer(&timer);
      for(int i = 0; i < target; i++){
        SHA1(data + (i*tlen), tlen, cpuH + (i*MD_LENGTH));
      }
      sdkStopTimer(&timer);
      cpuTimes.hashTime += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);

      //find hash in pool cpu SIDE
      sdkStartTimer(&timer);
      for(int i = 0; i < target; i++){
        for(int j = 0; j < target * secretMultiple; j++){
          if(0 == memcmpDevice((const void*) (cpuH + (i * MD_LENGTH)), (const void*) (pool + (j*MD_LENGTH)), MD_LENGTH)){
            //cpuResult &= 1 << i;  this does not work on large data sets
            cpuResult = i;
          }
        }
      }
      sdkStopTimer(&timer); everything with the unit tests done by tomorrow, with initial analytics by 12 tomorrow. I will let you know of the progress well before in case I dont.


      cpuTimes.findTime += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      cpuTimes.totalTime += cpuTimes.findTime + cpuTimes.hashTime;

      //hash data on GPU and find match
      handleData(data, tlen, target, pool, target*secretMultiple, &gpuResult, &tempGpuTimes);
/*
      printf("Test %s:\n", __PRETTY_FUNCTION__);
      printf("Target size: %d\n", target);
      printf("CPU hash time\t\t[%f ms]\n", cpuHashTime);
      printf("CPU find time\t\t[%f ms]\n", cpuFindTime);
      printf("CPU total time\t\t[%f ms]\n", cpuTimes.totalTime);
      printf("Result from cpu computation: %d\n", cpuResult);
      printf("GPU malloc time\t\t[%f ms]\n", gpuTimes.mallocTime);
      printf("GPU memcpyHTD time\t[%f ms]\n", gpuTimes.memcpyHTDTime);
      printf("GPU execution time\t[%f ms]\n", gpuTimes.executionTime);
      printf("GPU memcpyDTH time\t[%f ms]\n", gpuTimes.memcpyDTHTime);
      printf("GPU total time\t\t[%f ms]\n", gpuTimes.totalTime);
      printf("Result from gpu computation: %d\n", gpuResult);

      gpuTimes.totalTime += tempGpuTimes.totalTime;
      gpuTimes.findTime += tempGpuTimes.findTime;
      gpuTimes.hashTime += tempGpuTimes.hashTime;
      gpuTimes.totalTime += tempGpuTimes.totalTime;
      gpuTimes.mallocTime += tempGpuTimes.mallocTime;
      gpuTimes.memcpyDTHTime += tempGpuTimes.memcpyDTHTime;
      gpuTimes.memcpyHTDTime += tempGpuTimes.memcpyHTDTime;
      delete[] data;
      delete[] cpuH;
      deletehttp://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#axzz4B7SuVIht[] gpuH;
      delete[] pool;
    }
    avgCpuTimes.hashTime = cpuTimes.hashTime / sampleSize;
    avgCpuTimes.findTime = cpuTimes.findTime / sampleSize;
    avgCpuTimes.totalTime = avgCpuTimes.hashTime + avgCpuTimes.findTime;
    avgGpuTimes.mallocTime = gpuTimes.mallocTime / sampleSize;
    avgGpuTihttp://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#axzz4B7SuVIhtmes.memcpyHTDTime = gpuTimes.memcpyHTDTime / sampleSize;
    avgGpuTimes.findTime = gpuTimes.findTime / sampleSize;
    avgGpuTimes.hashTime = gpuTimes.hashTime / sampleSize;
    avgGpuTimes.memcpyDTHTime = gpuTimes.memcpyDTHTime / sampleSize;
    avgGpuTimes.totalTime = avgGpuTimes.mallocTime + avgGpuTimes.memcpyDTHTime +
                            avgGpuTimes.memcpyHTDTime + avgGpuTimes.hashTime +
                            avgGpuTimes.findTime;
    printf("=\nTarget size: %d\n", target);
    printf("Sample size: %d\n", sampleSize);
    printf("CPU avg hash time\t[%f ms]\n", avgGpuTimes.hashTime);
    printf("CPU avg find time\t[%f ms]\n", avgCpuTimes.findTime);
    printf("CPU avg total time\t[%f ms]\n", avgCpuTimes.totalTime);
    printf("GPU avg malloc time\t[%f ms]\n", avgGpuTimes.mallocTime);
    printf("GPU avg memcpyHTD time\t[%f ms]\n", avgGpuTimes.memcpyHTDTime);
    printf("GPU avg hash time\t[%f ms]\n", avgGpuTimes.hashTime);
    printf("GPU avg find time\t[%f ms]\n", avgGpuTimes.findTime);
    printf("GPU avg memcpyDTH time\t[%f ms]\n", avgGpuTimes.memcpyDTHTime);
    printf("GPU avg total time\t[%f ms]\n", avgGpuTimes.totalTime);
    printf("GPU is %f x faster\n", avgCpuTimes.totalTime / avgGpuTimes.totalTime);
    target = (target == 1)?256:target<<1;
  }
  sdkDeleteTimer(&timer);
  cudaDeviceReset();
  TEST_ASSERT( 0 == 0);
}*/

void CudaKernelTestSuite::GPU2dHashFindBench(){
  u_int target = 1024;
  int secretMultiple = 14; //This number is based off a ratio of number of secrets to data points
  int sampleSize = 1;
  u_int bValue = 4;
  u_int maxIdentity = 1 << bValue;
  u_int identitiesPerMD = MD_LENGTH / bValue;
  int maxRange = 1000;
  int iterations = 1;
  int checkSize = 0;
  struct btimes tempGpuTimes;
  struct btimes gpuTimes;
  struct btimes cpuTimes;
  struct btimes avgCpuTimes;
  struct btimes avgGpuTimes;
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  srand(time(NULL));
  u_char *data, *gpuH, *cpuH, *pool;
  
  printf("Test %s:\n", __PRETTY_FUNCTION__);
  for(int x = 0; x < iterations; x++){ 
    initBtimes(&gpuTimes);
    initBtimes(&cpuTimes);
    initBtimes(&avgGpuTimes);
    initBtimes(&avgCpuTimes);
    initBtimes(&tempGpuTimes); 
    for(int y = 0; y < sampleSize; y++){
      data = new u_char[target * sizeof(int)];
      memset(data, 0, (target* sizeof(int)));
      gpuH = new u_char[target * MD_LENGTH];
      memset(gpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
      cpuH = new u_char[target * MD_LENGTH];
      memset(cpuH, 0, (target*MD_LENGTH*sizeof(u_char)));
      pool = new u_char[secretMultiple*target * MD_LENGTH];
      memset(pool, 0, (secretMultiple*target*MD_LENGTH*sizeof(u_char)));

      int ranDataIndex =  (rand() % target); //This will be a random index which our hard example t1 will be inserted in our data pool
      int ranCheckIndex =  (rand() % (target*secretMultiple)); //This will be the random index which out hard check e1 will be inserted in our secrets
      u_char t1[] = "abc";
      u_int tlen = strlen((const char*) t1);
      u_char e1[] = "\xa9\x99\x3e\x36\x47\x06\x81\x6a\xba\x3e\x25\x71\x78\x50\xc2\x6c\x9c\xd0\xd8\x9d";

      //init data  and poolData
      u_int* temp = (u_int*)data;
      for(int i = 0; i < target/sizeof(u_int); i++){
        u_int ranNum = 1 + (rand() % maxRange);
        temp[i] = ranNum;
      }

      //insert hard example
      memcpy(data + (ranDataIndex * tlen), t1, tlen);

      temp = (u_int*)pool;
      for(int i = 0; i < (target*secretMultiple)/sizeof(int); i++){
        //memcpy(temp + (i*tlen),  t1, tlen);
        //This time fill data array with random values;
        u_int ranNum = 1 + (rand() % maxRange);
        temp[i] = ranNum;
      }

      //insert hard check
      memcpy(pool + (ranCheckIndex*MD_LENGTH), e1, MD_LENGTH);

      //now fill the indentifiers into the check;
      checkSize = (maxIdentity / (sizeof(int) * 8))+1;
      u_int* check = new u_int[checkSize];
      temp = (u_int*) pool;
      for(int i = 0; i < (target * secretMultiple * INTS_PER_MD); i++){
        u_int l = temp[i];
    	//u_int index = temp[i] / 32;
        //u_int shift = temp[i] % 32;
        //check[index] |= 1 << shift;
    	setBit(check, temp[i]);
      }
       
      //hash the data on cpu
      sdkStartTimer(&timer);
      for(int i = 0; i < target; i++){
        SHA1(data + (i*tlen), tlen, cpuH + (i*MD_LENGTH));
      }
      sdkStopTimer(&timer);
      cpuTimes.hashTime += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);

      //find hash in pool cpu SIDE
      bool found = false;
      sdkStartTimer(&timer);
      u_int* dataIdentifiers = (u_int*) cpuH;
      for(int i = 0; i < (target/sizeof(u_int))*INTS_PER_MD; i++){
    	  u_int x = dataIdentifiers[i];
    	  bool c1 = dataIdentifiers[i] != 0;
    	  bool c2 = dataIdentifiers[i] <= maxIdentity;
    	  bool c3 = c2 && (1 == isSet(check, dataIdentifiers[i]));
        if((dataIdentifiers[i] != 0) && (dataIdentifiers[i] <= maxIdentity) && (isSet(check, dataIdentifiers[i]))){
          found = true;
          printf("found %d\n", dataIdentifiers[i]);
        } 
      }
      sdkStopTimer(&timer);
      cpuTimes.findTime += sdkGetTimerValue(&timer);
      sdkResetTimer(&timer);
      cpuTimes.totalTime += cpuTimes.findTime + cpuTimes.hashTime;

      //hash data on GPU and find match
      u_int* gpuResult = new u_int[checkSize];
      //tlen will be the size of a given data point in bytes
      handleDataV2(data, tlen, target, check, checkSize, bValue, maxIdentity, gpuResult, &tempGpuTimes);
/*
      printf("Test %s:\n", __PRETTY_FUNCTION__);
      printf("Target size: %d\n", target);
      printf("CPU hash time\t\t[%f ms]\n", cpuHashTime);
      printf("CPU find time\t\t[%f ms]\n", cpuFindTime);
      printf("CPU total time\t\t[%f ms]\n", cpuTimes.totalTime);
      printf("Result from cpu computation: %d\n", cpuResult);
      printf("GPU malloc time\t\t[%f ms]\n", gpuTimes.mallocTime);
      printf("GPU memcpyHTD time\t[%f ms]\n", gpuTimes.memcpyHTDTime);
      printf("GPU execution time\t[%f ms]\n", gpuTimes.executionTime);
      printf("GPU memcpyDTH time\t[%f ms]\n", gpuTimes.memcpyDTHTime);
      printf("GPU total time\t\t[%f ms]\n", gpuTimes.totalTime);
      printf("Result from gpu computation: %d\n", gpuResult);
*/
      gpuTimes.totalTime += tempGpuTimes.totalTime;
      gpuTimes.findTime += tempGpuTimes.findTime;
      gpuTimes.hashTime += tempGpuTimes.hashTime;
      gpuTimes.totalTime += tempGpuTimes.totalTime;
      gpuTimes.mallocTime += tempGpuTimes.mallocTime;
      gpuTimes.memcpyDTHTime += tempGpuTimes.memcpyDTHTime;
      gpuTimes.memcpyHTDTime += tempGpuTimes.memcpyHTDTime;
      delete[] data;
      delete[] cpuH;
      delete[] gpuResult;
      delete[] pool;
      delete[] check;
      checkSize = 0;
      maxIdentity = 0;
    }
    avgCpuTimes.hashTime = cpuTimes.hashTime / sampleSize;
    avgCpuTimes.findTime = cpuTimes.findTime / sampleSize;
    avgCpuTimes.totalTime = avgCpuTimes.hashTime + avgCpuTimes.findTime;
    avgGpuTimes.mallocTime = gpuTimes.mallocTime / sampleSize;
    avgGpuTihttp://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#axzz4B7SuVIhtmes.memcpyHTDTime = gpuTimes.memcpyHTDTime / sampleSize;
    avgGpuTimes.findTime = gpuTimes.findTime / sampleSize;
    avgGpuTimes.hashTime = gpuTimes.hashTime / sampleSize;
    avgGpuTimes.memcpyDTHTime = gpuTimes.memcpyDTHTime / sampleSize;
    avgGpuTimes.totalTime = avgGpuTimes.mallocTime + avgGpuTimes.memcpyDTHTime +
                            avgGpuTimes.memcpyHTDTime + avgGpuTimes.hashTime +
                            avgGpuTimes.findTime;
    printf("=\nTarget size: %d\n", target);
    printf("Sample size: %d\n", sampleSize);
    printf("CPU avg hash time\t[%f ms]\n", avgGpuTimes.hashTime);
    printf("CPU avg find time\t[%f ms]\n", avgCpuTimes.findTime);
    printf("CPU avg total time\t[%f ms]\n", avgCpuTimes.totalTime);
    printf("GPU avg malloc time\t[%f ms]\n", avgGpuTimes.mallocTime);
    printf("GPU avg memcpyHTD time\t[%f ms]\n", avgGpuTimes.memcpyHTDTime);
    printf("GPU avg hash time\t[%f ms]\n", avgGpuTimes.hashTime);
    printf("GPU avg find time\t[%f ms]\n", avgGpuTimes.findTime);
    printf("GPU avg memcpyDTH time\t[%f ms]\n", avgGpuTimes.memcpyDTHTime);
    printf("GPU avg total time\t[%f ms]\n", avgGpuTimes.totalTime);
    target = (target == 1)?256:target<<1;
  }
  sdkDeleteTimer(&timer);
  cudaDeviceReset();
  TEST_ASSERT( 0 == 0);
}

void CudaKernelTestSuite::MockAttack(){
  char filename[] = "tvots_gs_packets.pcapng";
  const u_char* packet;
  char* tmp;
  int errBuffSize = 1024;
  struct pcap_pkthdr header;
  char* errBuff = new char[errBuffSize];
  pcap_t* pcapH = pcap_open_offline(filename, errBuff);;
  int pktcnt = 0;
  int secretcnt = 0;
  u_int ret = 0;
  u_char* data = new u_char[DATA_LENGTH * TARGET_NUMBER_PACKETS];
  u_char* secretPool = new u_char[MD_LENGTH * TARGET_NUMBER_PACKETS * (1 + NUMBER_K_SECRETS)];
  TEST_ASSERT(pcapH != NULL);
  
  do{
    packet = pcap_next(pcapH, &header);
    if(!packet || header.len != TARGET_PACKET_LENGTH) continue;
    //check fields of the new packet to the original.
    memcpy(data + (pktcnt*DATA_LENGTH), packet + DATA_POSITION, DATA_LENGTH); 
    
    memcpy(secretPool + (secretcnt * MD_LENGTH), packet + SALT_SECRET_POSITION, MD_LENGTH);
    secretcnt++;
    for(int i = 0; i < NUMBER_K_SECRETS; i++){
      memcpy(secretPool + (secretcnt * MD_LENGTH), packet + (K_SECRET_POSITION*i), MD_LENGTH);
      secretcnt++;
    }
    pktcnt++;
  }while(packet && pktcnt < TARGET_NUMBER_PACKETS);
 
  handleData(data, DATA_LENGTH, pktcnt, secretPool, secretcnt, &ret, NULL); 
  printf("%s\n", __PRETTY_FUNCTION__);
  printf("result: %d\n", ret);
  printf("pktcnt: %d, secretcnt: %d\n", pktcnt, secretcnt);
  delete[] data;
  delete[] secretPool;
  delete[] errBuff;

} 

void CudaKernelTestSuite::GetIdentifierTest(){
	const u_int dummyInt = 2863311530;
	//this int is for the pattern "10101010..1010" repeating to 32 bits
	u_char* buff = new u_char[MD_LENGTH];
	//conversion is to be consistent with real data
	int bValue, nth, ret, expected;
	u_int* temp = (u_int*) buff;
	for(int i = 0; i < MD_LENGTH / sizeof(u_int); i++){
		temp[i] = dummyInt;
	}
/*
 * | 10101010101010101010101010101010 | x5
 * | 10101010101010101010101010101010 | 10101010101010101010101010101010 |...
 */
	//this test will go against different bValues and different identifiers
	//in of range
	bValue = 7;
	nth = 15;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = 42;
	TEST_ASSERT(ret == expected);

	bValue = 10;
	nth = 3;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = 682;
	TEST_ASSERT(ret == expected);

	bValue = 12;
	nth = 5;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = 2730;
	TEST_ASSERT(ret == expected);

	//out of range
	bValue = 50;
	nth = 7;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	TEST_ASSERT(ret < 0); //this test has an integer overflow

	bValue = 1000;
	nth = 0;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = -1;
	TEST_ASSERT(ret == expected);

	bValue = 12;
	nth = 13;
	ret = getNthIdentifier(temp, nth, bValue, MD_LENGTH/sizeof(u_int));
	expected = -1;
	TEST_ASSERT(ret == expected);

	delete[] buff;
	TEST_ASSERT(0==0);
}

int main(void){
  CudaKernelTestSuite cts;
  Test::TextOutput output(Test::TextOutput::Verbose);
  cts.run(output);
  cudaDeviceReset();
  return 0;
}
