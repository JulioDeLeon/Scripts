#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <bitset>
using namespace std;
#define MD_LENGTH  20
void setBit(int* x, int val);
void clearBit(int* x, int val);
int isSet(int* x, int val);
void toggleBit(int* x, int val);
unsigned int getNthIdentifier(unsigned int *buff, int nth, int bValue, int size);

const char *byte_to_binary(int x)
{
  static char b[32];
  b[0] = '\0';

  unsigned int z;
  for (z = 1<<31; z > 0; z >>= 1)
  {
    strcat(b, ((x & z) == z) ? "1" : "0");
  }

  return b;
}


int main(void){
  unsigned char* buff = new unsigned char[MD_LENGTH];
  memset(buff, 0, MD_LENGTH);
  unsigned int bValue = 5;
  unsigned int maxIdentity = (1 << (bValue +1)) - 1;
  srand(time(NULL));
  
  //fill the array with random data
  //print out every bit in the array
  /*
    I dont need this code but it helps print of the thinds i do need
  */
  unsigned int* temp = (unsigned int*) buff;
  for(int i = 0; i < MD_LENGTH / sizeof(unsigned int); i++){
    //temp[i] =  pow(2,31) + (pow(2,31) - 1);
    temp[i] = rand();
    const char* s = byte_to_binary(temp[i]);
    printf("|%s\n",s);
  }
  printf("\n");
  int b = 7;
  int id = 15;
  unsigned int x = getNthIdentifier(temp, id, b, 5);
  printf("ident:%d\n", x); 
 /* for(int y = 0; y <= 20; y++){
    unsigned int x = getNthIdentifier(temp, y, b, 5);
    printf("ident:%d\n", x); 
  }*/
  delete[] buff; 
  return 0;
}

/*
 * @param buff: the integer array which will be examined
 * @param nth: the index of the Nth identifer that can be produced, starts at 0
 * @param bVale: the number of bits which can construct an identifer
 * @param length: the number of ints in the buffer
 */
unsigned int getNthIdentifier(unsigned int *buff, int nth, int bValue, int length){
  unsigned int maxIdentity = (1 << (bValue + 1)) - 1;
  unsigned int *temp = buff;
  unsigned int bytesPerInt = 4;
  unsigned int bitsPerByte = 8;
  unsigned int maxBit = length*bytesPerInt*bitsPerByte; //maximum number of bits that could be in the buff
  unsigned int maxNumberIdenties = maxBit / bValue;
  unsigned int nthStartBit = nth * bValue;
  unsigned int nthEndBit = nthStartBit + bValue - 1;
  unsigned int startIntIndex = nthStartBit / 31; //this will be the unsigned
                                               //int which will contain our first bit.
  unsigned int startIntRangeBeg = startIntIndex * 32;
  unsigned int startIntRangeEnd = (startIntIndex+1) * 32 - 1; 
  unsigned int accum = 0;
  unsigned int exp = bValue -1;
  unsigned int startIntBitIndex = nthStartBit - startIntRangeBeg;
  printf("params: nth: %d bv: %d len: %d\n", nth, bValue, length); 
  printf("pre:\tstartIntRange: %d - %d\n", startIntRangeBeg, startIntRangeEnd); 
  printf("\tnth Bit range: %d - %d\n", nthStartBit, nthEndBit);
  printf("\tmax Id: %d max bit: %d max Ids: %d\n", maxIdentity, maxBit, maxNumberIdenties);
  printf("\tmax bit index: %d max Ids index: %d\n", maxBit - 1, maxNumberIdenties -1);
  printf("\tStart Int index: %d\n", startIntIndex);
   
  if(nth > maxNumberIdenties || nth < 0 || 
    nthEndBit > maxBit || nthStartBit < 0){ return -1;}
  //get the int closest to the first bit we need
  //if need get the rest of the bits 
  for(int i = exp; i >= 0; i--){
    //if(startIntIndex > maxNumberIdenties) return -1;
    //start accumulating the identifer 
    printf("status: accum: %d exp: %d bit: %d nthStartBit: %d startIntIndex: %d startIntBitIndex: %d\n",
      accum, i, (temp[startIntIndex] >> (31 - startIntBitIndex)) & 1, nthStartBit, startIntIndex, startIntBitIndex);
    string binary = bitset<32>(temp[startIntIndex]).to_string();
    printf("<%d><%s>\n", temp[startIntIndex], binary.c_str());
    accum += ((temp[startIntIndex] >> (31 - startIntBitIndex)) & 1) << i; 
    
    //check if still in range
    //iterate to next bit
    if((startIntBitIndex + 1) > 31){
     printf("found border\n");
      startIntIndex++;
      startIntBitIndex = 0;
    } else {
      printf("in range\n");
      startIntBitIndex++;
    }
  }
  printf("\n");
  return accum;
}

void setBit(int* x, int val){
  int index = val / 32;
  int shift = val % 32;
  x[index] |= 1 << shift;
}

void clearBit(int* x, int val) {
  int index = val / 32;
  int shift = val % 32;
  x[index] &= ~(1 << shift);
 
}

void toggleBit(int* x, int val){ 
  int index = val / 32;
  int shift = val % 32;
  x[index] ^= 1 << shift;
}

int isSet(int* x, int val){
   int index = val / 32;
  int shift = val % 32;
  return (x[index] >> shift) & 1;
}

