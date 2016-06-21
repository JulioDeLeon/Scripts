#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

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
  for (z = pow(2,31); z > 0; z >>= 1)
  {
    strcat(b, ((x & z) == z) ? "1" : "0");
  }

  return b;
}


int main(void){
  unsigned char* buff = new unsigned char[MD_LENGTH];
  memset(buff, 0, MD_LENGTH);
  unsigned int bValue = 5;
  unsigned int maxIdentity = pow(2, bValue+1) - 1;
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
    printf("|%s",s);
  }
  printf("\n");
   
  delete[] buff; 
  return 0;
}

unsigned int getNthIdentifier(unsigned int *buff, int nth, int bValue, int size){
  unsigned int ret = 0;
  unsigned int max = pow(2, bValue+1) - 1;
  unsigned int taken = 0;
  unsigned int left = bValue;  
  unsigned int *temp = buff;
  unsigned int uiIndex = 0;
  unsigned int bitIndex = nth * bValue; 
  unsigned int bitEnd = bitIndex + bValue;
  //get the int closest to the first bit we need
  //if need get the rest of the bits 
   
  return ret;
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

