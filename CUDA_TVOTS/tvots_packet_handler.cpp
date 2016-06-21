#include "supplement.h"

void
printPacketInformation(struct pcap_pkthdr *header, const u_char *packetData){
  printf("Packet size %d\n", header->caplen);
  for (int i = 0; i < header->caplen; i++) {
    printf("%04x", packetData[i]);
  }
  printf("\n");
}

TVOTS_Packet::TVOTS_Packet( const u_char* packet){
  //data
  this->event_data = new u_char[DATA_LENGTH];
  memcpy((char*)this->event_data, packet + DATA_POSITION, DATA_LENGTH);
  //timestamp
  this->time_stamp = new u_char[TIMESTAMP_LENGTH];
  memcpy(this->time_stamp, packet + TIMESTAMP_POSITION, TIMESTAMP_LENGTH);

  //seq num
  this->sequence_number = new u_char[SEQUENCE_NUMBER_LENGTH];
  memcpy(this->sequence_number, packet + SEQUENCE_NUMBER_POSITION, SEQUENCE_NUMBER_LENGTH);

  //epoch
  this->epoch_ID = new u_char[EPOCH_ID_LENGTH];
  memcpy(this->epoch_ID, packet + EPOCH_ID_POSITION, EPOCH_ID_LENGTH);

  this->salt_secret = new u_char[SALT_SECRET_LENGTH];
  memcpy(this->salt_secret, packet + SALT_SECRET_POSITION, SALT_SECRET_LENGTH);
  //salt secrets
  this->k_secrets = new u_char*[NUMBER_K_SECRETS];
  for(int i = 0; i < NUMBER_K_SECRETS; i++){
    this->k_secrets[i] = new u_char[K_SECRET_LENGTH];
    memcpy(this->k_secrets[i], packet + K_SECRET_POSITION + ( i * K_SECRET_LENGTH), K_SECRET_LENGTH);
  }
}

TVOTS_Packet::~TVOTS_Packet(){
  delete[] this->event_data;
  delete[] this->time_stamp;
  delete[] this->salt_secret;
  delete[] this->epoch_ID;
  //delete the k secrete;
  for(int i = 0; i < NUMBER_K_SECRETS; i++){
    delete[] this->k_secrets[i];
  }
  delete[] this->k_secrets;
}

u_char* TVOTS_Packet::getEventData(){
  return this->event_data;
}

u_char* TVOTS_Packet::getTimeStamp(){
  return this->time_stamp;
}

u_char* TVOTS_Packet::getSequenceNumber(){
  return this->sequence_number;
}

u_char* TVOTS_Packet::getEpochID(){
  return this->epoch_ID;
}

u_char* TVOTS_Packet::getSaltSecret(){
  return this->salt_secret;
}

u_char** TVOTS_Packet::getKSecrets(){
  return this->k_secrets;
}

int* hashMSG(u_char* msg, int len){
  u_char*src =  msg;
  u_char*dst = (u_char*) malloc(sizeof(u_char) * len);
  SHA1(src, len, dst);
  return (int*) dst;
}


