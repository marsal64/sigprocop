/*
 * net_server.h
 *
 *  Created on: Sep 4, 2014
 *      Author: jarda
 */

#ifndef NET_SERVER_H_
#define NET_SERVER_H_

#include <string>
#include <vector>

#include "../processor.h"
#include "../legacy/active.h"

#define UPLINK_VECTOR_MAX_DATA_COUNT		1024
#define UPLINK_MAX_READOUT_COUNT_PER_PACKET	1024

#define UPLINK_PACKET_TYPE_SINGLE_READOUT	0
#define UPLINK_PACKET_TYPE_VECTOR_READOUT	1

/// structure to hold single value readout with timestamp
typedef struct uplink_single_readout_tag {
	/// timestamp
	struct timeval timestamp;
	/// data value
	double value;
}__attribute__ ((packed)) uplink_single_readout;

/// structure to hold vector value readout with timestamp
typedef struct uplink_vector_readout_tag {
	/// timestamp
	struct timeval timestamp;
	/// size of vector. maximum size is UPLINK_VECTOR_MAX_DATA_COUNT
	uint32_t size;
	/// vector data. maximu size is UPLINK_VECTOR_MAX_DATA_COUNT
	double value[UPLINK_VECTOR_MAX_DATA_COUNT];
}__attribute__ ((packed)) uplink_vector_readout;

/// header of network packet
typedef struct uplink_packet_header_tag {
	/// sync bytes must be 0x55 0x00 0x55
	uint8_t sync[3];			// 0x55 0x00 0x55
	/// packet type. 0 single value readouts, 1 vector readouts
	uint8_t packet_type;
	/// string identification of source device
	uint8_t device_id[32];
	/// string identification of sensor
	uint8_t sensor_id[32];
	/// sequence number to identify lost packets
	uint16_t packet_sequence_counter;
	/// number of readouts (either single or vector) in this packet
	uint16_t packet_readout_count;
	/// total packet size in bytes
	uint32_t packet_byte_size;
	/// packet header checksum
	uint32_t header_checksum;
}__attribute__ ((packed)) uplink_packet_header;

/// complete network packet
typedef struct uplink_packet_network_tag {
	/// header
	uplink_packet_header header;
	/// data
	uplink_single_readout data[UPLINK_MAX_READOUT_COUNT_PER_PACKET];
	/// checksum
	uint32_t checksum;	// position of checksum is different dependent on size of the packet, this position is for maximum size packet
	/**
	 * computes checksum of packet and stores it to appropriate position
	 */
	void compute_checksum() {
		uint32_t *p = (uint32_t *)this;
		uint32_t count = ( this->header.packet_byte_size )/4 - 1;
		uint32_t sum = 0;
		while(count >0) {
			sum += *p++;
			count--;
		}
		*((uint32_t *)p) = sum;
	}
	/**
	 * computes checksum of packet and compares it to stored checksum
	 * @return true checksum ok, false checksum error
	 */
	bool check_checksum() {
		uint32_t *p = (uint32_t *)this;
		uint32_t count = ( this->header.packet_byte_size )/4 - 1;
		uint32_t sum = 0;
		while(count >0) {
			sum += *p++;
			count--;
		}
		return (*((uint32_t *)p) == sum);
	}
	/**
	 * computes checksum of packet header and stores it to appropriate position
	 */
	void compute_header_checksum() {
		uint32_t *p = (uint32_t *)&this->header;
		uint32_t count = ( sizeof(this->header) )/4 - 1;
		uint32_t sum = 0;
		while(count >0) {
			sum += *p++;
			count--;
		}
		*((uint32_t *)p) = sum;
	}
	/**
	 * computes checksum of packet and compares it to stored checksum
	 * @return true checksum ok, false checksum error
	 */
	bool check_header_checksum() {
		uint32_t *p = (uint32_t *)&this->header;
		uint32_t count = ( sizeof(this->header) )/4 - 1;
		uint32_t sum = 0;
		while(count >0) {
			sum += *p++;
			count--;
		}
		return (*((uint32_t *)p) == sum);
	}
}__attribute__ ((packed)) uplink_packet_network;

#define	REC_STATUS_SYNC_0	0
#define	REC_STATUS_SYNC_1	1
#define	REC_STATUS_SYNC_2	2
#define	REC_STATUS_HEADER	3
#define	REC_STATUS_REST		4

/// holds information on opened client connection
typedef struct connection_descriptor_tag {
	/// socket file descritor
	int fd;
 	/// ip address from which connection initiated
	std::string ip_address;
	/// buffer to store received data
	uplink_packet_network packet_buffer;
	/// bytes allready received
	int bytes_read;
	/// total size of received packat taken from packet_byte_size field of struct uplink_packet_network
	int expected_packet_size;
	/**
	 * state of receiver logic defined as follows
	 * #define	REC_STATUS_SYNC_0	0	waiting for first sync byte
	 * #define	REC_STATUS_SYNC_1	1   waiting for second sync byte
	 * #define	REC_STATUS_SYNC_2	2   waiting for third sync byte
	 * #define	REC_STATUS_HEADER	3   waiting for header so we already know size of packet
	 * #define	REC_STATUS_REST		4   receive rest of packet data and process packet
	 */
	int receive_status;
	/// sequence number to identify lost packets
	uint16_t packet_sequence_counter;
	/// packet buffer for testing only
	std::vector<uplink_packet_network> test_buffer;
} connection_descriptor;

class Processor;

class NetServer: public ActiveObject
{
private:
  ///object for processing received packets
  Processor * _processor;
	/// number of port to listen to
	int _port;

	/// handle to listening socket
	int _sockfd;

	/// vector holding information about active connections
	std::vector<connection_descriptor> _datasockfds;

	/// if true all packets are stored to buffer for comparison with send data
	bool _testing;

	/**
	 * opens the socket and starts to listen on it
	 * @return 0 success, -1 error
	 */
	int open_listen_socket();
	/**
	 * closes all connection sockets and listening socket
	 * @return 0 success, -1 error
	 */
	int close_all_sockets();
	/**
	 * accepts connection and creates new connection socket
	 * @return
	 */
	int accept_connection();

	/**
	 * closes connection socket
	 * @param connection connection information to close
	 * @param index index of connection in _datasocksfds vector to close
	 * @return 0 ok, -1 index out of range
	 */
	int close_connection( connection_descriptor &connection, int index);

	/**
	 * read all available data from connection socket and process them
	 * @param connection connection information
	 * @param index index of connection in _datasocksfds vector
	 * @return 0 success, -1 connection closed, -2 connection error
	 */
	int read_data_from_connection( connection_descriptor &connection, int index);

	/**
	 * processes single packet read from any connection
	 * @param connection connection information
	 * @param index index of connection in _datasocksfds vector
	 * @return  0 if packet processed ok
	 *         -1 when packet processing failed
	 */
	int proceed_packet(connection_descriptor &connection, int index);
public:
	/**
	 * constructor
	 * @param s string identifier of instance
	 * @param o integer identifier of instance
	 * @param port number of port to listen on
   * @param processor processor which will process the received packets (can be NULL)
	 */
	NetServer(std::string s, int o, int port, Processor * processor = NULL);

	/**
	 * destructor
	 */
	~NetServer();

	/**
	 * thread body function
	 * opens listen socket and starts to listen. Accepts new connection and processes all data received on all connections
	 */
	virtual void Main();

	/**
	 *
	 */
	virtual void before_start_thread();  // called in same context as start_thread()
	/**
	 *
	 */
	virtual void before_join_thread();   // called in same context as stop_thread()
	/**
	 * enables testing functions
	 * if enabled all packets are stored in buffer for comparison with send packets
	 */
	void enable_testing() {_testing = true;};
	/**
	 * disables testing functions
	 * if enabled all packets are stored in buffer for comparison with send packets
	 */
	void disable_testing() {_testing = false;};

	/**
	 * get connection descriptor
	 * for testing purposes only. test can get received data and compare them with original data
	 */
	connection_descriptor *get_connection(int index);
};



#endif /* NET_SERVER_H_ */
