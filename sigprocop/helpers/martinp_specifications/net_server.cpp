/*
 * net_server.cpp
 *
 *  Created on: Sep 4, 2014
 *      Author: jarda
 */

#include "net_server.h"
#include "../legacy/simple_log.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


NetServer::NetServer(std::string s, int o, int port, Processor * processor):ActiveObject(s,o)
{
  _processor = processor;
	_sockfd = 0;
	_port = port;
	_testing = false;
}

NetServer::~NetServer()
{
	close_all_sockets();
}

int NetServer::open_listen_socket()
{
	struct sockaddr_in serv_addr;
	int portno;
	int res;
	int optval;
	socklen_t optlen;

	_sockfd = socket(AF_INET, SOCK_STREAM, 0);
	//int one = 1;
	//setsockopt(_sockfd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(int));

	if (_sockfd < 0)
	{
		LOG(LOG_ERROR,"Socket creation failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
		return _sockfd;
	}

	optval = 1;
	res = setsockopt(_sockfd,SOL_SOCKET,SO_REUSEADDR, &optval, sizeof(optval));
	if (res<0) {
		LOG(LOG_ERROR,"setsockopt() failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
	}

	res = getsockopt(_sockfd,SOL_SOCKET,SO_REUSEADDR, (void*) &optval, &optlen);
	if (res<0) {
		LOG(LOG_ERROR,"getsockopt() failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
	} else {
		LOG(LOG_INFO,"getsockopt() result " + to_string_const(optval) );
	}


	memset(&serv_addr, sizeof(serv_addr), 0);
	portno = _port;
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = INADDR_ANY;
	serv_addr.sin_port = htons(portno);

	res = bind(_sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr) );

	if (res < 0)
	{
		LOG(LOG_ERROR,"Socket binding failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
		return res;
	}

	res = listen(_sockfd,1);

	if (res < 0)
	{
		LOG(LOG_ERROR,"Listening on socket failed " + to_string_const(errno) + " "+ std::string(strerror(errno)));
		return res;
	}

	LOG(LOG_INFO,"listening on port: "+to_string(_port));

	/* moved to start_measurement() so it is executed in SpectrumReader thread context
	struct sockaddr_in cli_addr;
	socklen_t clilen;
	LOG(LOG_INFO,"Test specrometer is waiting for connection");
	_datasockfd = accept(_sockfd, (struct sockaddr *) &cli_addr, &clilen);
	LOG(LOG_INFO,"Test specrometer is connected");
	_current_bulk = 0;
	*/

	return 0;
}

int NetServer::close_all_sockets()
{
	int res;
	if (_sockfd != 0) {
		LOG(LOG_INFO,"closing descriptors");
		res = close(_sockfd);
		if (res<0) {
			LOG(LOG_ERROR,"close(_sockfd) failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
		}
		for(int i=0; i<_datasockfds.size(); i++ ) {
			close_connection(_datasockfds[i], i);
		}
		_datasockfds.clear();
	}
	//sleep(10);
	return 0;
}

int NetServer::accept_connection()
{
	int fd;
	int res;
	struct sockaddr_in cli_addr;
	socklen_t clilen;
	int optval;
	socklen_t optlen;
	//------------------- accept pending connection -------------------------------------------------

	clilen = sizeof(cli_addr);
	fd = accept(_sockfd, (struct sockaddr *) &cli_addr, &clilen);
	if (fd < 0) {
		LOG(LOG_ERROR,"accept_connection() accept() failed " + to_string_const(errno) +" "+ std::string(strerror(errno))+" ");
	}

	res = getsockopt(fd,SOL_SOCKET,SO_REUSEADDR, (void*) &optval, &optlen);
	if (res<0) {
		LOG(LOG_ERROR,"accept_connection() getsockopt(_datasockfd) failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
	} else {
		LOG(LOG_INFO,"accept_connection() getsockopt(_datasockfd) result " + to_string_const(optval) );
	}

	//-------------- determine IP address of incoming connection -------------------------------------
	char buffer[INET_ADDRSTRLEN];
	std::string ip_address;

	if (NULL!= inet_ntop(AF_INET, &(cli_addr.sin_addr), buffer, INET_ADDRSTRLEN) ) {
		ip_address = std::string(buffer);
	} else {
		LOG(LOG_WARNING,"accept_connection() inet_ntop() failed");
	}

	//------------- store connection info ------------------------------------------------------------
	connection_descriptor desc;

	desc.fd = fd;
	desc.bytes_read = 0;
	desc.receive_status = REC_STATUS_SYNC_0;
	desc.expected_packet_size = 0;
	desc.ip_address = ip_address;
	desc.packet_sequence_counter = 0;

	_datasockfds.push_back(desc);
	LOG(LOG_INFO,"new connection accepted: "+ip_address+" port: "+to_string_const(cli_addr.sin_port) );

	return 0;
}

int NetServer::close_connection( connection_descriptor &connection, int index)
{
	int res;
	char buffer[ 10*sizeof(uplink_packet_network) ];

	if (index < _datasockfds.size() ) {
		res = shutdown(_datasockfds[index].fd, SHUT_RDWR);
		if (res<0) {
			LOG(LOG_ERROR,"shutdown(_datasockfd) failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
		}

		res = recv(_datasockfds[index].fd, buffer, 10, 0);
		while( res != 0) {
			  LOG(LOG_INFO,"waiting for shutdown res: "+to_string(res));
			  sleep(1);
			  res = recv(_datasockfds[index].fd, buffer, 10*sizeof(uplink_packet_network), 0);
		}

		res = close(_datasockfds[index].fd);
		if (res<0) {
			LOG(LOG_ERROR,"close(_datasockfd) failed " + to_string_const(errno) +" "+ std::string(strerror(errno)) );
		}
		 LOG(LOG_INFO,"socket closed fd: "+to_string(_datasockfds[index].fd));

		_datasockfds.erase(_datasockfds.begin()+index);

		return 0;
	} else {
		return -1;
	}
}

int NetServer::read_data_from_connection(connection_descriptor &connection, int index)
{
	int received;
	int pointer = 0;
	int res;
	char buffer[ 10*sizeof(uplink_packet_network) ];
	// there are pending data on this connection -> read all available data into temporary buffer
	received = read(connection.fd, &buffer[0], 8*sizeof(uplink_packet_network) );
	if (_testing) {
		LOG(LOG_INFO,"received data size: "+to_string(received));
	}

	if (received == 0) { // connection closed by remote client
		LOG(LOG_INFO,"connection from "+connection.ip_address+" closed");
		close_connection(connection, index);
		return -1;
	}

	if (received < 0) { // connection error
		//TODO check for noncritical error EAGAIN etc and do not close connection on these errors
		if (errno == EAGAIN) {
			LOG(LOG_WARNING,"connection from "+connection.ip_address+" non-critical error : "+ to_string_const(errno) +" "+ std::string(strerror(errno)) );
		} else {
			LOG(LOG_WARNING,"connection from "+connection.ip_address+" closing connection after error : "+ to_string_const(errno) +" "+ std::string(strerror(errno)) );
			close_connection(connection, index);
			return -2;
		}
	}

	while(received > 0) {
		switch(connection.receive_status) {
			case(REC_STATUS_SYNC_0): {
				if (buffer[pointer] == 0x55) {
					connection.packet_buffer.header.sync[0] = 0x55;
					connection.bytes_read++;
					connection.receive_status = REC_STATUS_SYNC_1;
				}
				received--;
				pointer++;
				break;
			}
			case(REC_STATUS_SYNC_1): {
				if (buffer[pointer] == 0x00) {
					connection.packet_buffer.header.sync[1] = 0x00;
					connection.bytes_read++;
					connection.receive_status = REC_STATUS_SYNC_2;
					received--;
					pointer++;
				} else {
					connection.bytes_read = 0;
					connection.expected_packet_size = 0;
					connection.receive_status = REC_STATUS_SYNC_0;
				}
				break;
			}
			case(REC_STATUS_SYNC_2): {
				if (buffer[pointer] == 0x55) {
					connection.packet_buffer.header.sync[2] = 0x55;
					connection.bytes_read++;
					connection.receive_status = REC_STATUS_HEADER;
					received--;
					pointer++;
				} else {
					connection.bytes_read = 0;
					connection.expected_packet_size = 0;
					connection.receive_status = REC_STATUS_SYNC_0;
				}
				break;
			}
			case(REC_STATUS_HEADER): { // sync already received -> wait for complete header
				int bytes_to_copy = MIN( sizeof(uplink_packet_header)-connection.bytes_read , received );
				memcpy( (char*)&connection.packet_buffer+connection.bytes_read, &buffer[pointer], bytes_to_copy );
				connection.bytes_read += bytes_to_copy;
				received -= bytes_to_copy;
				pointer += bytes_to_copy;

				if (connection.bytes_read >= sizeof(uplink_packet_header)) {
					if (connection.bytes_read > sizeof(uplink_packet_header)) { // this should never happen
							LOG(LOG_ERROR,"more than sizeof(uplink_packet_header). Should never happen");
					}
					bool checksum = connection.packet_buffer.check_header_checksum();
					if (checksum) {
						connection.expected_packet_size = connection.packet_buffer.header.packet_byte_size; // actual size of packet
						connection.receive_status = REC_STATUS_REST;
					} else { // header is invalid-> reset receiver logic and reinsert header data to receive buffer without initial byte of sync sequence
						LOG(LOG_ERROR,"packet header checksum failed");
						connection.bytes_read = 0;
						connection.expected_packet_size = 0;
						connection.receive_status = REC_STATUS_SYNC_0;
						//------ reinsert data -----------------------------------------------
						int bytes_to_insert = sizeof(uplink_packet_header)-1;
						//----- move data forward to create space ------------------------------------
						memmove(&buffer[bytes_to_insert], &buffer[pointer], received);
						//----- copy header --------------------------------------------------
						memcpy(&buffer[0], (char*)&connection.packet_buffer+1, bytes_to_insert);
						received += bytes_to_insert;
						pointer = 0;
					}
				}
				break;
			}
			case(REC_STATUS_REST): {  // header already received -> copy rest of data
				int bytes_to_copy = MIN( connection.expected_packet_size-connection.bytes_read , received );
				memcpy( (char*)&connection.packet_buffer+connection.bytes_read, &buffer[pointer], bytes_to_copy );
				connection.bytes_read += bytes_to_copy;
				received -= bytes_to_copy;
				pointer += bytes_to_copy;
				if (connection.bytes_read >= connection.expected_packet_size) { // whole packet received -> proceed
					if (connection.bytes_read > connection.expected_packet_size) { // this should never happen
						LOG(LOG_ERROR,"more than sizeof(uplink_packet_network). Should never happen");
					}
					//---- check if checksum is OK ----------------------------
					bool checksum = connection.packet_buffer.check_checksum();
					if (checksum) {
						if (_testing) {
							LOG(LOG_INFO,"connection: "+to_string(index)+" packet received ok "+to_string_const(connection.packet_buffer.header.packet_sequence_counter)+" checksum: "+to_string(checksum));
						}
						res = proceed_packet(connection, index);
					} else { // packet not valid -> reinsert packet data to receive buffer without initial byte of sync sequence
						LOG(LOG_WARNING,"packet received checksum failed "+to_string_const(connection.packet_buffer.header.packet_sequence_counter)+" checksum: "+to_string(checksum));
						//------ reinsert data -----------------------------------------------
						int bytes_to_insert = connection.expected_packet_size-1;
						//----- move data to create space ------------------------------------
						memmove(&buffer[bytes_to_insert], &buffer[pointer], received);
						//----- copy header --------------------------------------------------
						memcpy(&buffer[0], (char*)&connection.packet_buffer+1, bytes_to_insert);
						received += bytes_to_insert;
						pointer = 0;
					}
					//----- reset packet buffer ----------------------------------
					connection.bytes_read = 0;
					connection.expected_packet_size = 0;
					connection.receive_status = REC_STATUS_SYNC_0;
				}
				break;
			}
		}
	}
	return 0;
}

int NetServer::proceed_packet(connection_descriptor &connection, int index)
{
		//----- check packet sequence count to detect lost packets -------------
		if (connection.packet_sequence_counter != connection.packet_buffer.header.packet_sequence_counter) {
			LOG(LOG_WARNING,"packet lost. expected sequence number "+to_string_const(connection.packet_sequence_counter)+" received: "+to_string_const(connection.packet_buffer.header.packet_sequence_counter));
			connection.packet_sequence_counter = connection.packet_buffer.header.packet_sequence_counter;
		}
		connection.packet_sequence_counter++;
		// ---- if in testing mode store packet to buffer ----------------------
		if (_testing) {
			connection.test_buffer.push_back(connection.packet_buffer);
		}

		//---- process packet data here ----------------------------------------
		if (_processor){
		  _processor->pushInputPacket(connection.packet_buffer);
		}

		return 0;
}

void NetServer::Main()
{
	struct sockaddr_in cli_addr;
	socklen_t clilen;
	int optval;
	socklen_t optlen;
	int res;
	fd_set socket_fds;
    struct timeval select_timeout;
    int max_socket;

	res = open_listen_socket();
	if (res != 0) {
		LOG(LOG_ERROR,"Main() open_listen_socket() failed. res = "+to_string(res));
	}



	while(_thread_run) {
		max_socket = _sockfd;
		FD_ZERO(&socket_fds);
		FD_SET(_sockfd,&socket_fds); // add listen socket
		for(int i=0; i< _datasockfds.size(); i++) {  // add all active connections
			FD_SET(_datasockfds[i].fd,&socket_fds);
			if (_datasockfds[i].fd > max_socket) {
				max_socket = _datasockfds[i].fd;
			}
		}
		select_timeout.tv_sec = 0;		// 500 msec select timeout
		select_timeout.tv_usec = 500000;
		//---------------- wait for connection using select(), -------------------------------------------------------------
		res = select(max_socket+1, &socket_fds, NULL, NULL, &select_timeout);
		if (res == -1) { // select() error
			LOG(LOG_ERROR,"Main() select() failed "+to_string(errno));
		} else {
			if (res == 0) { // timeout
				if (_testing) {
					LOG(LOG_INFO,"Main() select() timed out");
				}
			} else { // connection or data pending
				if (_testing) {
					LOG(LOG_INFO,"Main() select() connection or data pending");
				}
				for(int i=0; i<_datasockfds.size(); i++) {  // check all active connections
					if (FD_ISSET(_datasockfds[i].fd,&socket_fds)) {
						read_data_from_connection(_datasockfds[i], i);
					}
				}
				if (FD_ISSET(_sockfd,&socket_fds)) {
					accept_connection();
				}

			}
		}
	}

}

connection_descriptor* NetServer::get_connection(int index)
{
	if (index < _datasockfds.size() ) {
		return &_datasockfds[index];
	} else {
		return NULL;
	}
}

void NetServer::before_start_thread()  // called in same context as start_thread()
{
	/*nothing*/
}

void NetServer::before_join_thread()   // called in same context as stop_thread()
{
	/*nothing*/
}

