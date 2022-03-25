/*
 * uplink.cpp
 *
 *  Created on: Jul 23, 2014
 *      Author: jarda
 */
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>

#include "uplink.h"
#include "../net/net_server_uplink.h"

UplinkSender::UplinkSender(std::string s, int o, uint32_t ip_address, int port_number, int buffer_size) :
		ActiveObject(s, o), _buffer(s + "_B", o, buffer_size)
{
	_ip_address = ip_address;
	_port_number = port_number;
	_packet_counter = 0;
	_connected = false;
	_socket = 0;

	gettimeofday(&_last_lost_data_check, NULL);
}

UplinkSender::~UplinkSender()
{
}

// get sockaddr, IPv4 or IPv6:
void *get_addr(struct sockaddr *sa)
		{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*) sa)->sin_addr);
	}

	return &(((struct sockaddr_in6*) sa)->sin6_addr);
}

int UplinkSender::_establish_connection()
{
	struct addrinfo hints;
	struct addrinfo *server_address;
	struct addrinfo *p;
	std::string ip_address_string = NetServerUplink::ipIntToString(_ip_address);
	int sockfd;
	char server_address_string[INET6_ADDRSTRLEN];
	int res;

	if (_connected)
		_terminate_connection();

	LOG(LOG_INFO, "Establishing connection to server: " + ip_address_string + " port: " + std::to_string(_port_number));

	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	res = getaddrinfo(ip_address_string.c_str(), std::to_string(_port_number).c_str(), &hints, &server_address);
	if (res != 0) {
		LOG(LOG_ERROR, "getaddrinfo() failed for server: " + ip_address_string + " port: " + std::to_string(_port_number));
		return -1;
	}

	// loop through all the results and connect to the first we can
	for (p = server_address; p != NULL; p = p->ai_next) {
		sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
		if (sockfd == -1) {
			continue;
		}

		res = connect(sockfd, p->ai_addr, p->ai_addrlen);
		if (res == -1) {
			close(sockfd);
			continue;
		}

		break;
	}

	if (p == NULL) {
		LOG(LOG_ERROR, "Establishing connection failed.");
		return -2;
	}

	inet_ntop(p->ai_family, get_addr((struct sockaddr *) p->ai_addr), server_address_string, sizeof server_address_string);

	LOG(LOG_INFO, "Connection established.");

	_socket = sockfd;
	_connected = true;

	return 0;
}

int UplinkSender::_terminate_connection()
{
	if (!_connected)
		return 0;

	char buffer[256];
	int res;
	int timeout = 0;

	shutdown(_socket, 2);
	while ((res = recv(_socket, &buffer[0], 256, MSG_DONTWAIT)) != 0) {
		LOG(LOG_INFO, "Waiting for connection to terminate");
		if (res < 0) {
			if (res != EAGAIN) { // if other error than again terminate
				break;
			}
		}
		sleep(1);
		if (timeout++ > 10) { // max wait 10 seconds
			break;
		}
	}
	close(_socket);

	_connected = false;
	return 0;
}

int UplinkSender::_write_data(uplink_packet &packet)
{
	int res;
	int size;
	char *buffer;

	packet.network_packet.header.packet_byte_size = sizeof(uplink_packet_header) + packet.network_packet.header.packet_readout_count * sizeof(uplink_single_readout) + sizeof(uint32_t);
	size = packet.network_packet.header.packet_byte_size;
	buffer = (char*) &(packet.network_packet);
	packet.network_packet.compute_header_checksum();
	packet.network_packet.compute_checksum();
	while (size > 0) {
		res = send(_socket, buffer, size, MSG_DONTWAIT | MSG_NOSIGNAL);
		if (res < 0) { // error
			if (errno == EAGAIN || errno == EWOULDBLOCK){
				return 1;
			} else {
				LOG(LOG_ERROR, "send() failed with error: "+ std::string(strerror(errno)));
				return -1;
			}
		}
		size -= res;
		buffer += res;
	}

	return 0;
}

void UplinkSender::before_join_thread()
{
	_buffer.release_reader();
}

int UplinkSender::buffer_push(uplink_packet &packet)
{
	return _buffer.push(packet);
}

void UplinkSender::buffer_invalidate(uint64_t unique_id)
{
	_buffer.invalidate(unique_id);
	std::unique_lock<std::mutex> lock(_dummy_packet_mutex);
	if (_dummy_packet.unique_id == unique_id)
		_dummy_packet.valid = false;
}

void UplinkSender::Main()
{

	LOG(LOG_INFO, "thread Main() start");
	while (_thread_run) {

		// Get packet from the buffer. Wait for packet if the buffer is empty.
		int res = _buffer.pop(_dummy_packet);
		_dummy_packet.network_packet.header.packet_sequence_counter = _packet_counter++;

		if (res < 0 || (!_dummy_packet.valid))
			continue;

		// Send the packet. Restart the connection if necessary.
		while (true) {
			if (!_thread_run)
				break;

			// check if the packet was invalidated after it was popped from the buffer
			_dummy_packet_mutex.lock();
			if (!_dummy_packet.valid) {
				_dummy_packet_mutex.unlock();
				_packet_counter--;
				break;
			}
			_dummy_packet_mutex.unlock();

			// Establish connection.
			if (!_connected) {
				res = _establish_connection();
				if (!_connected)
					sleep(2);
			}

			bool finished = false;

			// Send the packet.
			if (_connected) {
				res = _write_data(_dummy_packet);

				if (res < 0) { // error
					_terminate_connection();
					sleep(2);
				} else if (res > 0){ // would block
					usleep(10000);
				} else { // ok
					finished = true;
				}
			}

			// Report number of lost packets.
			struct timeval time_now, time_diff;
			gettimeofday(&time_now, NULL);
			timersub(&time_now, &_last_lost_data_check, &time_diff);
			if (time_diff.tv_sec > 59) {
				_last_lost_data_check = time_now;
				uint64_t count = _buffer.get_and_reset_lost_data_count();
				if (count > 0)
					LOG(LOG_ERROR, std::to_string(count) + " packets were lost in the last " +
							std::to_string(time_diff.tv_sec) + " seconds because the buffer was full.");
			}

			if (finished)
				break;
		}
	}
	if (_connected)
		_terminate_connection();
	LOG(LOG_INFO, "thread Main() exit");
}

