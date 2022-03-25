/*
 * uplink.h
 *
 *  Created on: Jul 23, 2014
 *      Author: jarda
 */

#ifndef SOURCE_DIRECTORY__SRC_NET_UPLINK_H_
#define SOURCE_DIRECTORY__SRC_NET_UPLINK_H_

#include <map>
#include <mutex>
#include "../legacy/active.h"
#include "net_server.h"
#include "uplink_buffer.h"

#define UPLINK_VECTOR_MAX_DATA_COUNT		1024
#define UPLINK_MAX_READOUT_COUNT_PER_PACKET	1024

#define UPLINK_PACKET_TYPE_SINGLE_READOUT	0
#define UPLINK_PACKET_TYPE_VECTOR_READOUT	1
#define UPLINK_PACKET_TYPE_PING			2

typedef struct uplink_packet_tag {
	uint64_t unique_id;
	bool valid = false;
	uplink_packet_network network_packet;
} uplink_packet;

class UplinkSender: public ActiveObject {
public:
	UplinkSender(std::string s, int o, uint32_t ip_address, int port_number, int buffer_size);
	~UplinkSender();

	virtual void Main();
	virtual void before_join_thread();   // called in same context as stop_thread()

	int buffer_push(uplink_packet &packet);
	void buffer_invalidate(uint64_t unique_id);

private:
	UplinkBuffer<uplink_packet> _buffer;
	bool _connected;
	int _socket;
	uint32_t _ip_address;
	int _port_number;
	uint16_t _packet_counter;
	uplink_packet _dummy_packet;
	std::mutex _dummy_packet_mutex;
	struct timeval _last_lost_data_check;

	int _establish_connection();
	int _terminate_connection();
	int _write_data(uplink_packet &packet);
};

#endif /* SOURCE_DIRECTORY__SRC_NET_UPLINK_H_ */

