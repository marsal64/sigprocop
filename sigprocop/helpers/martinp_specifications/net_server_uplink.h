/*
 * net_server_uplink.h
 *
 *  Created on: Jun 24, 2019
 *      Author: jarda
 */

#ifndef SOURCE_DIRECTORY__SRC_NET_NET_SERVER_UPLINK_H_
#define SOURCE_DIRECTORY__SRC_NET_NET_SERVER_UPLINK_H_

#include "uplink.h"
#include <map>
#include <string>
#include <mutex>

class NetServerUplink {
public:
	NetServerUplink();
	virtual ~NetServerUplink();

	void stop_threads();
	/**
	 * Converts string holding an IP address to an unsigned integer.
	 * ip_string must be in a format X.X.X.X or it can be "localhost"
	 * returns zero if the address is not valid
	**/
	static uint32_t ipStringToInt(const std::string &address);

	/**
	 * Converts unsigned integer holding an IP address to a string.
	**/
	static std::string ipIntToString(uint32_t address);

	/**
	 * Push packet into a buffer.
	 */
	int pushPacket(uint32_t ip, int port, uplink_packet &packet);
	/**
	 * Invalidated packets with the given id.
	 */
	void invalidatePackets(uint32_t ip, int port, uint64_t id);

private:
	std::map<std::pair<uint32_t, int>, UplinkSender*> uplinks;
	std::mutex uplinksMutex;

	inline std::string soid() {return "UplinkManager";};
};

#endif /* SOURCE_DIRECTORY__SRC_NET_NET_SERVER_UPLINK_H_ */
