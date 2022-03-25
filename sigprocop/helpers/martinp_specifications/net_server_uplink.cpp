/*
 * net_server_uplink.cpp
 *
 *  Created on: Jun 24, 2019
 *      Author: jarda
 */

#include "net_server_uplink.h"
#include "../legacy/simple_log.h"
#include <bitset>
#include <fstream>

NetServerUplink::NetServerUplink() {

}

NetServerUplink::~NetServerUplink() {
}

int NetServerUplink::pushPacket(uint32_t ip, int port, uplink_packet &packet) {
	if (ip == 0 || port == 0)
		return -1;

	std::pair<uint32_t, int> server = std::make_pair(ip, port);

	uplinksMutex.lock();

	// find the server in the map
	auto it = uplinks.find(server);
	UplinkSender * uplink = nullptr;

	// if the server doesn't exist yet
	if (it == uplinks.end()) {
		int bufferSize = 1000;
		uplink = new UplinkSender("UP" + NetServerUplink::ipIntToString(ip) + ":" + std::to_string(port), 111, ip, port, bufferSize);
		auto res = uplinks.insert( { server, uplink });
		if (res.second) {
			LOG(LOG_INFO, "Created new uplink for streaming. Address " +
					NetServerUplink::ipIntToString(ip) + ":" + std::to_string(port) + ". Buffer size " + std::to_string(bufferSize));
			uplink->start_thread();
		} else {
			LOG(LOG_ERROR, "Creating new uplink for streaming failed. Address " + NetServerUplink::ipIntToString(ip) + ":" + std::to_string(port));
			delete uplink;
			uplinksMutex.unlock();
			return -2;
		}
	} else {
		uplink = it->second;
	}
	uplinksMutex.unlock();

	return uplink->buffer_push(packet);
}

void NetServerUplink::invalidatePackets(uint32_t ip, int port, uint64_t id) {
	std::pair<uint32_t, int> server = std::make_pair(ip, port);

	uplinksMutex.lock();

	// find the server in the map
	auto it = uplinks.find(server);
	UplinkSender * uplink = nullptr;

	// if the server exists
	if (it != uplinks.end()) {
		uplink = it->second;
	}
	uplinksMutex.unlock();

	if (uplink != nullptr)
		uplink->buffer_invalidate(id);
}

uint32_t NetServerUplink::ipStringToInt(const std::string &address) {

	if (address.empty())
		return 0;

	uint32_t ip = 0;
	std::vector<int> bytes;
	std::istringstream iss(address);
	std::string byte;

	while (getline(iss, byte, '.')) {
		bytes.push_back(stoi(byte));
	}

	if (bytes.size() != 4) {
		return 0;
	}

	for (auto item : bytes) {
		if (item < 0 || item > 255)
			return 0;
	}

	for (int i = 0; i < 4; ++i) {
		ip = ip | (bytes[i] & 0xFF) << ((3 - i) * 8);
	}
	return ip;
}

std::string NetServerUplink::ipIntToString(uint32_t address) {
	std::string ip;
	for (int i = 0; i < 4; ++i) {
		int byte = (address >> ((3 - i) * 8)) & 0xFF;
		if (i != 0)
			ip.append(".");
		ip.append(std::to_string(byte));
	}
	return ip;
}

void NetServerUplink::stop_threads() {
	std::lock_guard<std::mutex> guard(uplinksMutex);

	// stop all UplinkSenders, delete them and remove them from the map
	auto it = uplinks.begin();
	while (it != uplinks.end()) {
		it->second->stop_thread();
		delete it->second;
		it = uplinks.erase(it);
	}
}

