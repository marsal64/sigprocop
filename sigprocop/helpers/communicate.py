# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:45:25 2020

@author: martin
"""

import socket  # Importing socket library
import struct
import os
from netifaces import interfaces, ifaddresses, AF_INET
import sys
import datetime
import time

# sigproc host ip
HOST = '10.23.19.15'

# my setup to receive data to
MYPORT = 5555  # The port used by the server

# SigProc setup to send data to
SENDTOHOST = '10.23.19.15'
SENDTOPORT = 5555


class P:

    def __init__(self):
        self.packetcounter = -1

    def formpacket(self):
        """forms packet from object attributes saved outside the method, e.g.:

        self.packettype = 0
        self.deviceid = 'python'
        self.sensorid = 'result_01'
        self.values = [(ts, value * 2) ]

        saves bytes of message in self.message
    """

        self.message = bytearray(8192)

        self.message[0:2] = b'\x55\x00\x55'  # sync
        self.message[3:4] = b'\x00'  # packet type
        self.message[4:36] = self.deviceid.encode('utf-8')
        self.message[36:68] = self.sensorid.encode('utf-8')

        # increment and store packets counter
        self.packetcounter = (self.packetcounter + 1) % 65536
        self.message[68:70] = struct.pack('H', self.packetcounter)

        # evaluate number of readouts
        self.readouts = len(self.values)
        self.message[70:72] = struct.pack('H', self.readouts)

        # packet byte size
        self.packetsize = 80 + self.readouts * 24 + 4
        self.message[72:76] = struct.pack('I', self.packetsize)

        # header checksum
        self.header_checksum = chksm(self.message)
        self.message[76:80] = struct.pack('I', self.header_checksum)

        # data
        for i in range(0, self.readouts):
            ts = self.values[i][0]
            sec = int(ts)
            microsec = int((ts - sec) * 1000000)

            # debug
            microsec = self.timestamp_microsecondsin
            sec = self.timestamp_secondsin

            # seconds
            self.message[(80 + i * 24):(80 + i * 24) + 8] = struct.pack('Q', sec)
            # microseconds
            self.message[(80 + i * 24) + 8:(80 + i * 24) + 16] = struct.pack('Q', microsec)
            # value
            self.message[(80 + i * 24) + 16:(80 + i * 24) + 24] = struct.pack('d', self.values[i][1])

        # checksum
        chs = chksm(self.message)
        self.message[(self.packetsize - 4):self.packetsize] = struct.pack('I', chs)

        # fix length of bytearray
        self.message = self.message[0:self.packetsize]


def chksm(din):
    """checksum calculation from bytes

    checksum convention - data as array of unsigned 32 integers

    b: bytes
    returns: int with checksum
"""

    b = [din[i:i + 4] for i in range(0, len(din), 4)]
    c = [int.from_bytes(i, byteorder="little") for i in b]
    d = sum(c) % (2 ** (4 * 8))

    return d


# ping Sigproc (VPN)
print(f'\n*** Checking if server {HOST} to read from alive...')
response = os.system('ping ' + HOST + ' -c 2')
if response == 0:
    print(HOST, 'is up')
else:
    print(HOST, 'is down')

print('\n*** Checking network addressess...')
for ifaceName in interfaces():
    addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr': 'No IP addr'}])]
    print('%s: %s' % (ifaceName, ', '.join(addresses)))

# {D1E2A146-BBEF-409E-A370-B439B355726A}: 10.23.250.28


# Client socket
print('\n*** Opening client socket...')
# Create a TCP/IP socket
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Connect the socket to the port where the server is listening
server_address = (SENDTOHOST, SENDTOPORT)
print(f'*** Connecting to {SENDTOHOST}, port {SENDTOPORT}')
clientsocket.connect(server_address)

# new instance of data for packet to be sent with some defaults
p = P()
p.packettype = 0
p.deviceid = 'python'
p.sensorid = 'sensor_01'

# Server socket
print('\n*** Opening server socket...')
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# bind the socket to a public host, and a well-known port
serversocket.bind(('', MYPORT))
# become a server socket
serversocket.listen(5)

print('\n*** Server waiting for connection from SigProc...')
conn, address = serversocket.accept()

print(f"Connection from {address} detected")

while True:
    # accept connections from outside
    # now do something with the clientsocket
    # in this case, we'll pretend this is a threaded server

    # receive data stream. it won't accept data packet greater than 1024 bytes
    data = conn.recv(65536)
    if not data:
        # if data is not received break
        break

    # process incoming packet

    print(f'*** processing incoming packet, length {len(data)}')

    # parse some values from
    msync = data[0:3].hex()
    mtype = data[3:4].hex()
    deviceid = repr(data[4:36])
    sensorid = repr(data[36:68])
    checksum_header = chksm(data[0:(80 - 4)])
    checksum_message = chksm(data[0:(len(data) - 4)])
    timestamp_seconds = struct.unpack('Q', data[80:88])[0]
    timestamp_microseconds = struct.unpack('Q', data[88:96])[0]

    # debug
    p.timestamp_secondsin = timestamp_seconds
    p.timestamp_microsecondsin = timestamp_microseconds

    timestamp = datetime.datetime.utcfromtimestamp(timestamp_seconds) + datetime.timedelta(
        microseconds=timestamp_microseconds)
    value = struct.unpack('d', data[96:104])[0]
    packetcounterin = struct.unpack('H', data[68:70])[0]

    # debug
    # p.packetcounterin = packetcounterin

    print(repr(data))
    print(f'\n{data.hex()}')
    print(f"\nReceived message length: {len(data)}")
    print(f"SYNC: {msync}")
    print(f"TYPE: {mtype}")
    print(f"Device ID: {deviceid}")
    print(f"Sensor ID: {sensorid}")
    print(f"Packet counter: {data[68:70].hex()}, converted as {packetcounterin}")
    print(f"Number of readouts: {data[70:72].hex()}")
    print(f"Packet byte size: {data[72:76].hex()}")
    print(f"Readout 1 timestamp seconds: {data[80:88].hex()}, converted as {timestamp_seconds}")
    print(f"Readout 1 timestamp microseconds: {data[88:96].hex()}, converted as {timestamp_microseconds}")
    print(f"Readout 1 timestamp converted: {timestamp}")
    print(f"Readout 1 value: {data[96:104].hex()}, converted as {value}")
    print(
        f"Header checksum: {data[76:80].hex()}, {checksum_header.to_bytes(4, byteorder='little').hex()} calculated for verification")
    print(
        f"Packet checksum: {data[104:108].hex()}, {checksum_message.to_bytes(4, byteorder='little').hex()} calculated for verification")
    print()

    # send packet to input channel
    print('*** sending outgoing packet')

    ts = timestamp_seconds + timestamp_microseconds / 1000000

    # one or more values to send
    p.values = [(ts, value * 10)]

    # form packet bytearray
    message = p.formpacket()

    print(f"*** Sending {len(p.message)} bytes, value {p.values[0][1]}")
    print(f'\n{repr(p.message)}')
    print(f'\n{p.message.hex()}\n')

    clientsocket.sendall(p.message)

    # Look for the response
    # datarec = clientsocket.recv(8192)
    # print(f'*** Response after sending: received {repr(datarec)} bytes')

print('*** Closing client socket')
clientsocket.close()

print('*** Closing server socket')
conn.close()
