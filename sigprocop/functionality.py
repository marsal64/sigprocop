### functionality for optiguardml

import socket
import selectors
import types
import time
import os
import json
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import datetime
import asyncio
import struct

class Packet():

    packet_count_1 = 0
    packet_count_2 = 0

    def __init__(self, deviceid, sensorid, values, id):

        self.deviceid = deviceid
        self.sensorid = sensorid
        self.values = values

        # read and increment class packet_count attribute
        if id == 1:
            self.packetcounter = Packet.packet_count_1
            Packet.packet_count_1 += 1
            if Packet.packet_count_1 > 65535:
                Packet.packet_count_1 = 0
        elif id == 2:
            self.packetcounter = Packet.packet_count_2
            Packet.packet_count_2 += 1
            if Packet.packet_count_2 > 65535:
                Packet.packet_count_2 = 0
        else:
            pass

        self.message = bytearray(8192)

        self.message[0:2] = b'\x55\x00\x55'  # sync
        self.message[3:4] = b'\x00'  # packet type
        self.message[4:36] = deviceid.encode('utf-8')
        self.message[36:68] = sensorid.encode('utf-8')

        # increment and store packets counter
        self.message[68:70] = struct.pack('H', self.packetcounter)

        # evaluate number of readouts
        self.readouts = len(values)
        self.message[70:72] = struct.pack('H', self.readouts)

        # packet byte size
        self.packetsize = 80 + self.readouts * 24 + 4
        self.message[72:76] = struct.pack('I', self.packetsize)

        # header checksum
        
        self.header_checksum = chksm(self.message)
        self.message[76:80] = struct.pack('I', self.header_checksum)

        # build readouts
        for i in range(0, self.readouts):
            # seconds
            self.message[(80 + i * 24):(80 + i * 24) + 8] = struct.pack('Q', self.values[i][0])
            # microseconds
            self.message[(80 + i * 24) + 8:(80 + i * 24) + 16] = struct.pack('Q', self.values[i][1])
            # value
            self.message[(80 + i * 24) + 16:(80 + i * 24) + 24] = struct.pack('d', self.values[i][2])

       
        chs = chksm(self.message)
        self.message[(self.packetsize - 4):self.packetsize] = struct.pack('I', chs)

        # fix length of bytearray
        self.message = self.message[0:self.packetsize]


def chksm_old(din):
    """checksum calculation from bytes (old slow naive version)
    checksum convention - data as array of unsigned 32 integers
    b: bytes
    returns: int with checksum
"""

    b = [din[i:i + 4] for i in range(0, len(din), 4)]
    c = [int.from_bytes(i, byteorder="little") for i in b] # not so fast, may optimize here
    d = sum(c) % (2 ** (4 * 8))

    return d


def chksm(din):
    """checksum calculation from bytes
    checksum convention - data as array of unsigned 32 integers
    b: bytes
    returns: int with checksum
"""
    return int(np.frombuffer(din, dtype="uint32").sum() % (2 ** (4 * 8)))


def packetparse(data, F):
    """ parses packet, returns deviceid, sensoid, values
    
    expects valid packet data
    
    """

    deviceid = (data[4:36]).split(b'\x00')[0].decode('utf-8')
    sensorid = (data[36:68]).split(b'\x00')[0].decode('utf-8')
    readouts = struct.unpack('H', data[70:72])[0]
    
    packetcounter = struct.unpack('H', data[68:70])[0]
    F.logger.debug(f"Input packet counter: {packetcounter}")

    values = []
    for i in range(80, 80 + readouts * 24, 24):
        ts = struct.unpack('Q', data[i:(i+8)])[0]
        tms = struct.unpack('Q', data[(i+8):(i+16)])[0]
        val = struct.unpack('d', data[(i + 16):(i + 24)])[0]
        values.append((ts, tms, val))

    return deviceid, sensorid, values


def preparescoring(F):
    """Initiates scoring"""


def inprocess(deviceid, sensorid, values, F):
    """process input packet
    
    Structure of the buffer F.db:
    
    column 0 - sec timestamp when the row arrived
    column 1 - 'status' column, counting row measurements
    column 2,3,... - columns with measurements
    
    """


    ### put to buffer
    # find if the timestamp already exists in the buffer F.db
    for v in values:
        sec = v[0]
        ms = v[1]
        val = v[2]
        arrived = time.time()   # epoch seconds 

        # find colname, verify deviceid and sensorid are valid
        try: 
            colname = F.iotr.loc[(deviceid, sensorid), 'inputid']
        except Exception as e:
            F.logger.warning(f"device and/or sensor names in message ('{deviceid}' and '{sensorid}'), not found in config .json")
            return None

        # write to existing row or insert the new row with the value
        F.db.loc[(sec, ms), colname] = val

        # find index of the row
        indrow = F.db.index.get_loc((sec, ms))
        
        # set 'arrived' column value (may replace previous values)
        F.db.iat[indrow, 0] = arrived
        
        # increment the value count in 'status' column
        F.db.iat[indrow, 1] = 1 if np.isnan(F.db.iat[indrow, 1]) else F.db.iat[indrow, 1] + 1
        
        # check statis all data in the row? Then score, put to output and erase from db
        if F.db.iat[indrow, 1] == F.numinputs:   # 0 column - 'stats'
            try:
                # run the values through the model
                X = F.db.iloc[indrow, 2:].values.reshape(1, -1)
                y = F.spop_predict(X)
                F.logger.debug(f"from input values {X}, the spop_predict function  calculated the output value {y}, sending to outputs")

                # send to output
                valout = [(sec, ms, y)]
                p = Packet(F.output['device'], F.output['sensor'], valout, 1)
                F.logger.debug(
                    f"sending {len(p.message)} bytes output packet deviceid={p.deviceid}, sensorid={p.sensorid}, values: {p.values}")
                # put message to output queue
                F.outq.appendleft(p.message)

                # drop the row
                F.db.drop(F.db.index[indrow], inplace=True)

            except Exception as e:
                F.logger.warning(f"Error when trying to predict from values {X}. Exception:\n{e}")
                F.logger.warning(f"Current length of buffer: {F.db.index.size}")



    ### emergency trim if db too long?
    if F.db.index.size > F.maxdblines:
        F.logger.warning(f'buffer size exceeded maximum value ({F.maxdblines}), dropping older values')
        # leave the first row and last n/2 rows
        db1 = F.db.iloc[0:1]
        db2 = F.db.iloc[-F.maxdblines//2:]
        del F.db
        F.db = db1.append(db2)

        
    ### periodical maintenance of the db
    if F.buffermaintaintime < arrived:
    
        # mark next maintaintime
        F.buffermaintaintime = arrived + F.dbmaintainsec

        # debug buffer size
        F.logger.debug(f"starting db periodical maintenance, db size: {F.db.index.size}")

        # browse through database and find obsolete rows
        for i, r in F.db.iloc[1:].iterrows():

            # tries to gc "old" F.db
            dbtemp = F.db.loc[(F.db.index.get_level_values(0) == 0) | (F.db.arrived + F.dbsectoerase > arrived)]

            if dbtemp.index.size < F.db.index.size:
                F.logger.warning(f'cleansing {F.db.index.size - dbtemp.index.size} rows in the buffer with too old timestamps')
            del F.db
            F.db = dbtemp
