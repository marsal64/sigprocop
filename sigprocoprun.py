#####################################################
#####################################################
###
### SigProcOpenPython - SigProc add on for user programming in python
###
### SigProcOpenPython runner
###
#####################################################

import socket
import select
import time
import os
import json
import argparse
import logging
import sys
import numpy as np
import pandas as pd
import collections
import dill
import struct

from sigprocop.functionality import chksm, Packet, packetparse, inprocess

### parse command line arguments
parser = argparse.ArgumentParser(description='SigProcOpenPython main runner')

parser.add_argument('--paramfile', type=str, default='sigprocop.json', help='.json file with parameters.')
parser.add_argument('--debug', action='store_true', default=False, help='If present, runs in debug mode.')
parser.add_argument('--rundir', type=str, default='', help='If present, switches to the given directory. If not present, it runs in the directory where the sigprocoprun.py script is present. ')
parser.add_argument('--nolog', action='store_true', default=False, help='If present, disables logging.')

# development related only
parser.add_argument("--mode", default='client', help='Internal parameter used by pydev')
parser.add_argument("--port", default=36829, help='Internal parameter used by pydev')

# collect parameters from line arguments to F namespace object
F = parser.parse_args()


### cd to the working directory
if F.rundir == '':
    # switch to the directory where the file is present
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
else:
    # switch to the directory as given in parameter '--rundir'
    try:
        os.chdir(F.rundir)
    except Exception as e:
        print(f"cannot switch to '{F.rundir}', exiting. Exception:\n{e}")
        sys.exit(1)

# add base directory to F as 'dir'
F.dir = os.getcwd()


### load main set of parameters from the json file and add to F
try:
    with open(F.paramfile, 'r') as fpf:
        L = json.load(fpf)
except Exception as e:
    print(F"file '{F.paramfile}' cannot be loaded, exiting. Exception:\n{e}")
    sys.exit(1)

# add .json attributes to already existing F namespace
for key in L:
    setattr(F, key, L[key])


### initiate logging
# create logs directory if not exists
try:
    os.mkdir(os.path.join(F.dir, 'logs'))
except Exception as e:
    pass

F.logger = logging.getLogger(F.name)  # logger name
loggerconsoleh = logging.StreamHandler()  # log to console
loggerfileh = logging.FileHandler(
    os.path.join(F.dir, 'logs', F.name + "_run.log"))  # log to file in 'logs' subdir
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add loging to console and file
loggerconsoleh.setFormatter(formatter)
loggerfileh.setFormatter(formatter)
# set logger handler for logging
F.logger.addHandler(loggerconsoleh)
F.logger.addHandler(loggerfileh)

# disable all logging?
if F.nolog:
    logging.disable(logging.ERROR)


### convenience settings
# avoid np warnings (always raise error)
np.seterr('raise')
# avoid pandas warning
pd.options.mode.chained_assignment = None
# tune screen column width
pd.set_option("display.max_colwidth", 500)
# tune numpy print (for percentiles)
np.set_printoptions(threshold=sys.maxsize)


### set logging level
if F.debug:
    F.logger.setLevel(logging.DEBUG)
else:
    F.logger.setLevel(logging.INFO)

F.logger.info("started")
F.logger.info(f"parameters:\n{F}")


### load the function spop_predict
try:
    F.spop_predict = dill.load(open(F.spop_predict_function['modelfile'], 'rb'))
except Exception as e:
    F.logger.error(f"cannot load spop_predict function from file '{F.spop_predict_function['modelfile']}', exiting. Exception:\n{e}")
    sys.exit(1)


### initialize the data buffer
F.logger.info("data buffer initiation")
ins = [i['name'] for i in F.inputs]
# build list for F.db with one empty row
Fdbl = {
    'sec': pd.Series([0], dtype=np.int64),
    'ms': pd.Series([0], dtype=np.int64),
    'arrived': pd.Series([0], dtype=np.int64),
    'status': pd.Series([0], dtype=np.int8)
}
# add columns for channels
for i in ins:
    Fdbl[i] = pd.Series([0], dtype=np.float64)
# initiate buffer as pandas DataFrame with one dummy row
F.db = pd.DataFrame(Fdbl)
# set index
F.db.set_index(['sec', 'ms'], drop=True, inplace=True)

# helper dataframe F.iotr that translates input device id and sensor id to name of the input for the model
dtlist = [('deviceid', str), ('sensorid', str), ('inputid', str)]
F.iotr = pd.DataFrame(np.empty(0, dtype=np.dtype(dtlist))).set_index(['deviceid', 'sensorid'])
for i, j in enumerate(F.inputs):
    F.iotr.loc[(j['fromdevice'], j['fromsensor']), 'inputid'] = j['name']
        
# helper value F.numinputs - number of input channels
F.numinputs = F.iotr.index.size


### sockets initiation
# lsock: listen as server to clients sending measurements data to SigProcOpenPython
lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    lsock.bind((F.listen['ip'], F.listen['port']))
except Exception as e:
    F.logger.error(f"exception when binding:\n{e}")
    sys.exit(1)
try:
    lsock.listen()
except Exception as e:
    F.logger.error(f"exception when starting to listen:\n{e}")
    sys.exit(1)

lsock.setblocking(False)
F.logger.info(f"listening on {F.listen['ip']}:{F.listen['port']}")

# osock: connect as client to the server which is waiting for outputs from SigProcOpenPython
server_addr = (F.output["ip"], F.output["port"])
F.logger.info(f"connecting to server {F.output['ip']}:{F.output['port']}...")
osock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# during reboot, may not have networking ready
while True:
    try:
        osock.connect(server_addr)
        break
    except Exception as e:
        F.logger.warning(f'waiting for the connection 1 to {server_addr}')
        time.sleep(3)
        pass
        
osock.setblocking(False)
# now connected, can change to nonblocking
F.logger.info(f"connected to server {F.output['ip']}:{F.output['port']} to send output")



# lists of sockets to read from
inputs = [lsock]
# lists of sockets to write to
outputs = [osock]

# input queue
F.inq = collections.deque()
# output queue 1
F.outq = collections.deque()

# counter for erasing older rows in buffer
F.buffermaintaintime = time.time()

# input data buffer
databuf = bytearray()

### main processing loop
while True:

    # exceeded length of internal input messages buffer?
    if len(databuf) > F.maxdatabuf:
        F.logger.error(f'internal messagging buffer databuf exceeded maximum length {F.maxdatabuf} ({len(databuf)}), exiting')
        sys.exit(1)

    # process input and output events
    readable, writable, exceptional = select.select(inputs, outputs, inputs, 0)

    # Handle inputs
    for s in readable:
        if s is lsock:
            # A "readable" server socket is ready to accept a connection
            connection, client_address = s.accept()
            F.logger.info(f"new connection from {client_address}")
            connection.setblocking(0)
            inputs.append(connection)
        else:
            dati = s.recv(8192)   # read header of the message (80 bytes), if exists
            if dati:
                # read rest of the message
                databuf.extend(dati)
                #F.logger.debug(f"received {repr(data)} from {s.getpeername()}")
            else:
                # Interpret empty result as closed connection
                F.logger.debug(f"closing {client_address} after reading no data")
                # Stop listening for input on the connection
                inputs.remove(s)
                s.close()

    # process all full messages in the buffer and put them to input queue
    while len(databuf) >= 108:
        # verify if message start, either flush the buffer
        if databuf[0:4] != b'\x55\x00\x55\x00':
            F.logger.warning(f'buffer first bytes are not x55x00x55x00 as expected, flushing')

            # try to find next
            nextstart = databuf.find(b'\x55\x00\x55\x00')
            if nextstart == -1:
                # nothing found, cleanse (and gc in this occasion) the buffer
                databuf = bytearray()
            else:
                # found
                databuf = databuf[nextstart:]
            break

        # find message length
        inmeslen = struct.unpack('I', databuf[72:76])[0]

        # message length non-standard?
        if inmeslen not in (108, 132, 156, 180, 204, 228, 252, 276, 300, 324):
            F.logger.warning(f"found non-standard message length {inmeslen} in packet, flushing part of the buffer")

            # try to find next message
            nextstart = databuf.find(b'\x55\x00\x55\x00', 4)
            if nextstart == -1:
                # nothing found, cleanse (and gc in this occasion) the buffer
                databuf = bytearray()
            else:
                # found
                databuf = databuf[nextstart:]
            break

        # full message in the buffer?
        if len(databuf) >= inmeslen:
            data = databuf[0:inmeslen]

            # verify checksums
            checksum_header_calc = chksm(data[0:(80 - 4)])
            checksum_header_packet = struct.unpack('I', databuf[76:80])[0]

            checksum_message_calc = chksm(data[0:(len(data) - 4)])
            checksum_message_packet = struct.unpack('I', data[-4:])[0]

            if checksum_header_calc == checksum_header_packet and checksum_message_calc == checksum_message_packet:
                # chechskums test passed, put to input queue for processing
                F.inq.appendleft(data)
                # remove message from the buffer
                databuf = databuf[inmeslen:]
            else:
                # checksums test failed
                F.logger.warning(f'header or message checksum check failed, flushing part of the buffer')

                # try to find next message
                nextstart = databuf.find(b'\x55\x00\x55\x00', 4)
                if nextstart == -1:
                    # nothing found, cleanse the buffer (and gc) 
                    databuf = bytearray()
                else:
                    # found
                    databuf = databuf[nextstart:]
                break

        else:
            break

    # If output message, put to output
    errout = False # error for sending to output flag
    for s in writable:
        if s is osock and F.outq:   # not empty
            try:
                next_msg = F.outq.pop()
                F.logger.debug(f"sending {repr(next_msg)} (output) to {s.getpeername()}")
                osock.send(next_msg)
            except Exception as e:
                F.logger.error(f"exception when sending message to output. Exception:\n{e}")
                errout = True
                # will try next time


    # if errout, reconnect
    if errout:
        F.logger.warning("Trying to reconnect")
    
        # osock: connect as client to the server which is waiting for outputs from sigprocopenpython
        server_addr = (F.output["ip"], F.output["port"])
        F.logger.info(f"connecting to server {F.output['ip']}:{F.output['port']}...")
        osock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # during reboot, may not have networking ready
        while True:
            try:
                osock.connect(server_addr)
                break
            except Exception as e:
                F.logger.warning(f'waiting for the connection 1 to {server_addr}')
                time.sleep(3)
                pass
                
        osock.setblocking(False)
        # now connected, can change to nonblocking
        F.logger.info(f"connected to server {F.output['ip']}:{F.output['port']} to send output")

        outputs = [osock]
   

    # Handle "exceptional conditions"
    for s in exceptional:
        F.logger.debug(f"handling exceptional condition for {s.getpeername()}")
        # Stop listening for input on the connection
        inputs.remove(s)
        s.close()

    # Input messages?
    while F.inq:
        # process input message
        message = F.inq.pop()
        # parse message packet
        rev = packetparse(message, F)
        if rev is not None:
            deviceid, sensorid, values = rev
            F.logger.debug(f"received message {len(message)} bytes, deviceid={deviceid}, sensorid={sensorid}, values: {values}")

            # process input message
            inprocess(deviceid, sensorid, values, F)

### extraordinary end of main processing cycle
F.logger.info(f"stopped")
