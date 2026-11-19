"""Puerto serie y sesion UCI: hilos de escritura/lectura, IPC por zmq y configuracion del puerto."""
import sys
import time
from datetime import datetime

import serial
import zmq

import config as cfg
import state as st
from cir_capture import handle_cir_fragment, is_cir_candidate
from presence import record_presence, save_presence_raw
from util import output


class SIGINThandler():
    def __init__(self):
        self.sigint = False
    
    def signal_handler(self, signal, frame):
        output("You pressed Ctrl+C!")
        self.sigint = True


def write_to_serial_port():
    
    usb_out_packet = bytearray()
    
    output("Write to serial port started")
    while (not st.stop_write_thread):
        if (st.retry_cmd):
            st.retry_cmd = False
        else:
            uci_command = st.command_queue.get()
        
        if (uci_command[0] == 0xFF and uci_command[1] == 0xFF):
            break
        if (uci_command[0] == 0xFF and uci_command[1] == 0x00):
            time.sleep(uci_command[2])
            continue
        
        usb_out_packet.clear()
        usb_out_packet.append(0x01)
        usb_out_packet.append(int(len(uci_command) / 256))
        usb_out_packet.append(len(uci_command) % 256)
        usb_out_packet.extend(uci_command)
        
        if (uci_command[0] == 0x21 and uci_command[1] == 0x00):
            st.last_session_started = uci_command[4:8]
        if (uci_command[0] == 0x22 and uci_command[1] == 0x00):
            st.session_status.allow_start.wait()
            st.go_stop.clear()
        
        if (uci_command[0] == 0x22 and uci_command[1] == 0x01):
            st.session_status.allow_stop.wait()
            st.go_stop.wait()
        
        if (uci_command[0] == 0x21 and uci_command[1] == 0x01):
            st.session_status.allow_deinit.wait()
        
        st.write_wait.acquire()
        if st.serial_port.isOpen():
            if (cfg.is_timestamp):
                output(datetime.now().isoformat(sep=" ", timespec="milliseconds") + " NXPUCIX => " + \
                      "".join("{:02x} ".format(x) for x in uci_command))
            else:
                output("NXPUCIX => " + "".join("{:02x} ".format(x) for x in uci_command))
            
            try:
                st.serial_port.write(serial.to_bytes(usb_out_packet))
            except:
                output("Fail to write on serial port")
            
            notified = st.write_wait.wait(0.25)  
            if (not (notified)): st.retry_cmd = True
        st.write_wait.release()
    
    output("Write to serial port exited")


def read_from_serial_port():
    
    output("Read from serial port started")
    while (not st.stop_read_thread):
        if st.serial_port.isOpen():
            if st.serial_port.isOpen():
                uci_hdr = st.serial_port.read(4)
                st.write_wait.acquire()
                if len(uci_hdr) == 4:
                    count = uci_hdr[3]
                    if (uci_hdr[1] & 0x80) == 0x80:
                        count = int((uci_hdr[3] << 8) + uci_hdr[2])
                    
                    if count > 0:
                        if st.serial_port.isOpen():
                            uci_payload = st.serial_port.read(count)
                            
                            if (not (cfg.is_cir_save and is_cir_candidate(uci_hdr))):
                                if (cfg.is_timestamp):
                                    output(datetime.now().isoformat(sep=" ", timespec="milliseconds") + \
                                          " NXPUCIR <= " + "".join("{:02x} ".format(h) for h in uci_hdr) + \
                                          "".join("{:02x} ".format(p) for p in uci_payload))
                                else:
                                    output("NXPUCIR <= " + "".join("{:02x} ".format(h) for h in uci_hdr) + \
                                          "".join("{:02x} ".format(p) for p in uci_payload))
                            
                            if len(uci_payload) == count:
                                if (uci_hdr[0] & 0xF0) == 0x40: st.write_wait.notify()
                                if (uci_hdr[0] == 0x41 and uci_hdr[1] == 0x00):
                                    session_handle =  uci_payload[1:]
                                    for command in st.command_queue.queue:
                                        if (command[0] == 0x21 or command[0] == 0x22 or command[0] == 0x2F or command[0] == 0x2D ):
                                            if(st.last_session_started == command[4:8]):
                                                command[4]= session_handle[0]
                                                command[5]= session_handle[1]
                                                command[6]= session_handle[2]
                                                command[7]= session_handle[3]
                                if (uci_hdr[0] == 0x60 and uci_hdr[1] == 0x07 and uci_hdr[3] == 0x01 and \
                                        uci_payload[0] == 0x0A):
                                    st.retry_cmd = True
                                    st.write_wait.notify()
                                
                                if (uci_hdr[0] == 0x61 and uci_hdr[1] == 0x02 and uci_hdr[3] == 0x06):
                                    st.session_status.set(uci_payload[4])
                                    
                                    if (uci_payload[5] == 0x01):
                                        st.go_stop.set()
                                    elif (uci_payload[5] == 0x02):
                                        st.go_stop.set()
                                    elif (uci_payload[5] == 0x83):
                                        st.go_stop.set()
                                    
                                    if ((cfg.is_ipc) and (uci_payload[4] == 0x02)):
                                        st.file_ipc.close()
                                        if(st.socket is not None):
                                            try:
                                                st.socket.send_string("started")
                                            except:
                                                output("Fail to send started on socket")
                                
                                if (uci_hdr[0] == 0x69 and uci_hdr[1] == 0x0A):
                                    if (cfg.is_cir_save):
                                        save_presence_raw(uci_hdr, uci_payload)
                                    record_presence(uci_payload)
                                    
                                    if (uci_payload[6] == 0x01):
                                        output("Presence Detected!")
                                        if ( cfg.detection_mode == 0x03 ):
                                            distance = (uci_payload[11] << 8) + uci_payload[10]
                                            output("presence_distance : " + str(distance) + "cm")

                                            angle = uci_payload[12]
                                            if( angle >= 128 ):
                                                angle = angle - 256

                                            output("presence_angle : " + str(angle) + "deg")

                                    else:
                                        output("Presence Lost...")

                                if (cfg.is_cir_save and is_cir_candidate(uci_hdr)):
                                    handle_cir_fragment(uci_hdr, uci_payload)

                                if (uci_hdr[0] == 0x4F and uci_hdr[1] == 0x22 and uci_payload[2] == 0x04):
                                    is_set_tx_power = False
                                    for txpower_cal_idx in range(0, uci_payload[3]-1):
                                        if (uci_payload[txpower_cal_idx+5] != 0):
                                            is_set_tx_power = True
                                            break
                                    if (is_set_tx_power == True):
                                        for command in st.command_queue.queue:
                                            if (command[0] == 0x2F and command[1] == 0x21 and command[5] == 0x04):
                                                if (command[4] == 0x09):
                                                    for txpower_cal_idx in range(0, 10):
                                                        command[8+txpower_cal_idx] = uci_payload[5+txpower_cal_idx]
                            else:
                                output("\nExpected Payload bytes is " + str(count) + \
                                      ", Actual Paylod bytes received is " + str(len(uci_payload)))
                        else:
                            output("Port is not opened")
                    else:
                        output("\nUCI Payload Size is Zero")
                st.write_wait.release()
            else:
                output("Port is not opened (2)")
        else:
            output("Port is not opened (1)")
    
    if st.serial_port.isOpen(): st.serial_port.close()
    
    output("Read from serial port exited")


def ipc_file_name():
    
    output("IPC for output file name started")
    
    st.socket = zmq.Context().socket(zmq.PAIR)
    st.socket.connect("tcp://localhost:9999")
    
    while (not st.stop_ipc_thread):
        new_file_name = ""
        event = st.socket.poll(timeout=1000, flags=zmq.POLLIN)
        if event == zmq.POLLIN:
            new_file_name = st.socket.recv_string()
        
        if (new_file_name != ""):
            if (new_file_name == "STOP"):
                st.session_status.set_all()
            else:
                if ((st.file_ipc is not None) and (not st.file_ipc.closed)):
                    st.file_ipc.close()
                
                output("New file name: " + cfg.prefix_ipc + new_file_name)
                st.file_ipc = open(cfg.prefix_ipc + new_file_name, "w")
            
    if ((st.file_ipc is not None) and (not st.file_ipc.closed)):
        st.file_ipc.close()
    
    try:
        st.socket.send_string("closed")
    except:
        output("Fail to send closed on socket")
        
    output("IPC for output file name exited")


def serial_port_configure():
    
    st.serial_port.baudrate = 3000000
    st.serial_port.timeout = 1
    st.serial_port.port = cfg.com_port
    st.serial_port.rtscts = True

    if st.serial_port.isOpen(): st.serial_port.close()
    
    try:
        st.serial_port.open()
    except:
        output("#=> Fail to open " + cfg.com_port)
        sys.exit(1)

    try:
        InitCmd = [0x04,0x00,0x00]
        output("#=> INIT_CMD: " + "".join("{:02x} ".format(h) for h in InitCmd))
        st.serial_port.write(InitCmd)
    except:  
        output("#=> Fail to Write Board Init CMD on " + cfg.com_port)
        sys.exit(1)
     
    time.sleep(0.2)
    
    rx = st.serial_port.read(st.serial_port.in_waiting)
    if (len(rx) != 4):
        output("#=> Fail to Read Board Init RSP " )
        output("############ Please RESET the Board and re-run the script ############")
        sys.exit(1)
    else:
        output("#<= INIT_RSP: " + "".join("{:02x} ".format(h) for h in rx))
        
    try:
        InitCmd = [0x12,0x00,0x00]
        output("#=> HDLL_TO_UCI_CMD: " + "".join("{:02x} ".format(h) for h in InitCmd))
        st.serial_port.write(InitCmd)
    except:  
        output("#=> Fail to Write HDLL to UCI CMD on " + cfg.com_port)
        sys.exit(1)
     
    time.sleep(0.2)

    rx = st.serial_port.read(st.serial_port.in_waiting)
    if (len(rx) != 5):
        output("#=> Fail to Read Write HDLL to UCI CMD " )
        output("############ Please RESET the Board and re-run the script ############")
        sys.exit(1)
    else:
        output("#<= HDLL_TO_UCI_RSP: " + "".join("{:02x} ".format(h) for h in rx))
