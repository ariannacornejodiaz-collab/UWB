"""Punto de entrada:  python main.py [COM12] [notime] [nocir] [noplot] [ipc <prefijo>] [scenario=..] [window=..] [avg=..] [bg=..] [offset=..] [thr=..] [tau=..] [label=..]"""
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime
from threading import Thread

import numpy as np

import config as cfg
import state as st
from csv_export import save_dist_aoa, save_ml_csv, save_presence_log
from plot_live import run_live_plot
from serial_io import (SIGINThandler, ipc_file_name, read_from_serial_port, serial_port_configure,
                       write_to_serial_port)
from snapshots import save_final_averages
from uci_commands import (
    UWB_CORE_SET_ANTENNAS_DEFINE,
    UWB_CORE_SET_CONFIG,
    UWB_CORE_SET_PDOA_CALIB_TABLE_DEFINE,
    UWB_INIT_BOARD_VARIANT,
    UWB_RANGE_SESSION_DEINIT,
    UWB_RANGE_START,
    UWB_RANGE_STOP,
    UWB_RESET_DEVICE,
    UWB_SESSION_INIT_RADAR,
    UWB_SESSION_SET_APP_CONFIG,
    UWB_SESSION_SET_VENDOR_APP_CONFIG_NXP,
    UWB_SET_CALIBRATION_AOA_ANTENNAS_PDOA_CALIB_PAIR1_CH9,
    UWB_SET_CALIBRATION_AOA_ANTENNAS_PDOA_CALIB_PAIR2_CH9,
    UWB_SET_CALIBRATION_AOA_THRESHOLD_PDOA_CH9,
    UWB_SET_CALIBRATION_PDOA_OFFSET_CALIB_CH9,
    UWB_SET_CALIBRATION_RX_ANT_DELAY_CALIB_CH9,
)
from util import output


def start_processing():

    if (cfg.is_ipc):
        st.stop_ipc_thread = False
        read_thread = Thread(target=ipc_file_name, args = ())
        read_thread.start()
    
    st.stop_read_thread = False
    read_thread = Thread(target=read_from_serial_port, args=())
    read_thread.start()
    
    st.stop_write_thread = False
    write_thread = Thread(target=write_to_serial_port, args=())
    write_thread.start()
    
    handler = SIGINThandler()
    signal.signal(signal.SIGINT, handler.signal_handler)
    
    if (cfg.is_plot and cfg.is_cir_save):
        run_live_plot(handler)
    else:
        while (st.session_status.allow_end.is_set() == False):
            if handler.sigint:
                break
            time.sleep(0.01)
    
    cfg.is_ipc = False
    
    st.stop_write_thread = True
    st.stop_read_thread = True
    st.stop_ipc_thread = True
    
    st.command_queue.put([0xFF, 0xFF])
    st.session_status.set_all()
    st.go_stop.set()


def main():
    
    path = ""
    
    for arg in sys.argv[1:]:
        if (arg.startswith("COM") | arg.startswith("/dev")):
            cfg.com_port = arg
        elif (arg == "notime"):
            cfg.is_timestamp = False
        elif (arg == "nocir"):
            cfg.is_cir_save = False
        elif (arg == "noplot"):
            cfg.is_plot = False
        elif (arg == "ipc"):
            cfg.is_ipc = True
        elif (arg.startswith("scenario=")):
            cfg.scenario_name = arg.split("=", 1)[1] or cfg.scenario_name
        elif (arg.startswith("bg=")):
            _bg = np.load(arg.split("=", 1)[1])
            st.bg_data = {"ref": _bg["ref"], "sigma": _bg["sigma"]}
        elif (arg.startswith("offset=")):
            cfg.SAMPLE_OFFSET = max(0, min(cfg.BLOCK_SAMPLES - 8, int(arg.split("=", 1)[1])))
            cfg.N_TAPS_EFF = cfg.BLOCK_SAMPLES - cfg.SAMPLE_OFFSET
        elif (arg.startswith("avg=")):
            cfg.AVG_MODE = "coherent" if arg.split("=", 1)[1].startswith("coh") else "abs"
        elif (arg.startswith("label=")):
            cfg.label = arg.split("=", 1)[1]
        elif (arg.startswith("thr=")):
            cfg.PRESENCE_THR_ON = float(arg.split("=", 1)[1])
            cfg.PRESENCE_THR_OFF = max(1.0, 0.75 * cfg.PRESENCE_THR_ON)
        elif (arg.startswith("tau=")):
            cfg.PRESENCE_TAU_S = max(1.0, float(arg.split("=", 1)[1]))
        elif (arg.startswith("window=")):
            cfg.AVG_WINDOW = max(1, int(arg.split("=", 1)[1]))
            st.win_raw = deque(maxlen=cfg.AVG_WINDOW)
            st.win_clean = deque(maxlen=cfg.AVG_WINDOW)
        else:
            path = arg
    
    if (cfg.is_ipc):
        cfg.is_timestamp = False
        
        if (path == ""):
            data_path = os.getcwd()
            cfg.prefix_ipc = "ipc_"
            
        else:
            slash_idx = path.rfind("/")
            backslash_idx = path.rfind("\\")
            
            if ((slash_idx >= 0) or (backslash_idx >= 0)):
                cfg.prefix_ipc = path[max(slash_idx, backslash_idx) + 1:]
                data_path = os.getcwd() + "/" + path[:max(slash_idx, backslash_idx)]
            else:
                cfg.prefix_ipc = path
                data_path = os.getcwd()
            
            if (cfg.prefix_ipc[-1] != "_"):
                cfg.prefix_ipc = cfg.prefix_ipc + "_"
            
            try:
                os.stat(data_path)
            except:
                os.makedirs(data_path)
        
        os.chdir(data_path)
        output("Working directory: " + data_path)
        
        new_file_name = "_init.txt"
        output("New file name: " + cfg.prefix_ipc + new_file_name)
        st.file_ipc = open(cfg.prefix_ipc + new_file_name, "w")
    
    if (cfg.is_cir_save):
        st.cir_dir = os.path.join(os.getcwd(), "cir_" + datetime.now().strftime("%y%m%d_%H%M%S"))
        os.makedirs(st.cir_dir, exist_ok=True)
        with open(os.path.join(st.cir_dir, "index.csv"), "w") as f:
            f.write("time,file,gid,oid,bytes\n")
        with open(os.path.join(st.cir_dir, "cir_summary.csv"), "w") as f:
            f.write("time,file,seq,rx_tag,peak_idx,peak_amp,power_db\n")
        with open(os.path.join(st.cir_dir, "cir_headers.csv"), "w") as f:
            f.write("time,seq,rx_tag,header_hex\n")
        with open(os.path.join(st.cir_dir, "presence_raw.csv"), "w") as f:
            f.write("time,n,hdr,payload_hex\n")
        with open(os.path.join(st.cir_dir, "presence.csv"), "w") as f:
            f.write("time_epoch,time,n,presence,dist_cm,aoa_deg\n")
    
    output("############################################################################")
    output("File Name: " + os.path.basename(__file__))
    output("Date Time: " + str(datetime.now()))
    output("############################################################################")
    output("Port:" + cfg.com_port + "   Timestamp:" + str(cfg.is_timestamp) + "   IPC:" + str(cfg.is_ipc) + \
           "   CIR save:" + str(cfg.is_cir_save) + "   send_cir_ntf:" + str(cfg.send_cir_ntf))
    if (cfg.is_cir_save):
        output("CIR directory: " + st.cir_dir)
    
    output("Configure serial port...")
    serial_port_configure()
    output("Serial port configured")
    
    output("Start adding commands to the queue...")
    st.command_queue.put(UWB_INIT_BOARD_VARIANT)
    st.command_queue.put(UWB_RESET_DEVICE)
    st.command_queue.put([0x20, 0x02, 0x00, 0x00])
    st.command_queue.put(UWB_CORE_SET_CONFIG)
    st.command_queue.put(UWB_CORE_SET_ANTENNAS_DEFINE)

    # SR250 CALIBRATION
    st.command_queue.put(UWB_CORE_SET_PDOA_CALIB_TABLE_DEFINE)
    st.command_queue.put(UWB_SET_CALIBRATION_AOA_ANTENNAS_PDOA_CALIB_PAIR1_CH9)
    st.command_queue.put(UWB_SET_CALIBRATION_AOA_ANTENNAS_PDOA_CALIB_PAIR2_CH9)
    st.command_queue.put(UWB_SET_CALIBRATION_PDOA_OFFSET_CALIB_CH9)
    st.command_queue.put(UWB_SET_CALIBRATION_AOA_THRESHOLD_PDOA_CH9)
    st.command_queue.put(UWB_SET_CALIBRATION_RX_ANT_DELAY_CALIB_CH9)

    # UWB SESSION CONFIG
    st.command_queue.put(UWB_SESSION_INIT_RADAR)
    st.command_queue.put(UWB_SESSION_SET_VENDOR_APP_CONFIG_NXP)
    st.command_queue.put(UWB_SESSION_SET_APP_CONFIG)

    st.command_queue.put(UWB_RANGE_START)
    st.command_queue.put(UWB_RANGE_STOP)
    st.command_queue.put(UWB_RANGE_SESSION_DEINIT)
    
    output("adding commands to the queue completed")
    
    output("Start processing...")
    start_processing()
    
    if (cfg.is_cir_save and len(st.cir_all) > 0):
        lens = set(len(c) for c in st.cir_all)
        if (len(lens) == 1):
            np.save(os.path.join(st.cir_dir, "cir_all.npy"), np.array(st.cir_all))
        else:
            m = min(lens)
            np.save(os.path.join(st.cir_dir, "cir_all.npy"), np.array([c[:m] for c in st.cir_all]))
        print("Saved %d CIRs to %s" % (len(st.cir_all), os.path.join(st.cir_dir, "cir_all.npy")))
        shape0 = st.cir_clean_all[0].shape
        np.save(os.path.join(st.cir_dir, "cir_clean_all.npy"), np.array([c for c in st.cir_clean_all if c.shape == shape0]))
        try:
            save_dist_aoa()
        except Exception as e:
            print("Could not save dist/AoA: %s" % e)
        try:
            save_ml_csv()
        except Exception as e:
            print("Could not save ML CSV: %s" % e)
        try:
            save_presence_log()
        except Exception as e:
            print("Could not save presence log: %s" % e)
        try:
            save_final_averages()
        except Exception as e:
            print("Could not save averages: %s" % e)
    
    output("Processing finished")


if __name__ == "__main__":
    main()
