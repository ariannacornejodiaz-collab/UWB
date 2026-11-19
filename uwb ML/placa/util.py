"""Utilidades comunes: salida por consola o por el fichero IPC."""
import config as cfg
import state as st


def output(string):
    
    if (cfg.is_ipc):
        if ((st.file_ipc is not None) and (not st.file_ipc.closed) and (st.file_ipc.writable())):
            st.file_ipc.write(string + "\n")
            return True
    else:
        try:
            print(string)
        except:
            pass
        
    return False
