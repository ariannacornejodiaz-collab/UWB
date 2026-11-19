"""Lanza los cuatro detectores uno tras otro con las mismas opciones y luego compare.py.

    .venv/bin/python train_all.py
    .venv/bin/python train_all.py --data data/real --tune

Acepta las mismas opciones que train_rf.py.
"""
import subprocess
import sys

from uwb.training import parse_args

SCRIPTS = ["train_rf.py", "train_gb.py", "train_svm.py", "train_umbral.py"]

if __name__ == "__main__":
    options = sys.argv[1:]
    args = parse_args("rf", options)
    for script in SCRIPTS:
        subprocess.run([sys.executable, script, *options], check=True)
    subprocess.run([sys.executable, "compare.py", "--results", args.out], check=True)
