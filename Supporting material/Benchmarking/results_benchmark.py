import re
import os
import pandas as pd
import numpy as np
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

# ROBERTA:
files = os.listdir("ROBERTA")
files = [f for f in files if re.search("benchmark", f)]
file = open("RoBERTa/" + files[0], "r")
f1 = file.read().split(':')[6]
f1 = f1.split('[')[1]
f1 = f1.split(']')[0]
f1 = f1.split(',')
f1
np.mean([float(x) for x in f1]) # 0.602


# BERT:
files = os.listdir("BERT")
files = [f for f in files if re.search("benchmark", f)]
file = open("BERT/" + files[0], "r")
f1 = file.read().split(':')[6]
f1 = f1.split('[')[1]
f1 = f1.split(']')[0]
f1 = f1.split(',')
f1
np.mean([float(x) for x in f1]) # 0.505

