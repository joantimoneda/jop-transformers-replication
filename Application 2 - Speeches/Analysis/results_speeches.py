import re
import os
import pandas as pd
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR / "Raw data results")

######
## XLM-R
######

files = os.listdir("xlmr")
files = [f for f in files if re.search("speechtype", f)]
files 

f1_scores = []
for i in range(0, 3):
    file = open("xlmr/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_xlmr = pd.DataFrame(f1_scores).mean()
sd_xlmr = pd.DataFrame(f1_scores).std()
results_xlmr
sd_xlmr

# Recall
recall_scores = []
for i in range(0, 10):
    file = open("xlmr/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_xlmr = pd.DataFrame(recall_scores).mean()
recall_xlmr

# Precision:
precision_scores = []
for i in range(0, 10):
    file = open("xlmr/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_xlmr = pd.DataFrame(precision_scores).mean()
precision_xlmr


######
## mBERT
######

files = os.listdir("mbert")
files = [f for f in files if re.search("speechtype", f)]

f1_scores = []
for i in range(0, 3):
    file = open("mbert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_mbert = pd.DataFrame(f1_scores).mean()
sd_mbert = pd.DataFrame(f1_scores).std()
results_mbert
sd_mbert

# Recall
recall_scores = []
for i in range(0, 10):
    file = open("mbert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_mbert = pd.DataFrame(recall_scores).mean()
recall_mbert

# Precision:
precision_scores = []
for i in range(0, 10):
    file = open("xlmr/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_xlmr = pd.DataFrame(precision_scores).mean()
precision_xlmr

######
## mDEBERTA
######

files = os.listdir("mdeberta")
files = [f for f in files if re.search("speechtype", f)]

f1_scores = []
for i in range(0, 3):
    file = open("mdeberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
f1_scores
pd.DataFrame(f1_scores)
results_mdeberta = pd.DataFrame(f1_scores).mean()
sd_mdeberta = pd.DataFrame(f1_scores).std()
results_mdeberta
sd_mdeberta

# Recall
recall_scores = []
for i in range(0, 10):
    file = open("mdeberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_mdeberta = pd.DataFrame(recall_scores).mean()
recall_mdeberta

# Precision:
precision_scores = []
for i in range(0, 10):
    file = open("mdeberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_mdeberta = pd.DataFrame(precision_scores).mean()
precision_mdeberta

