import re
import os
import pandas as pd
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR / "Raw data results")

######
## RoBERTa
######

files = os.listdir("roberta")
files = [f for f in files if re.search("civility", f)]
files

# F1_scores:
f1_scores = []
for i in range(0, 10):
    file = open("roberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
sd_roberta = pd.DataFrame(f1_scores).std()

results_roberta
sd_roberta

# Recall:
recall_scores = []
for i in range(0, 10):
    file = open("roberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_roberta = pd.DataFrame(recall_scores).mean()
recall_roberta

# Precision:
precision_scores = []
for i in range(0, 10):
    file = open("roberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_roberta = pd.DataFrame(precision_scores).mean()
precision_roberta





########
## BERT
########
files = os.listdir("bert")
files = [f for f in files if re.search("civility", f)]
files


# F1 score
f1_scores = []
for i in range(0, 3):
    file = open("bert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_bert = pd.DataFrame(f1_scores).mean()
sd_bert = pd.DataFrame(f1_scores).std()
results_bert
sd_bert


# Recall
recall_scores = []
for i in range(0, 10):
    file = open("bert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_bert = pd.DataFrame(recall_scores).mean()
recall_bert

# Precision:
precision_scores = []
for i in range(0, 10):
    file = open("bert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_bert = pd.DataFrame(precision_scores).mean()
precision_bert

######
## DEBERTA
######

files = os.listdir("deberta")
files = [f for f in files if re.search("civility", f)]

f1_scores = []
for i in range(0, 3):
    file = open("deberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
f1_scores
pd.DataFrame(f1_scores)
results_deberta = pd.DataFrame(f1_scores).mean()
sd_deberta = pd.DataFrame(f1_scores).std()
results_deberta
sd_deberta

# Recall
recall_scores = []
for i in range(0, 10):
    file = open("deberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_deberta = pd.DataFrame(recall_scores).mean()
recall_deberta

# Precision:
precision_scores = []
for i in range(0, 10):
    file = open("deberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_deberta = pd.DataFrame(precision_scores).mean()
precision_deberta
