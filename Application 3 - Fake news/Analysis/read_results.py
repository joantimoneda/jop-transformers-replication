import re
import os
import pandas as pd
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
os.chdir(SCRIPT_DIR)

######
## ROBERTA-COVID
######

files = os.listdir("Pretrained models/roberta-covid")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/roberta-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta
pd.DataFrame(f1_scores).std()

# Recall:
recall_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/roberta-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_roberta = pd.DataFrame(recall_scores).mean()
recall_roberta

# Precision:
precision_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/roberta-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_roberta = pd.DataFrame(precision_scores).mean()
precision_roberta


########
## BERT-COVID
########

files = os.listdir("Pretrained models/bert-covid")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/bert-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_bert = pd.DataFrame(f1_scores).mean()
results_bert
pd.DataFrame(f1_scores).std()

# Recall
recall_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/bert-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_bert = pd.DataFrame(recall_scores).mean()
recall_bert

# Precision:
precision_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/bert-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_bert = pd.DataFrame(precision_scores).mean()
precision_bert


########
## DEBERTA-COVID
########

files = os.listdir("Pretrained models/deberta-covid")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/deberta-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_deberta = pd.DataFrame(f1_scores).mean()
results_deberta
pd.DataFrame(f1_scores).std()

# Recall
recall_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/deberta-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_deberta = pd.DataFrame(recall_scores).mean()
recall_deberta

# Precision:
precision_scores = []
for i in range(0, len(files)):
    file = open("Pretrained models/deberta-covid/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_deberta = pd.DataFrame(precision_scores).mean()
precision_deberta


########
## ROBERTA ORIGINAL
########

files = os.listdir("Original models/roberta")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("Original models/roberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta
pd.DataFrame(f1_scores).std()

# Recall:
recall_scores = []
for i in range(0, len(files)):
    file = open("Original models/roberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_roberta = pd.DataFrame(recall_scores).mean()
recall_roberta

# Precision:
precision_scores = []
for i in range(0, len(files)):
    file = open("Original models/roberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_roberta = pd.DataFrame(precision_scores).mean()
precision_roberta


########
## BERT ORIGINAL
########

files = os.listdir("Original models/bert")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("Original models/bert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_bert = pd.DataFrame(f1_scores).mean()
results_bert
pd.DataFrame(f1_scores).std()

# Recall
recall_scores = []
for i in range(0, len(files)):
    file = open("Original models/bert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_bert = pd.DataFrame(recall_scores).mean()
recall_bert

# Precision:
precision_scores = []
for i in range(0, len(files)):
    file = open("Original models/bert/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_bert = pd.DataFrame(precision_scores).mean()
precision_bert

########
## DEBERTA ORIGINAL
########

files = os.listdir("Original models/deberta")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("Original models/deberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_deberta = pd.DataFrame(f1_scores).mean()
results_deberta
pd.DataFrame(f1_scores).std()

# Recall
recall_scores = []
for i in range(0, len(files)):
    file = open("Original models/deberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("recall\"", f1)]
    recall_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(recall_scores)
recall_deberta = pd.DataFrame(recall_scores).mean()
recall_deberta

# Precision:
precision_scores = []
for i in range(0, len(files)):
    file = open("Original models/deberta/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("prec\"", f1)]
    precision_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(precision_scores)
precision_deberta = pd.DataFrame(precision_scores).mean()
precision_deberta