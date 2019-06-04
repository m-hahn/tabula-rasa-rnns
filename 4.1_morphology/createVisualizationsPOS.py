import os
basePath = "results/pos/" #"/checkpoint/mhahn/pos/"

import sys

language = sys.argv[1]



import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np


matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)

#for language in ["Japanese", "Sesotho", "Indonesian"]:
if True:
   prefix = "char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2.py_wiki-german-nospaces-bugfix-checkpoints_CHECKPOINT"
   files = [x for x in os.listdir(basePath) if language in x]
   print(basePath+"/"+prefix)
   data = {"LSTM" : [], "Autoencoder" : [], "RNN" : [], "WordNLM" : []}
   for filename in files:
       if "withOOV" in filename:
            continue
       with open(basePath+"/"+filename, "r") as inFile:
           dataNew = inFile.read().strip().split("\n")
           trainSize = int(dataNew[1])
           model = ("RNN" if "rnn" in dataNew[2] else ("WordNLM" if "word" in dataNew[2] else "LSTM")) if "bptt" in dataNew[2] else "Autoencoder"
           result = float(dataNew[3])
           data[model].append((trainSize, result))
   
   for model in data:
      data[model] = sorted(data[model], key=lambda x:x[0])
      print(model)
      print(data[model])
   for group, datapoints in data.items():
          datapoints = sorted(datapoints, key=lambda x:x[0])
          plt.plot([x[0] for x in datapoints], [x[1] for x in datapoints], label=group)
   plt.legend()
   plt.show()
   plt.savefig("figures/"+language+"_pos_nouns_verbs.pdf", bbox_inches='tight')
   plt.close()

