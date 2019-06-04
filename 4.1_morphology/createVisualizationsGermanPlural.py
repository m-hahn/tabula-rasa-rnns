from paths import FIGURES_HOME
import os
basePath = "/checkpoint/mhahn/trajectories/"


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

#for language in ["Japanese", "Sesotho", "Indonesian"]:
if True:
   prefix = "char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki-plurals-2.py_wiki-german-nospaces-bugfix-checkpoints_CHECKPOINT"
   files = [x for x in os.listdir(basePath) if x.startswith(prefix)]
   print(basePath+"/"+prefix)
   files = zip(files, range(len(files))) #(x, int(x[len(prefix):])) for x in files]
   files = sorted(files, key=lambda x:x[1])
   data = {}
   for filename, number in files:
       with open(basePath+"/"+filename, "r") as inFile:
           dataNew = [x.split("\t") for x in inFile.read().strip().split("\n")]
           for line in dataNew:
              train = line[0]
              test = line[1]
              acc = line[2]
              if train not in data:
                  data[train] = {}
              if test not in data[train]:
                 data[train][test] = [None for _ in files]
              data[train][test][number] = float(acc)
   print(data)
   for train, datapoints in data.items():
     for test, values in datapoints.items():
          plt.plot(range(len(values)), values, label=test)
     plt.legend()
     plt.show()
     plt.savefig(FIGURES_HOME+"/figures/german_plural_"+train+".png")
     plt.close()


