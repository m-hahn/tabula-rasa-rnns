from paths import FIGURES_HOME
import os
basePath = "/checkpoint/mhahn/trajectories/"


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

#for language in ["Japanese", "Sesotho", "Indonesian"]:
if True:
   prefix = "char-lm-ud-stationary-separate-bidir-with-spaces-probe-baseline-prediction-wiki.py_wiki-german-nospaces-bugfix-checkpoints_CHECKPOINT"
   files = [x for x in os.listdir(basePath) if x.startswith(prefix)]
   print(basePath+"/"+prefix)
   files = zip(files, range(len(files))) #(x, int(x[len(prefix):])) for x in files]
   files = sorted(files, key=lambda x:x[1])
   data = [[[None for _ in files] for _ in range(3)] for _ in range(3)]
   genders = ["m", "f", "n"]
   articles = ["der", "die", "das"]
   for filename, number in files:
       with open(basePath+"/"+filename, "r") as inFile:
           dataNew = [[float(y) for y in x.split("\t")] for x in inFile.read().strip().split("\n")]
           for i in range(3):
             for j in range(3):
                 data[i][j][number] = dataNew[i][j]
   print(data)
   dataByGroup = {}
   for i, gender in enumerate(genders):
     for j, article in enumerate(articles):
        values = data[i][j]
        plt.plot(range(len(values)), values, label=article)
     plt.legend()
     plt.show()
     plt.savefig(FIGURES_HOME+"/figures/german_gender_"+gender+".png")
     plt.close()


