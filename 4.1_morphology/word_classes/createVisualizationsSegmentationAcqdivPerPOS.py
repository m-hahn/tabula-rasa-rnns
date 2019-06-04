from config import TRAJECTORIES_HOME, VISUALIZATIONS_HOME


import os
basePath = TRAJECTORIES_HOME

groups = {"words" : ["undersegmented", "over", "under", "missegmented", "agreement"], "boundaries" : ["boundary_precision", "boundary_recall", "boundary_accuracy"], "tokens" : ["token_precision", "token_recall"], "lexical" : ["lexical_recall", "lexical_precision"]}

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np

for language in ["Japanese", "Sesotho", "Indonesian"]:
   prefix = "lm-acqdiv-segmentation-analyses-morph-pos.py_acqdiv-"+language.lower()+"-optimized_EPOCH_"
   files = [x for x in os.listdir(basePath) if x.startswith(prefix)]
   print(basePath+"/"+prefix)
   files = zip(files, range(len(files))) #(x, int(x[len(prefix):])) for x in files]
   files = sorted(files, key=lambda x:x[1])
   data = {}
   for filename, number in files:
       with open(basePath+"/"+filename, "r") as inFile:
           dataNew = [x.split("\t") for x in inFile.read().strip().split("\n")]
           for line in dataNew:
              if line[0] == "over":
                 line[0] = "oversegmented"
              if line[0] == "under":
                 line[0]= "undersegmented"
              if line[0] == "agreement":
                 line[0] = "correct"
              if len(line) == 2:
                 line.append("unknown_pos")
              assert len(line) == 3, line
              if line[0] not in data:
                 data[line[0]] = {} 
              if line[2] not in data[line[0]]:
                  data[line[0]][line[2]] = [None for _ in files]
              data[line[0]][line[2]][number] = float(line[1])
   print(data)
   dataByGroup = {}
   for name, datapoints in data.items():
     for group, values in datapoints.items():
       if None in values:
         print("Filtered Nones")
         continue
       if max(values) > 1.0 and sum(values)/len(values) < 10:
          continue
       if group == "none":
          continue
       if group not in dataByGroup:
           dataByGroup[group] = {}
       if name not in dataByGroup[group]:
           dataByGroup[group][name] = values
       plt.plot(range(len(values)), values, label=group)
     plt.legend()
     plt.show()
     plt.savefig(VISUALIZATIONS_HOME+"/byPOS_"+language+"_"+name+".png")
     plt.close()

   for group, datapoints in dataByGroup.items():
     for name, values in datapoints.items():
       plt.plot(range(len(values)), values, label=name)
     plt.legend()
     plt.show()
     plt.savefig(VISUALIZATIONS_HOME+"/forPOS_"+language+"_"+group+".png")
     plt.close()




