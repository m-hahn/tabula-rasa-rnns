 
  
  
# based on https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
  
  
import subprocess
import random
  
from math import exp
  
  
import random
  
  
  
import sys
  
language = sys.argv[1]
version = "char-lm-ud.py"


 
  
import numpy as np
  
  
 
  
bounds = []



bounds.append(["batchSize", int] + [32])
bounds.append(["char_embedding_size", int, 100])
bounds.append(["hidden_dim", int, 1024])
bounds.append(["layer_num", int, 1])
bounds.append(["weight_dropout_in", float] + [0.1])
bounds.append(["weight_dropout_hidden", float] + [0.35])
bounds.append(["char_dropout_prob", float] + [0.0])
bounds.append(["char_noise_prob",  float] + [0.01])
bounds.append(["learning_rate", float, 0.2])
#bounds.append(["momentum", type = float, 0.0, 0.5, 0.9])


#x0=[0.5] * len(names)
  
values = [x[2:] for x in bounds]
names = [x[0] for x in bounds]
  
import random
  
 
  
def extractArguments(x):
   result = []
   result.append("--language")
   result.append(language)
   for i in range(len(bounds)):
      result.append("--"+bounds[i][0])
      result.append(x[i])
   return result
  
import os
import subprocess
  
import time
nextPoint = [random.choice(values[i]) for i in range(len(bounds))]
command = [str(x) for x in (["python", version] + extractArguments(nextPoint))]
print(" ".join(command))

