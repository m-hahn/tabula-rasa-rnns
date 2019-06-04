
import os
import sys
import random

files = [x for x in os.listdir("../stimuli/") if x.endswith("_counts.txt") and "prep" in x]
for fileName in files:
   counts = [0, 0]
   with open("../stimuli/"+fileName, "r") as inFile:
      data = inFile.read().strip().split("\n")
   for i in range(0, len(data), 2):
     countsHere = [int(x.split("\t")[1]) for x in data[i:i+2]]
     maximal = max(countsHere)
     argmax = [x for x in range(2) if countsHere[x] == maximal]
     chosen= argmax[random.randint(0, len(argmax)-1)]
#     if maximal == 0:
#        print(chosen)
     counts[chosen] += 1
   print(fileName)
#   print(i)
   print(counts)
   print([round(float(x)/sum(counts),2) for x in counts])

