import os
import subprocess

path = "/checkpoint/mhahn/jobs/"
files = os.listdir(path)

results = []
for fileName in files:
   if fileName.startswith("search-english-wiki-") and fileName.endswith(".out"):
    try:
      with open(path+fileName, "r") as inFile:
         setup = next(inFile).strip()
         crossEntropies = []
#         if fileName == "search-english-wiki-5306188.out":
#           assert False, 
         if "-words-" in setup:
#            for line in inFile:
#              if line.startswith("tensor("):
#                  crossEntropy = float(line[len("tensor("):line.index(",")])
#                  crossEntropies.append(crossEntropy)
#            if len(crossEntropies) > 100:
 #             performance = sum(crossEntropies[-100:])/100
             # print(fileName)
             # print(setup)
             # print(performance)
  #            results.append((performance, setup))

              arguments = "--"+setup[setup.index("batch"):-1]
              arguments = arguments.replace("load_from=None, ", "")
              arguments = arguments.replace("'", "")
              arguments = arguments.replace(", ", " --")
              arguments = arguments.replace("=", " ")
              arguments = arguments.replace("save_to", "load-from")
              print(arguments)
              p = subprocess.Popen(["python", "char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words-EVAL_DEV.py"] + arguments.split(" "))
              p.wait()
#              char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words-EVAL_DEV.py
    except StopIteration:
            _ = 0

