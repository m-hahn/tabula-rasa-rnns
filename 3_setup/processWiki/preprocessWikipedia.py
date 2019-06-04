path = "/private/home/mhahn/data/WIKIPEDIA/wikiextractor/enwiki/"

import os

dirs = os.listdir(path)
with open(path+"/extracted.txt", "w") as outFile:
  for directory in dirs:
     print(directory)
     if not os.path.isdir(path+"/"+directory):
        continue
     files = os.listdir(path+"/"+directory)
     for filename in files:
       with open(path+"/"+directory+"/"+filename, "r") as inFile:
          for line in inFile:
             if len(line) > 5 and not (line.startswith("<")):
                print(line.strip(), file=outFile)


