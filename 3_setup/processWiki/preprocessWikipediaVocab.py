path = "/private/home/mhahn/data/WIKIPEDIA/wikiextractor/enwiki/"

import os

vocab = {}

dirs = os.listdir(path)
try:
  for directory in dirs:
     print(directory)
     if not os.path.isdir(path+"/"+directory):
        continue
     files = os.listdir(path+"/"+directory)
     for filename in files:
       with open(path+"/"+directory+"/"+filename, "r") as inFile:
          for line in inFile:
             if len(line) > 5 and not (line.startswith("<")):
               for char in line.lower():
                if char != " " and char != "\n":
                  vocab[char] = vocab.get(char, 0) + 1
     print("".join(sorted([x for x in vocab if vocab[x] > 10000])))
except KeyboardInterrupt:
   print(0)
with open(path+"/vocab-char.txt", "w") as outFile:
#  itos = sorted([x for x, y in vocab.items() if y > 10000])
  itos = sorted([x[0] for x in sorted(list(vocab.items()), key=lambda x:x[1], reverse=True)[:60]])
  for char in itos:
      print(char, file=outFile)
#                print(line.strip(), file=outFile)


