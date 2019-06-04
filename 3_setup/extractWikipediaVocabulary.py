from paths import WIKIPEDIA_HOME
unigrams = {}


if True:
 pathIn = WIKIPEDIA_HOME+"english-train-tagged.txt"
 pathOut = WIKIPEDIA_HOME+"english-wiki-word-vocab.txt"
elif False:
 assert False
 pathIn = WIKIPEDIA_HOME+"german-train-tagged.txt"
 pathOut = "vocabularies/german-wiki-word-vocab-50000.txt"
else:
 assert False
 pathIn = WIKIPEDIA_HOME+"itwiki-train-tagged.txt"
 pathOut = WIKIPEDIA_HOME+"italian-wiki-word-vocab.txt"

import random
with open(pathIn, "r") as inFile:
   for line in inFile:
      line = line[:-1]
      index = line.find("\t")
      if index == -1:
#         print(line)
         continue
      word = line[:index].lower()
      unigrams[word] = unigrams.get(word, 0) + 1
 #     if random.random() > 0.99:
#          break
unigrams = sorted(list(unigrams.items()), key=lambda x:x[1],reverse=True)
with open(pathOut, "w") as outFile:
  for word, count in unigrams:
      print(f"{word}\t{count}", file=outFile)
      

