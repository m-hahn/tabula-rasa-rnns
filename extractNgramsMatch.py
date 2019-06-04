from paths import WIKIPEDIA_HOME
import sys

#length = int(sys.argv[1])
infile = sys.argv[1]

with open(infile, "r") as inFile:
   patterns = inFile.read().strip().split("\n")

counts = [0 for _ in patterns]

lengths = set([len(x.split(" "))  for x in patterns])
assert len(lengths) == 1, (infile, lengths)
length = list(lengths)[0]

indices = dict(zip(patterns, range(len(patterns))))

import collections

queue = collections.deque(maxlen=length)

counter = 0
with open(WIKIPEDIA_HOME+"german-train-tagged.txt", "r") as inFile:
 for line in inFile:
   counter += 1
   ind = line.find("\t")
   if ind >= 0:
     queue.append(line[:ind].lower())
   string = (" ".join(queue))
   index = indices.get(string, -1)
   if index > -1:
      counts[index] += 1
   if counter % 1000000 == 0:
       print(infile, length, float(counter)/819597764)
   if counter % 10000000 == 0:
       with open(infile+"_counts.txt", "w") as outFile:
          for ngram, count in zip(patterns, counts):
             print(f"{ngram}\t{count}", file=outFile)
with open(infile+"_counts.txt", "w") as outFile:
          for ngram, count in zip(patterns, counts):
             print(f"{ngram}\t{count}", file=outFile)


