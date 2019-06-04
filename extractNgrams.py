from paths import WIKIPEDIA_HOME
import sys

length = int(sys.argv[1])

import collections

queue = collections.deque(maxlen=length)

counter = 0
with open(WIKIPEDIA_HOME+"german-train-tagged.txt", "r") as inFile:
 for line in inFile:
   counter += 1
   ind = line.find("\t")
   if ind >= 0:
     queue.append(line[:ind].lower())
#   print(" ".join(queue))
   if counter % 100000 == 0:
       print(float(counter)/819597764)



