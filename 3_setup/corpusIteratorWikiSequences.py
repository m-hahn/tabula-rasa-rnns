# For phonotactic experiments

from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition, sequence, sequences=None):
    chunks = []
    skippedLines = 0
    totalLines = 0
    with open(WIKIPEDIA_HOME+""+language+"-"+partition+".txt", "r") as inFile:
      for line in inFile:
        line = line.strip().lower()
        totalLines += 1
        if sequence is not None and sequence in line.replace(" ",""):
          skippedLines += 1
          if skippedLines % 1000 == 0:
              print("SKIPPED LINES", skippedLines/totalLines)
          continue
        if sequences is not None:
           foundViolation = 0
           for seq in sequences:
#              print(seq)
              if seq is not None and seq in line.replace(" ", ""):
                skippedLines += 1
                if skippedLines % 1000 == 0:
                    print("SKIPPED LINES ", skippedLines/totalLines, seq)
                foundViolation = True
                break
           if foundViolation:
               continue
           assert sequences[0] not in line, (sequences[0], line)
           assert sequences[1] not in line, (sequences[0], line)
        chunks.append(line)
        if len(chunks) > 20000:
           random.shuffle(chunks)
           yield "".join(chunks)
           chunks = []
    yield "".join(chunks)

def training(language, sequence, sequences=None):
  return load(language, "train", sequence, sequences=sequences)


