from paths import MSR_COMP_HOME
import random
 

def load(language, partition):
    assert language == "english"
    chunks = []
    path = MSR_COMP_HOME+"//holmes"
    with open(path+"-"+partition+".txt", "r") as inFile:
      for line in inFile:
        chunks.append(line.strip().lower())
        if len(chunks) > 10000:
           random.shuffle(chunks)
           yield "".join(chunks)
           chunks = []
    yield "".join(chunks)

def training(language):
  return load(language, "train")
def dev(language):
  return load(language, "valid")


