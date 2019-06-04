from paths import MSR_COMP_HOME
import random
 

def load(language, partition):
    assert language == "english"
    chunk = []
    path = MSR_COMP_HOME+"//holmes-tokenized"
    with open(path+"-"+partition+".txt", "r") as inFile:
     for line in inFile:
      index = line.find("\t")
      if index == -1:
          continue
      word = line[:index]
      chunk.append(word.lower())
      if len(chunk) > 10000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
    yield chunk

def training(language):
  return load(language, "train")
def dev(language):
  return load(language, "valid")


