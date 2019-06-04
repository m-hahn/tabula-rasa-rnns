from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition, removeMarkup=True):
  if language == "italian":
    path = WIKIPEDIA_HOME+"/itwiki-"+partition+"-tagged.txt"
  elif language == "english":
    path = WIKIPEDIA_HOME+"/english-"+partition+"-tagged.txt"
  elif language == "german":
    path = WIKIPEDIA_HOME+""+language+"-"+partition+"-tagged.txt"
  else:
    assert False
  chunk = []
  with open(path, "r") as inFile:
    for line in inFile:
      index = line.find("\t")
      if index == -1:
        if removeMarkup:
          continue
        else:
          index = len(line)-1
      word = line[:index]
      chunk.append(word.lower())
      if len(chunk) > 40000:
      #   random.shuffle(chunk)
         yield chunk
         chunk = []
  yield chunk

def training(language):
  return load(language, "train")
#   with open(WIKIPEDIA_HOME+""+language+"-train.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
def dev(language, removeMarkup=True):
  return load(language, "valid", removeMarkup=removeMarkup)

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)

#   with open(WIKIPEDIA_HOME+""+language+"-valid.txt", "r") as inFile:
#     data = inFile.read().strip().lower().split("\n")
#     print("Shuffling")
#     random.shuffle(data)
#     print("Finished shuffling")
#     return "".join(data)
#

#     for line in data:
#        yield line


