import os
import sys

language = sys.argv[1]

models = []

path = "/checkpoint/mhahn/"
for fileName in os.listdir(path):
   if fileName.startswith(language+"_char-lm") and "bptt-2.py" in fileName and not fileName.endswith(".tar"):
    try:
      with open(path+fileName, "r") as inFile:
          data = inFile.read().strip().split("\n")
          models.append((float(data[0].split(" ")[-1]), data[2]))
    except IndexError:
       print(fileName)

models = sorted(models, key=lambda x:x[0])
for m in models:
  print(m)


# (1.0832216289940846, "Namespace(batchSize=128, char_dropout_prob=0.0, char_embedding_size=100, char_noise_prob=0.0, hidden_dim=1024, language='italian', layer_num=2, learning_rate=3.0, load_from=None, lr_decay=0.98, myID=800916881, save_to='wiki-italian-nospaces-bptt-800916881', sequence_length=50, verbose=False, weight_dropout_hidden=0.0, weight_dropout_in=0.0)")


