from paths import WIKIPEDIA_HOME


from corpusIterator import CorpusIterator
forSingSum = 0
forPlurSum = 0
counter = 0


nouns = set()

singularsEqualToPlurals = {}
training = CorpusIterator("German", partition="train", storeMorph=True, removePunctuation=True, lowerCaseLemmas=False)
for sentence in training.iterator():
  for line in sentence:
     if line["posUni"] == "NOUN":
        if "|" in line["lemma"]:
            continue
        nouns.add(line["lemma"])

titleStart = len("    <title>")

keys = ["Genus"] + [x+" "+y for x in ["Nominativ", "Genitiv", "Dativ", "Akkusativ"] for y in ["Singular", "Plural"]]


def processTable(word, declensionTable):
   if len(declensionTable) == 0:
      return
#   if not declensionTable[0].startswith("|Genus"):
#     print(declensionTable)
 #    return
   #gender = declensionTable[0][7]
   table = {key : [] for key in keys}
   for line in declensionTable:
     keyValue = line.split("=")
     if len(keyValue) != 2:
#         print(keyValue)
         continue
     key, value = tuple(keyValue)
     key = key[1:]
     if key.endswith("*"):
         key = key[:-1]
     if key.endswith(" 1") or key.endswith(" 2"):
         key = key[:-2]
     if key in table:
        table[key].append(value.strip())
     else:
       continue
   #      print(key)
   print("###")
   print(word)
   for key in keys:
      if len(table[key]) > 0:
         print("\t".join([key] + table[key]))

currentPage = None
declensionTable = None
with open(WIKIPEDIA_HOME+"dewiktionary-20180720-pages-articles.xml", "r") as inFile:
  for line in inFile:
    if line == "  <page>\n":
       currentPage = None
    if line.startswith("    <title>"):
       word = line[titleStart:-9]
       if word in nouns:
#          print(word)
          currentPage = word
    elif currentPage is not None:
      if line.startswith("{{Deutsch Substantiv"):
         declensionTable = []
      elif declensionTable is not None:
         if line.startswith("}}"):
              processTable(word, declensionTable)
              declensionTable = None
         else:
             end = False
             if "}}" in line:
                line = line[:line.index("}}")]
                end = True
             declensionTable.append(line)
             if end:
               processTable(word, declensionTable)
               declensionTable = None


