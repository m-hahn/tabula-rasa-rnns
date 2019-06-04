import os

for filename in os.listdir("."):
 if filename.startswith("fix"):
    continue
 if filename.endswith(".py"):
   print(filename)
   with open(filename, "r") as inFile:
     data=inFile.read()
#   data = data.replace("WIKIPEDIA_HOME+\"/german-wiki-word-vocab.txt\"", "\"vocabularies/german-wiki-word-vocab-10000.txt\"")
#   data = data.replace("WIKIPEDIA_HOME+\"/itwiki/italian-wiki-word-vocab.txt\"", "\"vocabularies/italian-wiki-word-vocab-10000.txt\"")
#   data = data.replace("WIKIPEDIA_HOME+\"/english-wiki-word-vocab.txt\"", "\"vocabularies/english-wiki-word-vocab-10000.txt\"")
#   data = data.replace("WIKIPEDIA_HOME+\"enwiki/english-wiki-word-vocab.txt\"", "\"vocabularies/english-wiki-word-vocab-10000.txt\"")
   data = data.replace("wiki-word-vocab-10000", "wiki-word-vocab-50000")

   with open(filename, "w") as inFile:
      inFile.write(data) #print(data, file=inFile)
   
#   quit()
