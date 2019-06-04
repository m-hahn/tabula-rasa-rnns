

# ./segment /checkpoint/mhahn/german-valid-500-exclude-punctuation.txt -w0 -o /checkpoint/mhahn/german-valid-500-exclude-punctuation-segmented-bayesian.txt

import sys

language = sys.argv[1]
partition = sys.argv[2]

infix = "" if language == "german" else "/itwiki/"
name = "german" if language == "german" else "itwiki"

sentences = [0,0]

with open(f"/private/home/mhahn/data/WIKIPEDIA/{infix}{name}-{partition}-tagged.txt", "r") as inFile:
   with open(f"/checkpoint/mhahn/{language}-{partition}-500.txt", "w") as outFile:
      currentLine = []
      currentLineCharCount = 0
      for line in inFile:
          i = line.find("\t")
          if i == -1:
             continue
          word = line[:i].lower()
 #         print(f"#{word}#")
#          print([line[i+1:i+3]])
          currentLine.append(word)
          currentLineCharCount += len(word) + 1
          if word == "." and ((line[i+1:i+3] == "$.") if language == "german" else (line[i+1:i+5] == "SENT")):
              if currentLineCharCount > 450:
                #print("ERROR "+str(currentLineCharCount))
                #print((currentLine))
                sentences[1] += 1
              else:
                print(" ".join(currentLine), file=outFile)
                sentences[0] += 1
                if sentences[0] % 10000 == 0:
                   print(sentences)
              currentLine = []
              currentLineCharCount = 0
print(sentences)
