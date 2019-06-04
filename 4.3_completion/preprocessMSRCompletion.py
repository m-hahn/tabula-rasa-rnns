from paths import MSR_COMP_HOME

import os
import sys
import codecs

path = MSR_COMP_HOME+"//Holmes_Training_Data/"
with open(MSR_COMP_HOME+"//holmes_training_data.txt", "w") as outFile:
 for fileName in os.listdir(path):
   print(path+fileName)
   try:
    with codecs.open(path+fileName, "r",encoding='utf-8', errors='replace') as inFile:
#    with open(path+fileName, "r") as inFile:
       data = inFile.read().strip()
       endOfHeader = data.find("*END*THE SMALL PRINT!")
       if endOfHeader == -1:
          endOfHeader = data.find("ENDTHE SMALL")
       assert endOfHeader > -1
       endOfHeader += 60
#       try:
#        try:
#         endOfText = data.index("End of The Project Gutenberg Etext of")
#       except ValueError:
#         endOfText = data.index("End of Project Gutenberg")
       endOfText = len(data) - 80
       #print((endOfHeader, endOfText))
       trimmed = data[endOfHeader:endOfText]
       ratio = (float(len(trimmed))/len(data))
       if ratio < 0.9:
         print(ratio)
       trimmed = trimmed.split("\r\n\r\n")
       for line in trimmed:
          line = line.replace("\r\n", " ")
          if len(line) > 0:
             print(line, file=outFile)       
   except UnicodeDecodeError:
        print("ERROR")
 
 
 

