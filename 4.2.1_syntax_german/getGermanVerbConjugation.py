

from corpusIterator import CorpusIterator
forSingSum = 0
forPlurSum = 0
counter = 0


verbs = set()

singularsEqualToPlurals = {}
training = CorpusIterator("German", partition="train", storeMorph=True, removePunctuation=True, lowerCaseLemmas=True)
for sentence in training.iterator():
  for line in sentence:
     if line["posUni"] == "VERB":
        if "|" in line["lemma"]:
            continue
        verbs.add(line["lemma"])

#print(verbs)

import urllib.request
import requests
import lxml
from lxml import etree



def recordTable(table, tableIndices):
     table = [x for x in table if len("".join(x)) > 0 and len(x) > 2]
     if len(table) > 2:
         print("################## "+tableIndices[0])
         index = tableIndices[0]
         print("\n".join(["\t".join(x) for x in table]))
        # if "Infinitiv" in index or "Imperativ" in index:
        #       _ = 0
        # else:
        #     assert len(table) in [6, 7, 8]
         del tableIndices[0]
         lengths = [len(x) for x in table]

for verb in verbs:
  print("// "+verb)
  if not verb.endswith("n"):
      continue
  path = "https://de.wiktionary.org/w/index.php?title=Flexion:"+verb+"&printable=yes"
  #print(path)
  r=requests.get(path)
  try:
     root = etree.fromstring(r.text)
  except lxml.etree.XMLSyntaxError:
      #print("SyntaxError")
      continue
  context = etree.iterwalk(root, events=("start", "end"))
  stack = []
#  print(verb)
  tableIndices = ["Infinitive", "erweiterte Infinitive" , "Imperativ", "Praesens", "Praeteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]

  startedConjugation = False

  for action, elem in context:
     if len(tableIndices) == 0:
        #print("BREAK")
        break
     values = elem.values()
#     if "toctext" in values:
#         context.skip_subtree()
#         continue
#     print(action)
#     print(elem.tag)
# #    print(dir(elem))
#     print(elem.sourceline)
#     print("TAIL "+str(elem.tail).strip())
#     print(elem.text)
#     print(elem.values())
     if "mw-headline" in values:
         if elem.text is not None and "(Konjugation)" in elem.text:
             startedConjugation = True
#         elif startedConjugation:
#             if elem.text == "Imperativ":
#                 while tableIndices[0] == "erweiterte Infinitive" or tableIndices[0] == "Infinitive":
#                    del tableIndices[0]
#                 assert tableIndices[0] == "Imperativ", tableIndices
#             elif elem.text == "Infinitive and Partizipien":
#                assert tableIndices[0] == "Infinitive", tableIndices
#             elif elem.text == "Indikativ und Konjunktiv":
#                assert tableIndices[0] == "Praesens", tableIndices
#             else:
#                 #print(elem.text)
#                _ = 0
#
##             print("QUIT")
# #            quit()
#         else:
 #            quit()
#     if elem.text is not None and "Konjunktiv" in elem.text:
 #         print("END")
  #        quit()
     if startedConjugation and action == "start" and elem.tag in ["tbody", "td", "tr"]:
       stack.append((elem.tag, [], []))
       if elem.text is not None:
          stack[-1][1].append(elem.text.strip())
       if elem.tail is not None:
          stack[-1][1].append(elem.tail.strip())

     elif startedConjugation and action =="end" and elem.tag in ["tbody", "td", "tr"]:
#        if stack[-1].text is not None and "Konjunktiv" in stack[-1].text:
 #           print(stack)
  #          quit()
#        if len(stack[-1][1]) > 0:
 #          print(stack[-1])
        lastOne = stack[-1]
        del stack[-1]
        if lastOne[0] == "td":
            assert len(stack) > 0
            assert stack[-1][0] == "tr"
            stack[-1][2].append(" ".join(lastOne[1]).strip())
        elif lastOne[0] == "tr":
             assert len(stack) == 1, stack
             assert stack[0][0] == "tbody"
             stack[0][2].append(lastOne[2])
             if len(lastOne[2]) < 3:
               lastOne = stack[-1]
               del stack[-1]
               assert len(stack) == 0
               stack.append(("tbody", [], []))
               recordTable(lastOne[2], tableIndices)
                          
        else:
             recordTable(lastOne[2], tableIndices)
#             print("\n".join([str(x) for x in lastOne[2]]))
     elif startedConjugation and action == "start" and len(stack) > 0:
       if elem.text is not None:
          stack[-1][1].append(elem.text.strip())
       if elem.tail is not None:
          stack[-1][1].append(elem.tail.strip())
#       stack[-1][1].append(elem.sourceline)
#       for val in values:
 #          stack[-1][1].append(val)
#  if len(tableIndices) > 0:
 #    print("ERROR")
#  assert len(tableIndices) == 0
#  response = urllib.request.urlopen(path)
 # data = response.read()
  #text = data.decode("utf-8")


