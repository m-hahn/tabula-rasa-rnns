with open("germanConjugation.txt") as inFile:
   data = inFile.read().strip().split("//")
for verb in data:
   praesens = None
   praeteritum = None
   withParticiple = None
   portion = verb.split("##################")
   assert len(portion) > 0
   if len(portion) == 1: # no tables were found
      continue
   else:
     forms = ["Infinitive", "erweiterte Infinitive" , "Imperativ", "Praesens", "Praeteritum", "Perfekt", "Plusquamperfekt", "Futur I", "Futur II"]
     verbLemma = portion[0].strip()
     del portion[0]
     if len(forms) != len(portion):
        continue
     for form, table in zip(forms, portion):
        table = table.strip().split("\n")
        assert table[0].strip() == form
        del table[0]
        table = [x.split("\t") for x in table]
#        print([len(x) for x in  table])
        if form == "Infinitive":
           assert len(table) >= 3
           table = table[-3:]
           
        elif form == "erweiterte Infinitive":
           assert len(table) >= 3
           table = table[-3:]
        elif form == "Imperativ": #don't care for the experiments
           assert len(table) >= 3
           table = table[-3:]
        else: #if form == "Praesens":
           if len(table) < 6:
              break
              #, table
           table = table[-6:]
           firstColumn = [x[0] for x in table]
           if max([len(x) for x in firstColumn]) == 0:
               table = [x[1:] for x in table]
           assert "Pers" in table[0][0], table
           assert "1." in table[0][0], table
           aktiv = [x[:3] for x in table]
           #print(aktiv)
           assert len(aktiv) == 6
           assert len(aktiv[0]) == 3
           assert "er/sie/es" in aktiv[2][1] or aktiv[2][1] == '—', aktiv
           if form == "Perfekt":
               if "er/sie/es" in aktiv[2][1]:
                    withParticiple = aktiv[2][1].split(" ")[1:]
           elif form == "Praesens":
               praesens = aktiv
           elif form == "Praeteritum":
               praeteritum = aktiv 
   if None in [verbLemma, praesens, praeteritum, withParticiple]:
       continue
   if len(withParticiple) > 2 and "sich" != withParticiple[1]:
        withParticiple = withParticiple[:2] #, withParticiple
   assert len(withParticiple) in [2, 3]
   if withParticiple[0] not in ["ist", "hat"]:
#      print("error")
      continue

   print("###")
   print(verbLemma)
   print("%")
   print("\n".join(["\t".join(x) for x in praesens]))
   print("%")
   assert len(praesens) == 6
   print("\n".join(["\t".join(x) for x in praeteritum]))
   print("%")
   assert len(praeteritum) == 6
   print(" ".join(withParticiple))
        #assert len(set([len(x) for x in table])) == 1, (form,table, [len(x) for x in table])
       # print((form, len(table)))
        #if form == "Infinitive":
         #   p
