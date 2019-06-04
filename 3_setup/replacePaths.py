# "/checkpoint/mhahn/char-vocab-wiki-"+args.language, "r") as inFile:               CHAR_VOCAB_HOME
# "/checkpoint/mhahn/"+args.load_from+".pth.tar"                                    MODELS_HOME
# "/checkpoint/mhahn/"+args.save_to+".pth.tar"                                    MODELS_HOME
#  with open("/checkpoint/mhahn/"+args.language+"_"+__file__+"_"+str(args.myID), "w") as outFile:    LOG_HOME



import os

files = os.listdir(".")

for name in files:
    if name == "replacePaths.py" or name == "config.py" or name == "paths.py":
        continue
    if name.endswith(".py"):
        required = set()
        with open(name, "r") as inFile:
            content = inFile.read().split("\n")
        for i in range(len(content)):
            line = content[i]
            if "/checkpoint/mhahn/char-vocab-wiki-" in line:
                line = line.replace('"/checkpoint/mhahn/char-vocab-wiki-', 'CHAR_VOCAB_HOME+"/char-vocab-wiki-')
                required.add("CHAR_VOCAB_HOME")
            if '"/checkpoint/mhahn/"+args.save_to' in line:
                line = line.replace('"/checkpoint/mhahn/"+args.save_to', 'MODELS_HOME+"/"+args.save_to')
                required.add("MODELS_HOME")
            if '"/checkpoint/mhahn/"+args.load_from' in line:
                line = line.replace('"/checkpoint/mhahn/"+args.load_from', 'MODELS_HOME+"/"+args.load_from')
                required.add("MODELS_HOME")
            if '"/checkpoint/mhahn/"+args.language+"_"+__file__+"_"+str(args.myID)' in line:
                line = line.replace('"/checkpoint/mhahn/"+args.language+"_"+__file__+"_"+str(args.myID)', 'LOG_HOME+"/"+args.language+"_"+__file__+"_"+str(args.myID)')
                required.add("LOG_HOME")
            if '"/private/home/mhahn/data/similarity/msr-completion' in line:
                line = line.replace('"/private/home/mhahn/data/similarity/msr-completion', 'MSR_COMP_HOME+"/')
                required.add("MSR_COMP_HOME")
            if 'plt.savefig("/checkpoint/mhahn/' in line:
                line = line.replace('plt.savefig("/checkpoint/mhahn/', 'plt.savefig(FIGURES_HOME+"/')
                required.add("FIGURES_HOME")
            if '"/private/home/mhahn/data/WIKIPEDIA/"+args.language+"-vocab.txt' in line:
                line = line.replace('"/private/home/mhahn/data/WIKIPEDIA/"+args.language+"-vocab.txt', 'WIKIPEDIA_HOME+"/"+args.language+"-vocab.txt')
                required.add("WIKIPEDIA_HOME")
            if '"/private/home/mhahn/data/WIKIPEDIA/german-wiki-word-vocab.txt' in line:
                line = line.replace('"/private/home/mhahn/data/WIKIPEDIA/german-wiki-word-vocab.txt', 'WIKIPEDIA_HOME+"/german-wiki-word-vocab.txt')
                required.add("WIKIPEDIA_HOME")
            if '"/private/home/mhahn/data/WIKIPEDIA/itwiki/italian-wiki-word-vocab.txt' in line:
                line = line.replace('"/private/home/mhahn/data/WIKIPEDIA/itwiki/italian-wiki-word-vocab.txt', 'WIKIPEDIA_HOME+"/itwiki/italian-wiki-word-vocab.txt')
                required.add("WIKIPEDIA_HOME")
            if '"/private/home/mhahn/data/WIKIPEDIA/' in line:
                line = line.replace('"/private/home/mhahn/data/WIKIPEDIA/', 'WIKIPEDIA_HOME+"')
                required.add("WIKIPEDIA_HOME")
            if '"/private/home/mhahn//data/WIKIPEDIA/' in line:
                line = line.replace('"/private/home/mhahn//data/WIKIPEDIA/', 'WIKIPEDIA_HOME+"')
                required.add("WIKIPEDIA_HOME")
            content[i] = line

#                print(line)

        #    if "mhahn" in line:
         #       print((name,line))

        if len(required) > 0:
            with open(name, "w") as outFile:
                for var in required:
                   print >> outFile, "from paths import "+var
                for line in content:
                   print >> outFile, line
            print(name)
#            quit()

