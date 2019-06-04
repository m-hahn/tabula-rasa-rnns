import numpy as np


PATH = "/home/user/CS_SCR/FAIR18/CHECKPOINTS/boundary-neuron/"


#%\begin{figure*}
#%        \includegraphics[width=0.9\textwidth]{figures/{english_wiki-english-nospaces-bptt-282506230_15.txt}.png}
#%        \includegraphics[width=0.9\textwidth]{figures/{italian_wiki-italian-nospaces-bptt-855947412_7.txt}.png}
#%        \includegraphics[width=0.9\textwidth]{figures/{german_wiki-german-nospaces-bptt-910515909_12.txt}.png}
#%        \caption{Behavior of the CNLM `word unit' TODO find a better Italian example?}
#%\end{figure*}

#for i in range(1, 100):
i=12
if True:
  # fileName = "english_wiki-english-nospaces-bptt-282506230_"+str(i)+".txt"
 #  signs = [1, -1, -1, -1, -1]
   
   
   fileName = "german_wiki-german-nospaces-bptt-910515909_"+str(i)+".txt"
   signs = [-1, 1, 1, -1, 1]
   
   
#   fileName = "italian_wiki-italian-nospaces-bptt-855947412_"+str(i)+".txt"
 #  signs = [1, 1, 1, 1, 1]
   
   
   inFile = PATH+fileName
   
   with open(inFile, "r") as inFile:
       data = [x.split(" ") for x in inFile.read().strip().split("\n")]
   
   pos = np.asarray([int(x[0]) for x in data])
   
   
   
   char = [x[1] for x in data]
   ys = []
   for i in range(2,7):
       ys.append([float(x[i]) for x in data])
       ys[-1] = np.asarray(ys[-1])
    #   ys[-1] = ys[-1] - ys[-1].mean()
   #    ys[-1] = ys[-1] / ys[-1].std()
       ys[-1] = ys[-1] * signs[i-2]
   
   boundary = [1 if x[7] != "None" else 0 for x in data]
   
   possibleStarts = [x for x in range(10,40) if boundary[x] == 1]
   START = possibleStarts[0]+1
   END = START+40
   print(START,END)
   
   import matplotlib
   import matplotlib.pyplot as plt
   
   print(ys[0])
   
   fig, ax = plt.subplots()
   for y in ys[:1]:
       ax.plot(pos[START:END]+0.5, y[START:END])
   
   #ax.plot(pos[START:END]+0.5, sum([ys[i][START:END] for i in range(5)]))
   
   
   plt.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.17)
   fig.set_size_inches(10, 1.7)
   #ax.grid(False)
   plt.xticks(pos[START:END], [x.decode("utf-8") for x in char[START:END]])
   
   for i in range(START,END):
       if boundary[i] == 1:
         plt.axvline(x=pos[i]+0.5, color="green")
   
   #ax.set(xlabel='time (s)', ylabel='voltage (mV)',
   #       title='About as simple as it gets, folks')
   ax.grid(False)
   
   fig.savefig(PATH+"/"+fileName+".png")
   plt.show()
   plt.gcf().clear()

   print(PATH+"/"+fileName+".png")
   
   
   
