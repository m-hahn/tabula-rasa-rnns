import os
import subprocess

files = os.listdir("stimuli/")
relevant = [x for x in files if not x.endswith("_counts.txt") and x+"_counts.txt" not in files]
for name in relevant:
  print(name)
  subprocess.Popen(["python", "extractNgramsMatch.py", "stimuli/"+name])


