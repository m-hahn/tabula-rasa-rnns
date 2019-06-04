
with open("/checkpoint/mhahn/plural-results-wiki-german-nospaces-bptt-910515909.txt", "r") as inFile:
   data = [["model"] + x.split(" ") for x in inFile.read().strip().split("\n")]
with open("/checkpoint/mhahn/plural-results-wiki-autoencoder.txt", "r") as inFile:
   data += [["baseline"] + x.split(" ") for x in inFile.read().strip().split("\n")]

means = []
keys = set([tuple(x[:4]) for x in data])
for key in keys:
   values = [float(x[4]) for x in data if tuple(x[:4]) == tuple(key)]
   meanValue = sum(values)/len(values)
   means.append((key, meanValue))
   print(key, meanValue)

means = dict(means)

points = range(4,100,4)

models = ["model", "baseline"]
curves = set([x[3] for x in keys])
curveYs = {}
for model in models:
  for curve in curves:
     curveY = []
     for point in points:
        key = (model, str(point), "NSE",  curve)
        curveY.append(means[key])
     print(curveY)
     curveYs[key] = curveY




