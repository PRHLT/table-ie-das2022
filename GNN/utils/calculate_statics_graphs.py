import sys, glob, os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error

fname = sys.argv[1]
# t = 0.5
t = float(sys.argv[2])
fname = os.path.abspath(fname)
gt, hyp = [], []
with open(fname, "r") as f:
    lines = f.readlines()[1:]
    for line in lines:
        line = line.split(" ")
        gt.append(int(line[1]))
        hyp.append(float(line[2]))
    gt = np.array(gt, dtype=int)
    hyp = np.exp(hyp)

hyp = np.where(hyp>t, 1, 0)
accuracy = accuracy_score(hyp, gt)
print("Accuracy: {} \n".format(accuracy))
for name in ['binary', 'micro', 'macro']:
    precision = precision_score(hyp, gt, average=name)
    recall = recall_score(hyp, gt, average=name)
    f1 = f1_score(hyp, gt, average=name)
    print("Precision on {} - {}".format(name, precision))
    print("Recall on {} - {}".format(name, recall))
    print("F1 on {} - {}".format(name, f1))
    print()
print("Classified: {}/{} ({} missclassified)".format(np.sum(hyp==gt), len(gt), len(gt)-np.sum(hyp==gt)))
FP = np.sum(np.logical_and(hyp == 1, gt == 0))
FN = np.sum(np.logical_and(hyp == 0, gt == 1))
print("False positve (joining COLS) {}".format(FP))
print("False negative  {}".format(FN))