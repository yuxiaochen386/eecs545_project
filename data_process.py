import numpy as np
import math
def getdata(FileName):
    f = open(FileName, 'r')
    lines = f.readlines()
    labels = np.zeros((6,len(lines)))
    features = np.zeros((294,len(lines)))
    for i in range(len(lines)):
        line = lines[i].split(' ')
        label = line[0].split(',')
        for j in range(len(label)):
            labels[int(label[j]),i] = 1
        feature = line[1:]
        for j in range(len(feature)):
            features[j,i] = float(feature[j].split(':')[1])
    return features, labels.transpose()

def unlabeldata(labels, rate):
    t = labels.size
    newlabels = np.copy(labels)
    samples = np.random.choice(t,int(math.floor(rate*t)))
    for i in range(len(samples)):
        c = int(math.floor(samples[i]/labels.shape[0]))
        r = int(math.fmod(samples[i],labels.shape[0]))
        newlabels[r,c] = -1
    return newlabels
