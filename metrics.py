from copy import deepcopy

import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        #x = deepcopy(predictions)
        #y = deepcopy(labels)
        x = predictions
        y = labels
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        #x = deepcopy(predictions)
        #y = deepcopy(labels)
        x = predictions
        y = labels
        return torch.mean((x - y) ** 2)

'''

with open("out.txt", "r") as f:
    xx = f.readlines()

pred = []
org = []
for x in xx:
    x = x.strip().split("\t")
    pred.append(float(x[0]))
    org.append(float(x[1]))
pred = torch.FloatTensor(pred)
org = torch.FloatTensor(org)

print(pred, org)
metrics = Metrics(5)

print(metrics.pearson(pred, org), metrics.mse(pred, org))
'''