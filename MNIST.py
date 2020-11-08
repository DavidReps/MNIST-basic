import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier

results = []
classifier = SGDClassifier()
lD = load_digits()

for i in range(50):
    index = random.choices(range(len(lD.data)), k=100)
    subData = lD.data[index,:]
    subTarget = lD.target[index]
    classifier.partial_fit(subData, subTarget, classes=np.unique(lD.target))
    score = classifier.score(lD.data, lD.target)
    results.append(score)

print(results)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Test number")
plt.plot(results)
plt.show()
