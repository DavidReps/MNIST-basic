# Your task is to write a Python program that loads the digits data set, initializes a
# model that is an instance of the sklearn.linear_model.SGDClassifier class, and
# then carries out 50 stochastic gradient descent training iterations, as follows. In each
# iteration, a random subset of 100 data examples (rows of the data set) should be
# selected via the random.choices function; the SGDClassifier.partial_fit method
# should then be used to train the model, using only the mini-batch consisting of the
# 100 data rows selected in that iteration, together with their target labels (once you’ve
# obtained the row numbers to use from choices and placed them in indices, say, you
# can just use the syntax A[indices,:] to extract the corresponding rows of A).
# Use the SGDClassifier.score method to keep track of the classification accuracy of
# the model relative to the entire data set at the end of each training iteration. Report the
# 50 accuracy scores obtained during the whole training procedure (one per iteration).
# Plot the accuracy as a function of the cumulative number of training iterations. You
# may use matplotlib.pyplot.plot and matplotlib.pyplot.show for this purpose.
# Study the online documentation. Submit your code and results.


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
