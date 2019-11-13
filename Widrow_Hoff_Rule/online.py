import numpy as np
import pandas as pd

# Importing the dataset
import numpy as np

dataset = pd.read_csv('data5D_train.csv', header=None, delimiter=r"\s+")
x = dataset.iloc[:, :-1].values
xnp = np.asarray(x)
dataset2 = pd.read_csv('data5D_test.csv', header=None, delimiter=r"\s+")
x_test = dataset2.iloc[:, :].values

file = open("output.csv", "w+")

y = dataset.iloc[:, -1:].values

ynp = np.asarray(y)

def perceptron_train(x, w, y):
   # w = np.zeros(len(X[0]))
    eta = 0.0003
    epochs = 1000
    print(x[0])
    print(len(x[0]))


    for t in range(epochs):
        #errors = 0
        ans = np.dot(x, w[1:]) + w[0]
        for i in range(len(x)):
            w[1:]+= eta*(y[i] - ans[i])*x[i]
            w[0] += eta*(y[i] - ans[i])
    print("this is w0:",w[0])
    return w

w = 2*np.random.rand(len(x[0])+1)
print("random Weights: ", w)
wtrain = perceptron_train(xnp, w, ynp)
print('weights in online mode=', wtrain)


def threshold(k):
    if k > 0:
        return 1
    else:
        return -1

def perceptron_test(x, w, w0):
    print(len(x_test))
    for i in range(0, len(x)):
        f = np.dot(x[i], w[1:]) + w[0]  # dot product2
        # activation function
        y=threshold(f)

        file.write(str(y) + "\n")
    return f

print("Testing...")
perceptron_test(x_test, w, w0test)
print("...Test Ended")

file.close()
