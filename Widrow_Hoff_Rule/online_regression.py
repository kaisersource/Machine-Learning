import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
# Importing the dataset
import numpy as np

file = open("output.csv", "w+")


# training data with noise (e.g., measurement errors)
sigma = 0.2
a = 0.6
b = -0.4


xmin = -6
xmax = 6

xnp = np.arange(xmin, xmax, 0.5)
def perceptron_train(x,w):
    eta = 0.01
    epochs = 1000
    d= a*xnp+b
    sigma = 0.2
    w_zero=0
    w_one=0

    tr_d = d + np.random.randn(len(d)) * sigma
    for t in range(epochs):
        errors = 0
        for i in range(0, len(x)):
            d = w_one*x[i]+ w_zero
            delta = tr_d[i] - d          
            w_one+= eta*np.multiply(delta,x[i])
            w_zero+=eta*delta
            errors+=(delta)*(delta)
    ans = w_one*x+w_zero
    mse = np.sqrt(errors)


    print('mse online mode = ', mse)

    plt.plot(xnp, ans)
    plt.plot(xnp, tr_d, 'o')
    plt.show()

    return w



w = np.random.rand(len(xnp)+1)

w = perceptron_train(xnp, w)
file.close()
