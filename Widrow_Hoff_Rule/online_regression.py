import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
# Importing the dataset
import numpy as np

#dataset = pd.read_csv('data5D_train.csv', header=None, delimiter=r"\s+")
#x = dataset.iloc[:, :-1].values
#xnp = np.asarray(x)
#dataset2 = pd.read_csv('data5D_test.csv', header=None, delimiter=r"\s+")
#x_test = dataset2.iloc[:, :].values

file = open("output.csv", "w+")

#y = dataset.iloc[:, 5].values  # becomes l in the function
#y = np.array([-1,-1,1,1,1])
#ynp = np.asarray(y)
w0test = 0
# training

# training data with noise (e.g., measurement errors)
sigma = 0.2
a = 0.6
b = -0.4


xmin = -6
xmax = 6
#ymin = -6
#ymax = 6
xnp = np.arange(xmin, xmax, 0.5)
def perceptron_train(x,w):
   # w = np.zeros(len(X[0]))
    eta = 0.01
    epochs = 1000
    d= a*xnp+b
    sigma = 0.2
    w_zero=0
    w_one=0
    #tr_d = np.multiply(np.add(y,random.randint(0,len(y))),sigma)
    #print("weights", w)
    tr_d = d + np.random.randn(len(d)) * sigma

    for t in range(epochs):
        errors = 0
        for i in range(0, len(x)):
            d = w_one*x[i]+ w_zero
            delta = tr_d[i] - d
            if  (delta!=0):
            #print("nparray:",t.multiply(np.multiply(eta,(y[i] - np.dot(wT,x[i]))),x[i,:]))
                w_one+= eta*np.multiply(delta,x[i])
                w_zero+=eta*delta
                errors+=(delta)*(delta)
            #print("weights updated", w)

        #w = np.sum(w,arg)
    ans = w_one*x+w_zero
    mse = np.sqrt(errors)
    # online
    print('weights in online mode=', w)

    print("this is w0:",w_zero)

    print('mse online mode = ', mse)

    #training data with noise (e.g., measurement errors)

    #tr_d = y + np.random.randn(len(y)) * sigma
    plt.plot(xnp, ans)
    plt.plot(xnp, tr_d, 'o')
    plt.show()

    return w


'''
###fake online iterative mode
    for t in range(epochs):
        #errors = 0
        for i in range(0, len(X)):
            for j in range(0, len(w)):
                w= np.sum(w, np.multiply(np.multiply(eta,(y[i] - np.dot(wT,x[i])),x[i][j])))
    for j in range(0,len(w)):
        print("weight", w[j])
'''
# print(errors)

# if(errors == 0):
#    break
#w0test = w0


print("random Weights: ", w)
w = perceptron_train(xnp, w)
file.close()
