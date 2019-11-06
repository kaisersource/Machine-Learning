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
file_train = open("output_train.csv", "w+")
y = dataset.iloc[:, 5].values  # becomes l in the function
#y = np.array([-1,-1,1,1,1])
ynp = np.asarray(y)
w0test = 0
# training

# training data with noise (e.g., measurement errors)
#sigma = 0.2


def perceptron_train(x, w, y):
   # w = np.zeros(len(X[0]))
    eta = 0.001
    epochs = 100

#    sigma = 0.01


    print(x[0])
    print(len(x[0]))
    #print("weights", w)

    for t in range(epochs):
        #errors = 0
        for i in range(0, len(x)):
            #print("nparray:",np.multiply(np.multiply(eta,(y[i] - np.dot(wT,x[i]))),x[i,:]))
            wT = w[1:].transpose()
            upd = np.multiply(eta * (y[i] - np.dot(wT, x[i, :])), x[i, :])
            w[1:] = np.add(w[1:], upd)
            w[0] += eta*(y[i] - (np.dot(wT,x[i])+w[0]))
            #print("weights updated", w)
    ans = np.dot(x, w[1:]) + w[0]
        #w = np.sum(w,arg)

    #mse = np.sqrt((np.subtract(ans,y) * np.subtract(ans,y)).sum())
    mse = np.sqrt(np.dot(np.subtract(ans,y),np.subtract(ans,y)))
    # online
    print('weights in online mode=', w)

    print("this is w0:",w[0])

    print('mse online mode = ', mse)


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


w = np.random.rand(len(x[0])+1)
print("random Weights: ", w)
w = perceptron_train(xnp, w, ynp)


#print("This is w0: ", w0test)

def perceptron_test(x, w, w0):
    print(len(x_test))
    for i in range(0, len(x)):
        f = np.dot(x[i], w[1:]) + w[0]  # dot product2
        # f = np.dot(x[i], w)+w0 dot product2

        # activation function
        if f > 0:
            y = 1
            #w0 += eta*y
            # print(str(y))
            file.write("Label "+str(y) + "\n")
        else:
            y = -1
            #w0 += eta*y
            # print(str(y))
            file.write("Label "+str(y) + "\n")
    return f


print("Testing...")

perceptron_test(x_test, w, w0test)
print("...Test Ended")
file.close()
file_train.close()