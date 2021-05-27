import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# Setting the random seed, feel free to change it and see different solutions.
time = datetime.now()
#print(time.microsecond*time.second)
np.random.seed(time.microsecond*time.second)
def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.1, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

def on_press(event):
    global i
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        i+=1
        line1.set_ydata(boundary_lines[i][0]*delim_x + boundary_lines[i][1])
        fig.canvas.draw()

if __name__== "__main__":
    dataframe = pd.read_csv("data.csv")
    #print(dataframe)
    Xy = dataframe.to_numpy()
    X = dataframe.to_numpy()[:,0:2]
    y = dataframe.to_numpy()[:,2]
    #print(X)
    #print(y)
    boundary_lines = trainPerceptronAlgorithm(X,y)
    #print(boundary_lines)
    #print(boundary_lines[0][0])
    delim_x = np.linspace(0,1,3)
    
    #print(Xy)
    #print([(x[0],x[1]) for x in Xy if x[2] == 1])
    #plot
    #plt.ion()
    i=4
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press) 
    #ax.plot([(x[0],x[1]) for x in Xy if x[2] == 1], 'go',[(x[0],x[1]) for x in Xy if x[2] == 0], 'ro')
    ax.plot([x[0] for x in Xy if x[2] == 0],[x[1] for x in Xy if x[2] == 0],'ro',[x[0] for x in Xy if x[2] == 1],[x[1] for x in Xy if x[2] == 1],'go')
    line1,= ax.plot(delim_x,boundary_lines[i][0]*delim_x + boundary_lines[i][1])
    ax.axis([0,1,0,1])
    ax.set_xlabel('fig')
    ax.set_title('Press a key')
    plt.show()