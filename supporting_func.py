import numpy as np
import random

def R_logistic(y):
    y1 = 1 - y
    R = np.multiply(y,y1)
    R = np.matrix(R)
    return np.diag(R.A1)

def R_poisson(y):
    R = y
    R = np.matrix(R)
    return np.diag(R.A1)

def sigmoid(a):
    y = 1 / (1 + np.exp(-a))
    return y


def calculateYR(a, algo):
    if(algo == "log"):
        y = sigmoid(a)
        R = R_logistic(y)
    if(algo == "pos"):
        y = np.exp(a)
        R = R_poisson(y)
    return y, R

def calculateDR(a, algo, train_y):
    s = 1
    levels = [-np.Inf, -2, -1, 0, 1, np.Inf]
    d = np.empty(train_y.shape[0])
    R = np.empty(train_y.shape[0])
    yij = sigmoid(s * (levels - a))
    t = np.array(train_y)
    t = t.astype(int)
    #print(t)
    for i in range(len(yij)):
        #print(int(t[i][0]))
        d[i] = yij[i,t[i][0]] + yij[i, t[i][0]-1] - 1
        R[i] = pow(s,2) * (yij[i, t[i][0]] * (1 - yij[i,t[i][0]]) + yij[i, t[i][0]-1] * (1 - yij[i, t[i][0]-1]))
#    for i in range(len(train_y)):
#        ti = int(train_y[i])
#        val = s * (levels[ti] - a[i])
#        y_i_ti = sigmoid(val)
#        
#        val2 = s * (levels[ti-1] - a[i])
#        y_i_ti1 = sigmoid(val2)
#        
#        d[i] = y_i_ti + y_i_ti1 - 1
#        R[i] = pow(s,2) * (y_i_ti * (1 - y_i_ti) + y_i_ti1 * (1 - y_i_ti1))
    return d.reshape(len(d),1), np.diag(R)

def predict(Wmap, test_x, algo):
    a = np.matmul(test_x, Wmap)
    if(algo == "log"):
        t = sigmoid(a)
        t_hat = [1 if k>=0.5 else 0 for k in t]
        return t_hat
    if(algo == "pos"):
        lambda_ = np.exp(a)
        t_hat = np.floor(lambda_)
        t_hat = list(t_hat)
        return t_hat
    if(algo == "ord"):
        s = 1
        levels = [-np.Inf, -2, -1, 0, 1, np.Inf]
        p_j = []
        for j in range(len(levels)):
            val = s * (levels[j] - a)
            y_j = sigmoid(val)
            val2 = s * (levels[j-1] - a)
            y_j1 = sigmoid(val2)
            pj = y_j - y_j1
            if(p_j == []):
                p_j = np.array(pj)
            else:
                p_j = np.hstack((p_j,pj))
        t_hat = np.argmax(p_j, axis=1)
        return list(t_hat)

def calculate_err(t, t_hat, algo):
    if(algo == "log"):
        err = []
        for i in range(len(t)):
            if(t[i] == t_hat[i]):
                err.append(0)
            else:
                err.append(1)
        final_err = np.mean(err)
        return final_err
    if(algo == "pos" or algo == "ord"):
        err = []
        for i in range(len(t)):
            diff = abs(t_hat[i] - t[i])
            err.append(diff)
        final_err = np.mean(err)
        return final_err

def GetRandomSample(train_x,train_y, nf):
    samp_x = []
    samp_y = []
    sample_size = int(len(train_y) * nf)
    temp_array = list(range(0,len(train_x)))
    c = random.sample(temp_array,sample_size)
    for index in c:
        samp_x.append(train_x[index])
        samp_y.append(train_y[index])
    samp_x = np.stack(samp_x, axis=0)
    samp_y = np.matrix.transpose(np.stack(samp_y, axis=1))
    return samp_x,samp_y,sample_size

def divideData(n_data, labels):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    sample_size = int(len(labels)/3)
    temp_array = list(range(0, len(labels)))
    c = random.sample(temp_array,sample_size)
    for i in range(len(labels)):
        if(i in c):
            test_x.append(n_data[i])
            test_y.append(labels[i])
        else:
            train_x.append(n_data[i])
            train_y.append(labels[i])
    #random.shuffle(train_x)
    #random.shuffle(train_y)
    #return train_x, train_y, test_x, test_y
    return np.asmatrix(train_x), np.asmatrix(train_y), np.asmatrix(test_x), np.asmatrix(test_y)

def getData(filename): 
    data_mtx = []
    with open(filename,'r') as f:
        lines = [lines for lines in f.read().split("\n")][:-1]
        for each_line in lines:
            each_line = [float(value) for value in each_line.split(",")]
            data_mtx.append(each_line)
    return data_mtx