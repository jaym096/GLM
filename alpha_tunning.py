import numpy as np
import random
import glm
import supporting_func as func

def convert_to_list(train_x,train_y):
    t_x = []
    t_y = []
    for i in range(len(train_x)):
        t_x.append(train_x[i].tolist()[0])
        t_y.append(train_y[i].tolist()[0])
    return t_x,t_y

def generateKfold(size,no_of_fold,train_x,train_y):
    t_x, t_y = convert_to_list(train_x,train_y)
    final_fold_x = []
    final_fold_y = []
    i_fold_x = []
    i_fold_y = []
    for i in range(len(train_x)):
        if((i%size==0) and (i>0)):
            final_fold_x.append(i_fold_x)
            final_fold_y.append(i_fold_y)
            i_fold_x = []
            i_fold_y = []
            i_fold_x.append(t_x[i])
            i_fold_y.append(t_y[i])
        else:
            i_fold_x.append(t_x[i])
            i_fold_y.append(t_y[i])
    final_fold_x.append(i_fold_x)
    final_fold_y.append(i_fold_y)
    return final_fold_x, final_fold_y

def getRandomAplha():
    random_a = []
    for i in range(0,50):
        x = random.randint(0,100)
        random_a.append(x)
    return random_a

def kfold_train_test(kfold,k):
    test_data = kfold[k]
    train_set = []
    for i in range(len(kfold)):
        if(i==k):
            continue
        else:
            train_set += kfold[i]
    train_data = np.stack(train_set, axis=0)
    test_data = np.stack(test_data, axis=0)
    return train_data, test_data

def calculate_MSE_t2(W,dataSamples,target):
    phi = dataSamples
    phi_W = np.matmul(phi,W)
    t = target
    diff = phi_W - t
    sqr = np.square(diff)
    mse = np.mean(sqr)
    return mse

def param_tuning(n_data, labels, algo):
    train_x, train_y, test_x, test_y = func.divideData(n_data, labels)
    no_of_fold = 10
    size = round(len(train_x)/(no_of_fold))
    kfold_x, kfold_y = generateKfold(size,no_of_fold,train_x,train_y)
    alpha_r = getRandomAplha()
    list_of_MSE = [] #holds avg MSE for a particular alpha
    for a in alpha_r:
        kfold_mse = [] #holds mse for each fold for a particular value of alpha
        for i in range(len(kfold_x)):
            samp_train_x, samp_test_x = kfold_train_test(kfold_x,i)
            samp_train_y, samp_test_y = kfold_train_test(kfold_y,i)
            shp = np.shape(samp_train_x)
            w = np.zeros((shp[1],1))
            w = glm.GLM2(samp_train_x, samp_train_y, w, a, algo)
            w = w[0]
            t_hat = func.predict(w, test_x, algo)
            mse = func.calculate_err(test_y, t_hat, algo)
            kfold_mse.append(mse)
        avg_mse = np.average(kfold_mse)
        list_of_MSE.append(avg_mse)
    min_index = list_of_MSE.index(min(list_of_MSE))
    final_alpha = alpha_r[min_index]
    return final_alpha, min(list_of_MSE)