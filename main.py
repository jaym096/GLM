import matplotlib.pyplot as plt
import supporting_func as func
import numpy as np
import timeit
import glm
import sys

def calculate_statistics(no_of_iterations, total_err, final_runtime_conv):
    final_iters = []
    mean_err = []
    std = []
    r_conv = []
    for i in range(0,10):
        collector  = []
        collector2 = []
        collector3 = []
        for j in range(0,30):
            collector.append(no_of_iterations[j][i])
            collector2.append(total_err[j][i])
            collector3.append(final_runtime_conv[j][i])
        it = np.mean(collector)
        mu = np.mean(collector2)
        rcv = np.mean(collector3)
        sd = np.std(collector2)
        final_iters.append(it)
        mean_err.append(mu)
        std.append(sd)
        r_conv.append(rcv)
    return mean_err, std, final_iters, r_conv

def likelihood(n_data, labels, alpha, algo):
    total_err = []
    no_of_iterations = []
    final_runtime_conv = []
    n = np.arange(0.1,1.1,0.1)
    for i in range(0,30):
        train_x, train_y, test_x, test_y = func.divideData(n_data, labels)
        size_err = []
        sizes = []
        runtime_conv = []
        iterations_for_sizes = []
        for i in range(len(n)):
            samp_x, samp_y, sample_size = func.GetRandomSample(train_x, train_y, n[i])
            sizes.append(sample_size)
            shp = np.shape(samp_x)
            w = np.zeros((shp[1],1))
            t1 = timeit.default_timer()
            Wmap, iterations = glm.GLM2(samp_x, samp_y, w, alpha, algo)
            t2 = timeit.default_timer()
            t_hat = func.predict(Wmap, test_x, algo)
            err = func.calculate_err(test_y, t_hat, algo)
            size_err.append(err)
            iterations_for_sizes.append(iterations)
            runtime_conv.append(t2-t1)
        total_err.append(size_err)
        no_of_iterations.append(iterations_for_sizes)
        final_runtime_conv.append(runtime_conv)
    
    mean_err, sd_err, avg_iterations, runtime_until_conv = calculate_statistics(no_of_iterations, total_err, final_runtime_conv)
    
    print("average iterations for each sizes: ", [round(x) for x in avg_iterations])
    print("runtime until convergence for each sizes: ", runtime_until_conv)
    print("Mean error: ", mean_err)
    print("Standar Deviation: ",sd_err)
    
    plt.errorbar(sizes,mean_err,sd_err)
    plt.xticks(sizes, [".1N",".2",".3N",".4N",".5N",".6N",".7N",".8N",".9N","1N"])
    plt.xlabel('size of sample data set')
    plt.ylabel('Error')
    plt.title("learning curve plots of training set size and error as a function of standard deviation",
              fontsize="small")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    algo = sys.argv[1]
    filename = sys.argv[2]
    alpha = 10
    data = func.getData("pp3data/"+filename+".csv")
    labels = func.getData("pp3data/labels-"+filename+".csv")
    z = np.ones(len(labels)).reshape(len(labels),1)
    n_data = np.hstack((z,data))
    t1 = timeit.default_timer()
    likelihood(n_data, labels, alpha, algo)
    t2 = timeit.default_timer()
    print("time taken to run: ",t2-t1)