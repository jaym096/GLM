# Generalized Linear Model (GLM)
This is an academic project done in the course **CSCI-B55 Machine Learning** at Indiana University.

**Tools and Technology**: Python, NumPy

Implemented the pipeline for **GLMs** from scratch including **Logistic**, **Poisson** and **Ordinal regression** using second-level 
maximum likelihood by approximating the posterior using Laplace approximation. Optimized the hyperparameters of the GLM using **random 
search** and compared the performances. 

The project consists of 4 code files (.py):\
_main.py_, _supporting_func.py_, _glm.py_, _alpha_tunning.py_

The pp3data folder contains all the data set required for this project. The code considers that the data folder is in the same directory 
as the code.

## HOW TO RUN THE CODE
    1. To run the code you only use the main.py file which takes 2 command line arguments
       -> likelihood function: "log", "pos", "ord" where
                       "log" : Logistic likelihood
                       "pos" : Poisson likelihood
                       "ord" : Ordinal likelihood
       -> filename (dataset) to use

    2. Example: If you want to run logistic regression for dataset A, you should run the following code:
	     python main.py log A

    3. Example 2: Similarly, if you want to run logistic regression for dataset usps, you should run the following code:
	     python main.py log usps

    4. Example 3: if you want to run ordinal regression for dataset AO, you should run the following code:
	     python main.py ord AO
