import numpy as np

import MLCourse.dataloader as dtl
import MLCourse.utilities as utils
import classalgorithms as algs

def getaccuracy(ytest, predictions):
    correct = 0
    # count number of correct predictions
    correct = np.sum(ytest == predictions)
    # return percent correct
    return (correct / float(len(ytest))) * 100

def geterror(ytest, predictions):
    return (100 - getaccuracy(ytest, predictions))

""" k-fold cross-validation
K - number of folds
X - data to partition
Y - targets to partition
Algorithm - the algorithm class to instantiate
parameters - a list of parameter dictionaries to test

NOTE: utils.leaveOneOut will likely be useful for this problem.
Check utilities.py for example usage.
"""
def cross_validate(K, X, Y, Algorithm, parameters):
    all_errors = np.zeros((len(parameters), K))
    #print("Y shape: ", Y.shape)
    for k in range(K):
        for i, params in enumerate(parameters):
            fold_index0 = int(len(X)*k/K)
            fold_index1 = int(len(X)*(k+1)/K)

            Xtest = X[fold_index0:fold_index1]
            ytest = Y[fold_index0:fold_index1]
            X = np.array(X)
            Y = np.array(Y)
            
            Xtrain = np.concatenate((X[: fold_index0], X[fold_index1: ] ))
            ytrain = np.concatenate((Y[: fold_index0], Y[fold_index1: ] ))

            algo = Algorithm(params)
            algo.learn(Xtrain,ytrain)
            error = geterror(ytest,algo.predict(Xtest))
            all_errors[i][k] = error
            
                
    avg_errors = np.mean(all_errors, axis=1)
    for i, params in enumerate(parameters):
        print('Cross validate parameters:', params)
        print('average error:', avg_errors[i])

    best_parameters_index = np.argmin(avg_errors)
    best_parameters = parameters[best_parameters_index]
    return best_parameters

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Arguments for running.')
    parser.add_argument('--trainsize', type=int, default=5000,
                        help='Specify the train set size')
    parser.add_argument('--testsize', type=int, default=5000,
                        help='Specify the test set size')
    parser.add_argument('--numruns', type=int, default=1,
                        help='Specify the number of runs')
    parser.add_argument('--dataset', type=str, default="susy",
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    trainsize = args.trainsize
    testsize = args.testsize
    numruns = args.numruns
    dataset = args.dataset



    classalgs = {
        #'Random': algs.Classifier,
        #'Naive Bayes': algs.NaiveBayes,
        #'Linear Regression': algs.LinearRegressionClass,
        
        'Logistic Regression': algs.LogisticReg,
        #'Neural Network': algs.NeuralNet,
        #'Kernel Logistic Regression': algs.KernelLogisticRegression,
    }
    numalgs = len(classalgs)

    # Specify the name of the algorithm and an array of parameter values to try
    # if an algorithm is not include, will run with default parameters
    parameters = {
        # name of the algorithm to run
        'Naive Bayes': [
            # first set of parameters to try
            { 'usecolumnones': True },
            # second set of parameters to try
            { 'usecolumnones': False },
        ],
        'Logistic Regression': [
            { 'stepsize': 0.001 },
            { 'stepsize': 0.01 },
        ],
        'Neural Network': [
            { 'epochs': 100, 'nh': 4 },
            { 'epochs': 100, 'nh': 8 },
            #{ 'epochs': 100, 'nh': 16 },
            #{ 'epochs': 100, 'nh': 32 },
        ],
        'Kernel Logistic Regression': [
            { 'centers': 10, 'stepsize': 0.01 },
            { 'centers': 20, 'stepsize': 0.01 },
            { 'centers': 40, 'stepsize': 0.01 },
            { 'centers': 80, 'stepsize': 0.01 },
        ]
    }

    # initialize the errors for each parameter setting to 0
    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        if dataset == "susy":
            trainset, testset = dtl.load_susy(trainsize, testsize)
        elif dataset == "census":
            trainset, testset = dtl.load_census(trainsize,testsize)
        else:
            raise ValueError("dataset %s unknown" % dataset)

        # print(trainset[0])
        Xtrain = trainset[0]
        Ytrain = trainset[1]
        # cast the Y vector as a matrix
        Ytrain = np.reshape(Ytrain, [len(Ytrain), 1])

        Xtest = testset[0]
        Ytest = testset[1]
        # cast the Y vector as a matrix
        Ytest = np.reshape(Ytest, [len(Ytest), 1])

        best_parameters = {}
        for learnername, Learner in classalgs.items():
            params = parameters.get(learnername, [ None ])
            best_parameters[learnername] = cross_validate(5, Xtrain, Ytrain, Learner, params)

        for learnername, Learner in classalgs.items():
            params = best_parameters[learnername]
            print("best param: ", params)
            learner = Learner(params)
            learner.learn(Xtrain, Ytrain)
            error = geterror(Ytest,learner.predict(Xtest))
            errors[learnername][r] = error

    for learnername in classalgs:
        print('error: ', errors)
        aveerror = np.mean(errors[learnername])
        print('Average error for ' + learnername + ': ' + str(aveerror))
