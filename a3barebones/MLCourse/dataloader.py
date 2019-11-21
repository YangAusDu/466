import math
import numpy as np

####### Main load functions
def load_census(trainsize=1000, testsize=1000):
    strtype = 'a50'
    censusdtype={'names': ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'), 'formats': ('f', strtype, 'f', strtype,'f',strtype,strtype,strtype,strtype,strtype,'f','f','f',strtype,strtype)}
    incomeindex = 14
    convs = {14: lambda s: int(b'=' in s)}
    dataset = loadTxtDataset('datasets/censusincome.txt', dtype=censusdtype, converters=convs)
    numsamples = dataset.shape[0]
    subsetsamples = trainsize + testsize

    # Doing this specifically for census, since we do not want to add a bias unit
    # and because we cannot normalize features
    randindices = np.random.choice(numsamples, subsetsamples, replace=False)
    vals = np.zeros(subsetsamples)
    for ii in range(subsetsamples):
        if b'1' == dataset[randindices[ii]][incomeindex]:
            vals[ii] = 1.0

    Xtrain = dataset[randindices[0:trainsize]]
    ytrain = vals[0:trainsize]
    Xtest = dataset[randindices[trainsize:trainsize+testsize]]
    ytest = vals[trainsize:trainsize+testsize]

    # Remove the targets from Xtrain and Xtest
    allfeatures = list(censusdtype['names'][0:incomeindex])
    Xtrain = Xtrain[allfeatures]
    Xtest = Xtest[allfeatures]

    return ((Xtrain, ytrain), (Xtest, ytest))

def load_susy(trainsize=500, testsize=1000):
    """ A physics classification dataset """
    filename = 'datasets/susysubset.csv'
    dataset = loadCsvDataset(filename)
    trainset, testset = splitdataset(dataset, trainsize, testsize)
    return trainset, testset

def load_ctscan(trainsize=5000, testsize=5000):
    """ A CT scan dataset """
    filename = 'datasets/slice_localization_data.csv'
    dataset = loadCsvDataset(filename)
    trainset, testset = splitdataset(dataset, trainsize, testsize, featureoffset=1)
    return trainset, testset


####### Helper functions

# cache all of the datasets lazily
# this way students with slow harddrives don't have to wait for the data to load every run
loaded_datasets = {}
def loadCsvDataset(filename):
    if filename in loaded_datasets:
        return loaded_datasets[filename]

    dataset = np.genfromtxt(filename, delimiter=',')
    loaded_datasets[filename] = dataset

    return dataset

def loadTxtDataset(filename, dtype=None, converters=None):
    if filename in loaded_datasets:
        return loaded_datasets[filename]

    dataset = np.loadtxt(filename, delimiter=',', dtype=dtype, converters=converters)
    loaded_datasets[filename] = dataset

    return dataset

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    randindices = np.random.choice(dataset.shape[0], trainsize + testsize, replace=False)
    featureend = dataset.shape[1] - 1
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize], featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize], outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize+testsize], featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize], outputlocation]

    if testdataset is not None:
        Xtest = dataset[:, featureoffset:featureend]
        ytest = dataset[:, outputlocation]

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0], 1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0], 1))))

    return ((Xtrain, ytrain), (Xtest, ytest))
