import numpy as np

import MLCourse.utilities as utils

# Susy: ~50 error
class Classifier:
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the training data """
        pass

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

# Susy: ~27 error
class LinearRegressionClass(Classifier):
    def __init__(self, parameters = {}):
        self.params = {'regwgt': 0.01}
        self.weights = None

    def learn(self, X, y):
        # Ensure y is {-1,1}
        y = np.copy(y)
        y[y == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = X.shape[0]
        numfeatures = X.shape[1]

        inner = (X.T.dot(X) / numsamples) + self.params['regwgt'] * np.eye(numfeatures)
        self.weights = np.linalg.inv(inner).dot(X.T).dot(y) / numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

# Susy: ~25 error
class NaiveBayes(Classifier):
    def __init__(self, parameters = {}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = utils.update_dictionary_items({'usecolumnones': False}, parameters)


    def learn(self, Xtrain, ytrain):
        # obtain number of classes
        if ytrain.shape[1] == 1:
            self.numclasses = 2
        else:
            raise Exception('Can only handle binary classification')

        if self.params['usecolumnones'] == False:
            Xtrain = [utils.leaveOneOut(each, 8) for each in Xtrain]


        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)
        s_0 = np.zeros(Xtrain.shape[1])
        s_1 = np.zeros(Xtrain.shape[1])
        count_mean_1 = 0
        count_mean_0 = 0
        for index in range(len(ytrain)):
            if ytrain[index] == 1:
                s_1 += Xtrain[index]
                count_mean_1 += 1
            else:
                s_0 += Xtrain[index]
                count_mean_0 += 1
        
        self.mean0 = s_0/count_mean_0
        self.mean1 = s_1/count_mean_1
        
        
        self.py0 = count_mean_0/len(Xtrain)
        self.py1 = count_mean_1/len(Xtrain)
        

        count_var_1 = 0
        count_var_0 = 0
        sv_0, sv_1 = np.zeros(Xtrain.shape[1]), np.zeros(Xtrain.shape[1])
        for index in range(len(ytrain)):
            if ytrain[index] == 1:
                sv_1 += np.square(Xtrain[index] - self.mean1)
                count_var_1 += 1
            else:
                sv_0 += np.square(Xtrain[index] - self.mean0)
                count_var_0 += 1

        self.var1 = sv_1/count_var_1
        self.var0 = sv_0/count_var_0

        

    def predict(self, Xtest):
        if self.params['usecolumnones'] == False:
            Xtest = [utils.leaveOneOut(each, 8) for each in Xtest]

        Xtest = np.array(Xtest)
        numsamples = Xtest.shape[0]
        px_y1 = np.array([utils.gaussian_pdf(each, self.mean1, np.sqrt(self.var1)) for each in Xtest])
        px_y0 = np.array([utils.gaussian_pdf(each, self.mean0, np.sqrt(self.var0)) for each in Xtest])
        px_y1 = np.array([np.sum(np.log(each)) for each in px_y1])
        px_y0 = np.array([np.sum(np.log(each)) for each in px_y0])

        c0 = px_y0 * self.py0
        c1 = px_y1 * self.py1
        
        prediction = c1 - c0
        predictions = []
        for each in prediction:
            if each < 0:
                predictions.append(0)
            else:
                predictions.append(1) 
        predictions = np.array(predictions)
        return np.reshape(predictions, [numsamples, 1])

# Susy: ~23 error
class LogisticReg(Classifier):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({'stepsize': 0.01, 'epochs': 100}, parameters)
        self.weights = None

    def learn(self, X, y):
        self.weights = np.zeros(X.shape[1])
        X = np.array(X)
        y = np.array(y)
        stepsize = self.params['stepsize']
        epochs = self.params['epochs']

        #using stochastic gradient descent
        for epoch in range(epochs):
            array = np.arange(len(X))
            np.random.shuffle(array)
            for each_index in array:
                gradient = utils.sigmoid(np.dot(X[each_index],self.weights)) - y[each_index]
                self.weights = self.weights - (stepsize) * gradient * X[each_index]
                
    def predict(self, Xtest):
        output = utils.sigmoid(np.dot(Xtest, self.weights))
        #print("predicting")
        threshold_probs = 0.5
        ypred = np.zeros(len(Xtest))
        for index in range(len(output)):
            if output[index] >= threshold_probs:
                ypred[index] = 1
            else:
                ypred[index] = 0
        return np.reshape(ypred, [len(Xtest), 1])


# Susy: ~23 error (4 hidden units)
class NeuralNet(Classifier):
    def __init__(self, parameters={}):
        self.params = utils.update_dictionary_items({
            'nh': 4,
            'transfer': 'sigmoid',
            'stepsize': 0.01,
            'epochs': 10,
        }, parameters)

        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')

        self.wi = None
        self.wo = None

    def learn(self, Xtrain, ytrain):
        print("learning..")
        epochs = self.params['epochs']
        stepsize = self.params['stepsize']
        nh = self.params['nh']
        num_features = Xtrain.shape[1]
        self.wi =  np.random.randn(nh, num_features) * (1/np.sqrt(num_features)) 
        self.wo =  np.random.randn(1, nh) * (1/np.sqrt(num_features)) 
        
        #array = np.arange(len(Xtrain))
        for epoch in range(epochs):
            array = np.arange(len(Xtrain))
            np.random.shuffle(array)
            for each_index in array:
                #forward propagation:
                hidden_layer, output_layer = self.evaluate(Xtrain[each_index])
                #back propagation:
                gradient_output = np.zeros((1,nh))
                delta_B = (output_layer - ytrain[each_index]) * 1
                for j in range(nh):
                    gradient_output[0][j] = delta_B * hidden_layer[j]

                gradient_input = np.zeros((nh, num_features))
                delta_A = np.zeros(nh)
                for i in range(nh):
                    delta_A[i] = self.wo[0][i] * delta_B * self.dtransfer(Xtrain[each_index])[i]
                    gradient_input[i] = np.dot(delta_A[i], Xtrain[each_index])
                
                #Gradient descent
                self.wo = self.wo - gradient_output * stepsize
                self.wi = self.wi - gradient_input * stepsize
                
    def predict(self,Xtest):
        print("predicting...")
        Xtest = np.array(Xtest)
        ypred = np.zeros(Xtest.shape[0])
        index = 0
        for each_sample in Xtest:
            hidden_layer = self.transfer(np.dot(self.wi, each_sample.T))
            output_layer = self.transfer(np.dot(self.wo, hidden_layer)).T
            if output_layer >= 0.5:
                ypred[index] = 1
                index += 1
            else:
                ypred[index] = 0
                index += 1
        return np.reshape(ypred, [len(Xtest), 1])
        

    def evaluate(self, inputs):
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs.T))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah)).T

        return (
            ah, # shape: [nh, samples]
            ao, # shape: [classes, nh]
        )

    def update(self, inputs, outputs):
        pass

# Note: high variance in errors! Make sure to run multiple times
# Susy: ~28 error (40 centers)
class KernelLogisticRegression(LogisticReg):
    def __init__(self, parameters = {}):
        self.params = utils.update_dictionary_items({
            'stepsize': 0.01,
            'epochs': 100,
            'centers': 10,
        }, parameters)
        self.weights = None

    def learn(self, X, y):
        pass

    def predict(self, Xtest):
        pass
