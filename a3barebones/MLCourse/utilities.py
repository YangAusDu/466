import math
import numpy as np

def gaussian_pdf(x, mean, stdev):
    #x = np.array(x)
    #mean = np.array(mean)
    #stdev = np.array(stdev)
    if stdev.any() < 1e-3:
        if math.fabs(x - mean) < 1e-2:
            return 1.0
        else:
            return 0.0
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
    return (1 / (np.sqrt(2 * math.pi) * stdev)) * exponent

def sigmoid(x):
    # Cap -x, to avoid overflow
    # Underflow is okay, since it gets set to zero
    if not np.isscalar(x):
        x[x < -100] = -100
    elif x < -100:
        x = -100

    return 1.0 / (1.0 + np.exp(np.negative(x)))

def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones((len(probs), 1), dtype='int')
    classes[probs < 0.5] = 0
    return classes

# takes an array and an index
# returns a new array without the element at the index
# example: leaveOneOut([2, 4, 6, 8], 2) -> [2, 4, 8]
def leaveOneOut(arr, idx):
    return [ arr[i] for i in range(len(arr)) if i != idx ]

def update_dictionary_items(dict1, dict2):
    """
    Replace any common dictionary items in dict1 with the values in dict2
    There are more complicated and efficient ways to perform this task,
    but we will always have small dictionaries, so for our use case, this simple
    implementation is acceptable.
    """
    if dict2 is None:
        return dict1
    for k in dict1:
        if k in dict2:
            dict1[k] = dict2[k]

    return dict1
