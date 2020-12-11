"""
I. Random Prediction Algorithm
II. Zero Rule Algorithm
"""

### generate random prediction
from random import seed
from random import randrange
def random_alg(train, test):
    output_val = []
    for row in train:                       #get all the actual output value first
        output_val.append(row[-1])

    unique = []
    for val in output_val:                  #keep only the unique value from the above list
        if val not in unique:
            unique.append(val)

    predictions = []
    for row in test:                        #randomly predict the class value
        index = randrange(len(unique))
        predictions.append(unique[index])

    return predictions

seed(1)
train = [[0], [1], [0], [1], [0], [1]]
test = [[None], [None], [None], [None]]
predictions = random_alg(train, test)
print("Using Random Prediction Algorithm: ", predictions)


### Zero Rule Algorithm

#For Classification
def zero_rule_alg_classification(train, test):
    output_val = []
    for row in train:
        output_val.append(row[-1])

    freq = {}
    for val in output_val:              #get the unique output value
        if val not in freq:             #store it in a frequency dictionary
            freq[val] = 1
        else:
            freq[val] += 1

    highest = -1
    for key in freq:                    #get the most common output value
        if highest < freq[key]:         #-> that is our prediction!
            highest = freq[key]
            most_common = key

    predictions = []
    for i in range(len(test)):
        predictions.append(most_common)

    return predictions

seed(1)
train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
test = [[None], [None], [None], [None]]
predictions = zero_rule_alg_classification(train, test)
print("Using Zero Rule Algorithm for Classification Problem: ", predictions)

#For Regression
def zero_rule_alg_regression(train, test):
    output_val = []
    for row in train:
        output_val.append(row[-1])

    mean = sum(output_val) / len(output_val)

    predictions = []
    for i in test:
        predictions.append(mean)

    return predictions

seed(1)
train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]
predictions = zero_rule_alg_regression(train, test)
print("Using Zero Rule Algorithm for Regression Problem", predictions)
