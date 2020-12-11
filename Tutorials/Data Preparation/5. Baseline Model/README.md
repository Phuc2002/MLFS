## Baseline Models
1. Random prediction algorithm
2. Zero rule prediction algorithm

- After the ML model has made it predictions and you evaluate it with a metric of choice, it's good to know if the algorithm is good enough by looking at its error/loss from the evaluation metric. Therefore, it requires that we establish baseline performance on a problem/dataset so we can compare to, and the algorithms we'll be using are as mentioned above. 
- The scores from these baseline algorithms will provide a point of comparison when evaluating other ML algorithms on the same problems later on, and of course we want to do better than the baseline performance.
- In a sense, these baseline algorithms are like ML model but they predict naively.

#### Random Prediction Algorithm
- The algorithm works for both classification and regression problems.
- The algorithm predicts a random outcomes as observed in the training dataset and therefore it's the simplest algorithm to implement: 
```python
from random import seed
from random import randrange
def random_alg(train, test):
    output_val = []
    for row in train: 
        output_val.append(row[-1])

    unique = []
    for val in output_val:  
        if val not in unique:
            unique.append(val)

    predictions = []
    for _ in test: 
        index = randrange(len(unique))
        predictions.append(unique[index])

    return predictions
```
- Assuming that the output values are at the last column of the train dataset, which is typical for a normal dataset, store all of them in a list.
- We then get the unique values only from the previous list, which makes sense since we're predicting and therefore we only want to predict the values that indeed exist in the dataset. For regression problem with real numerical value, the number of unique values can be large but for classification problem with limited output class values, the number can be small.
- We randomly get the value from the `unique` list and make it our predicted value! (continue to do so and store the randomly predicted values in a list to return).
- Since we're doing things randomly, we want to fix the random number `seed` so that every later when we call the same algorithm on the same dataset, it still makes the same decision.
```python
seed(1)
train = [[0], [1], [0], [1], [0], [1]]
test = [[None], [None], [None], [None]]
predictions = random_alg(train, test)
print("Using Random Prediction Algorithm: ", predictions)
```
```
Using Random Prediction Algorithm:  [0, 0, 1, 0]
```
- Our small contrived data will just contain the output column only with no other supporting data, just for simplicity sake. The test data is unknown for now since we don't know the predictions yet.
- Since the unique values are only 0 and 1, our baseline model will also have 0 and 1 for its prediction and it made 4 of them which correspond to the 4 'row of data' from the test dataset.


#### Zero Rule Algorithm
The 0 rule algorithm performs better than the random algorithm as it uses more information about the dataset depending on the problem we're working on.

1. Classification
- Intuitively, a good way to predict something is to predict the class value that is most common in the training dataset.
- Suppose in a training set there are 90 instances of an email is spam and 10 instances of an email is not spam, then this 0 rule algorithm would achieve 90% accuracy.
- This is easily implemented using Python dictionary to count up the frequency of the class values, then traverse through the dictionary to get the class value with the highest frequency i.e. appear most often in the dataset.
```python
def zero_rule_alg_classification(train, test):
    output_val = []
    for row in train:
        output_val.append(row[-1])

    freq = {}
    for val in output_val:
        if val not in freq:
            freq[val] = 1
        else:
            freq[val] += 1

    highest = -1
    for key in freq: 
        if highest < freq[key]: 
            highest = freq[key]
            most_common = key

    predictions = []
    for i in range(len(test)):
        predictions.append(most_common)

    return predictions
```
```python
train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
test = [[None], [None], [None], [None]]
predictions = zero_rule_alg_classification(train, test)
print("Using Zero Rule Algorithm for Classification Problem: ", predictions)
```
- Test the code with our new contrived dataset, and we should be able to predict ourselves that the algorithm predicts the value `'0'`.
<br><br>

2. Regression
- Intuitively, when we try to predict a number from a huge dataset, we can calculate the mean (average) or the median (the middle value of the entire dataset) and make it our prediction. The following algorithm will compute the mean from given training dataset's output values:
```python
def zero_rule_alg_regression(train, test):
    output_val = []
    for row in train:
        output_val.append(row[-1])

    mean = sum(output_val) / len(output_val)

    predictions = []
    for i in test:
        predictions.append(mean)

    return predictions
```
```python
train = [[10], [15], [12], [15], [18], [20]]
test = [[None], [None], [None], [None]]
predictions = zero_rule_alg_regression(train, test)
print("Using Zero Rule Algorithm for Regression Problem", predictions)
```
- From our small dataset, we can compute ourselves that the mean value here is 15.0
<br><br>
```
Using Zero Rule Algorithm for Classification Problem:  ['0', '0', '0', '0']
Using Zero Rule Algorithm for Regression Problem [15.0, 15.0, 15.0, 15.0]
```
- Since we come up with a single rule depending on the data, compute a single value and make it our only prediction; for our entire test dataset we would then only have that 1 value to be the predicted value!
