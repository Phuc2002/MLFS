"""
I. Classification Accuracy
II. Confusion Matrix
III. Mean Absolute Error
IV. Root Mean Squared Error
"""

### Classification Accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):            #or len(predicted)... whatever
        if actual[i] == predicted[i]:
            correct += 1

    return (correct / len(actual)) * 100

# Test accuracy
actual = [0,0,0,0,0,1,1,1,1,1]
predicted = [0,1,0,0,0,1,0,1,1,1]
accuracy = accuracy_metric(actual, predicted)
print("Accuracy metric: ", accuracy)

### Confusion Matrix
def confusion_matrix(actual, predicted):
    unique_class = {}
    encode = 0
    for value in actual:                #get the unique class value only
        if value in unique_class:       #and encode them into integers
            continue
        unique_class[value] = encode
        encode += 1
    num_unique = len(unique_class)


    matrix = []
    for i in range(num_unique):         #creating rows
        matrix.append([])
    for i in range(num_unique):         #creating columns and initialise the entries.
        for j in range(num_unique):
            matrix[j].append(0)


    for i in range(len(actual)):        #count the prediction
        x = unique_class[actual[i]]
        y = unique_class[predicted[i]]
        matrix[x][y] += 1

    return unique_class, matrix


actual_new =    [0,0,0,0,0,1,1,1,1,1]
predicted_new = [0,1,1,0,0,1,0,1,1,1]
class_val, mat = confusion_matrix(actual_new, predicted_new)
print("Class value encoding: ", class_val)
print("Confusion Matrix: ", mat)


def print_confusion_matrix(unique, matrix):
    keys = []
    print("   ", end="")
    for key in unique:
        print(key, end="  ")
        keys.append(key)
    print()
    for i in range(len(matrix[0])):
        print(keys[i], matrix[i])

print_confusion_matrix(class_val, mat)


### mean absolute error
def mae_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(actual[i] - predicted[i])

    return sum_error / float(len(actual))

actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
mae = mae_metric(actual, predicted)
print("Mean Absolute Error: ", mae)

### root mean squared error
from math import sqrt
def rmse_metric(actual, predicted):
    mse = 0.0
    for i in range(len(actual)):
        mse += (actual[i] - predicted[i])**2

    return sqrt(mse / len(predicted))

actual = [0.1, 0.2, 0.3, 0.4, 0.5]
predicted = [0.11, 0.19, 0.29, 0.41, 0.5]
rmse = rmse_metric(actual, predicted)
print("Root Mean Squared Error: ", rmse)



