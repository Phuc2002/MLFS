


### Compute Mean and Variance
def mean(values):
    return sum(values) / len(values)

def variance(values, mean):
    tmp = []
    for value in values:
        tmp.append((value - mean)**2)

    return sum(tmp)

#contrived dataset
dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
mean_x, mean_y = mean(x), mean(y)
var_x, var_y = variance(x, mean_x), variance(y, mean_y)
print('x stats: mean = %.4f \tvariance = %.4f' % (mean_x, var_x))
print('y stats: mean = %.4f \tvariance = %.4f' % (mean_y, var_y))


### Compute Covariance
def covariance(x, y, mean_x, mean_y):
    covar = 0.0
    for i in range(len(x)):     #or len(y)...whatever
        covar += (x[i] - mean_x) * (y[i] - mean_y)

    return covar

covar = covariance(x, y, mean_x, mean_y)
print("Covariance: %.4f" % covar)



### Compute the Coefficient
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]

    mean_x, mean_y = mean(x), mean(y)

    b1 = covariance(x, y, mean_x, mean_y) / variance(x, mean_x)
    b0 = mean_y - b1 * mean_x

    return b1, b0

b1, b0 = coefficients(dataset)
print("Coefficients: b1 = %.4f \t b0 = %.4f" % (b1, b0))


### Simple Linear Regression
def simple_linear_regression(train, test):
    predictions = []
    b1, b0 = coefficients(train)

    for row in test:
        xhat = row[0]
        yhat = b0 + b1 * xhat
        predictions.append(yhat)

    return predictions


### Evaluate

from math import sqrt
def rmse_metric(actual, predicted):
    mse = 0.0
    for i in range(len(actual)):
        mse += (actual[i] - predicted[i])**2

    return sqrt(mse / len(predicted))

def evaluate_alg(dataset, algorithm):
    test = []
    for row in dataset:
        data = row.copy()
        data[-1] = None
        test.append(data)

    predictions = algorithm(dataset, test)
    print(predictions)

    actual = [row[-1] for row in dataset]
    error = rmse_metric(actual, predictions)
    return error

rmse = evaluate_alg(dataset, simple_linear_regression)
print('RMSE: %.4f' % rmse)







