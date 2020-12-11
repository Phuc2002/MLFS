"""
I. Normalize data
II. Standardize data
III. When to normalize vs. standardize
"""

from csv import reader

pima_data = "pima-indians-diabetes.data.csv"
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as f:
        lines = reader(f)
        for line in lines:
            if not line:
                continue
            dataset.append(line)
    return dataset

pima_dataset = load_csv(pima_data)
def peak(dataset, num_rows):
    print(dataset[:num_rows])

def to_float(dataset, col):
    for row in dataset:
        row[col] = float(row[col])

for i in range(len(pima_dataset[0])):
    to_float(pima_dataset, i)

print("Loaded data %s with %d rows and %d columns." % (pima_data, len(pima_dataset), len(pima_dataset[0])))
print("Original Data: ")
peak(pima_dataset, 1)


#Normalize Data
## find min max of an attribute first (in each column)
def dataset_minmax_column(dataset):
    minmax = []
    for i in range(len(dataset[0])):
        col_values = []
        for row in dataset:
            col_values.append(row[i])

        min_value = min(col_values)
        max_value = max(col_values)
        minmax.append([min_value, max_value])           #2D list

    return minmax

##rescale the data value to between the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            min_value, max_value = minmax[i]
            row[i] = (row[i] - min_value) / (max_value - min_value)


minmax = dataset_minmax_column(pima_dataset)
print("Min & Max: ", minmax)
pima_dataset_normalize = pima_dataset.copy()
normalize_dataset(pima_dataset_normalize, minmax)
print("Normalized Data: ")
peak(pima_dataset_normalize, 1)


#Standardize Data
##Find the mean for an attribute (each column)
def means_column(dataset):
    means = []
    for i in range(len(dataset[0])):
        sum_column = 0
        for row in dataset:
            sum_column += row[i]

        means.append(sum_column / len(dataset))

    return means

##Find standard deviation for a column
from math import sqrt

def stdev_column(dataset, means):
    stdevs = []
    for i in range(len(dataset[0])):
        variance = 0
        for row in dataset:
            variance += (row[i] - means[i])**2

        stdev = sqrt(variance / (len(dataset) - 1))
        stdevs.append(stdev)

    return stdevs


def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(dataset[0])):
            row[i] = (row[i] - means[i]) / stdevs[i]


means = means_column(pima_dataset)
stdevs = stdev_column(pima_dataset, means)
pima_dataset_standardize = pima_dataset.copy()
standardize_dataset(pima_dataset_standardize, means, stdevs)
print("Standardized Data:")
peak(pima_dataset_standardize, 1)






