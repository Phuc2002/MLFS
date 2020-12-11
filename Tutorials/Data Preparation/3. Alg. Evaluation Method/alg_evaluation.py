"""
I. Implement a train and test split of your data
II. Implement a k-fold cross validation split of the data
III. How to choose a resampling method
"""

from random import seed
from random import randrange

### train and test split of the data
def train_test_data_split(dataset, split = 0.60):
    train = []
    train_size = split * len(dataset)
    temp_dataset = dataset.copy()
    while len(train) < train_size:
        index_row = randrange(len(temp_dataset))
        train.append(temp_dataset.pop(index_row))

    test = temp_dataset
    return train, test

# contrived dataset
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
seed(1)
train_dataset, test_dataset = train_test_data_split(dataset)
print("Training Data", train_dataset)
print("Testing Data", test_dataset)


### k-fold Cross-Validation Split
def cross_validation_split(dataset, folds = 3):
    dataset_split = []
    temp_dataset = dataset.copy()
    fold_size = len(dataset) // folds       #make sure it's an integer!!

    for i in range(folds):
        fold = []
        while len(fold) < fold_size:
            index_row = randrange(len(temp_dataset))
            fold.append(temp_dataset.pop(index_row))

        dataset_split.append(fold)
    return dataset_split

seed(1)
folds = cross_validation_split(dataset, 4)
print("The folds/groups of data are: ")
print(folds)


