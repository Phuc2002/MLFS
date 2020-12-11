## Algorithm Evaluation Method
1. Train and test data split
2. k-fold cross-validation
3. Which method is better?

- ML is all about predictive modeling and the goal is for that model to make good prediction. The problem is we don't know how good the prediction is because we don't know the new data yet and all of our available data has already been trained (if you try to test the model by using the train data of course it would predict almost 100% correctly. It already learned that data before!) Therefore we need to come up with statistical methods to estimate the performance of a model on new data. This class of method is called resampling method, because you're resampling your available training data into different portions and make use of them. We will implement 2 methods as mentioned above.

#### Train and Test Split
- This is the easiest resampling method both in terms of understanding and implementing. Therefore, it's the most widely used as well. Your available dataset is simply split into 2 datasets: a train dataset for training the model and a test data to hold back and later use to evaluate the performance of the model.<br><br>
- Each dataset is selected randomly so that the training and evaluation of the model is objective.<br><br>
- For a general dataset, a good data split percentage is 60%, where train data is 60% and test data is the rest 40% of the original dataset.<br><br>
- When we are comparing/benchmarking multiple algorithms, it makes sense that we're talking about the same dataset and the algorithms receive the exact same training and testing data. This is to make sure the performance comparison is consistent, or apple-to-apple.
```python
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
```
- We have our split function that take in the dataset (it's a list of list as we've already familiar) and the default split percentage. You can change this percentage by passing a new value into this parameter).<br><br>
- We first need to know what's the size, or how many rows, our training dataset would be<br><br>
- We then make a copy of our original dataset so we can manipulate and split it.<br><br>
- While we can still adding to out train data, select the rows of the temporary dataset randomly (by using the `randrange()` that generate a number from 0-size of temp dataset we're splitting). Add it to the training dataset.<br><br>
- As we finish adding, the rest of the temp dataset is now the test data.<br><br>
<br>
- A contrived dataset is a very small dataset that you may manually create just to test out if your functions are working:
```python
# contrived dataset
dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
seed(1)
train_dataset, test_dataset = train_test_data_split(dataset)
print("Training Data", train_dataset)
print("Testing Data", test_dataset)
```
- As we mentioned early, we select the train and test data randomly but have to make sure that these data must stay the same when using with other algorithms. The random `seed()` is used and fixed here before we randomly generate anything so that the next time we use this same fixed seed, we can randomly generate the same train and test data still!
```
Training Data [[3], [2], [7], [1], [8], [9]]
Testing Data [[4], [5], [6], [10]]
```

#### k-fold Cross-Validation Split
- Also can be just called Cross-Validation, is a method that first split the data into k groups. Therefore the algorithm will be trained and tested k times and the performance is summarized by taking the mean/average. Each group is called a fold, hence the name k-fold.<br><br>
- We first train the model on k - 1 group and test/evaluate the model on the remaining kth group. You then continue to do this so that by the end every fold or group will be used as the test dataset.<br><br>
- It makes sense that the k groups has the same number of rows and therefore k should be divisible by the number rows of the dataset. But if not, then there're some remaining rows of data that won't be used at all. <br><br>
- Generally, for a small dataset k=3 or there are 3 folds/groups to split. For large dataset, k=10. You can determine k yourself by compute the statistic like mean and standard deviation to better split the dataset large enough the it can resemble the original data.
```python
def cross_validation_split(dataset, folds = 4):
    dataset_split = []
    temp_dataset = dataset.copy()
    each_fold_size = len(dataset) // folds

    for i in range(folds):
        fold = []
        while len(fold) < each_fold_size:
            index_row = randrange(len(temp_dataset))
            fold.append(temp_dataset.pop(index_row))

        dataset_split.append(fold)
    return dataset_split
```
- The steps is mostly the same as splitting into 2 datasets as above, only now we're specified how many folds/groups for the dataset to be split into.
```
The folds/groups of data are: 
[[[3], [2]], [[7], [1]], [[8], [9]], [[10], [6]]]
```
- Notice that we have a multidimensional array now where there're 4 folds represented as list. Inside each fold is the rows of data and in this case each fold had 2 rows of single data value.<br><br>
- Also notice that our small contrived dataset has 10 rows of data by after cross-validation, we only have 8 rows of data in use. That's because when computing `each_fold_size`, size of dataset is not divisible by k.

#### How to Choose Resampling Method
- The go-to method of helping estimate the performance of ML model on a new data is the Cross-Validation method. When well-configured, that is you choose a good number of k-folds for the dataset to be split around, this method is very powerful as the algorithm is trained and tested multiple times on a dataset that look just like original dataset. But as the result, it can be very time consuming as well, especially when you have an incredibly large dataset or the model takes too long to train.
- When dataset is large (in those millions of records of data), train and test data split will be just as good. There are some noise or outliers in our dataset that will be eventually ignored since the data is so large.
- But on the other hand, in a small dataset, train and test split can be quite bad since the algorithm is only trained & evaluated once on just these 2 datasets and it would take into account the outliers which in turns gives noisy, unreliable estimate of the performance of a model on new data. 
- For a small dataset, Cross-Validation is the one to go for. Cross-Validation instead will average the performance score which gives us the reliable result. 


