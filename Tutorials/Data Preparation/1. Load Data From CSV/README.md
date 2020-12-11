## Load Data From CSV
1. How to load a CSV file
2. How to convert strings from a file to floating point numbers
3. How to convert class value (string) to integers (encoding)

- CSV: Comma Separated Value. As the name suggest the data in the file in the format of rows of data, where each row is divided into columns using a comma `,`.
- The dataset will be use is: Pima Indian Diabetes Dataset and Iris Flower Species Dataset.

#### Load CSV File
- Use a Python standard built-in module/library called `csv` and its function `reader()` taking a filename as argument.
<br><br>
- We'll use our own function that use `reader()` to get and store the data in the way we want.
<br><br>
```python
from csv import reader
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as f:
        lines = reader(f)
        for line in lines:
            if not line:
                continue
            dataset.append(line)
    return dataset
```

- `reader()` returns an (iterable) object that stores rows of data from the file (think of it as raw/unprocessed data). To work with it loop through each row/line of the raw data and append the appropriate one to the list (some might just be an emtpy row)
- We finally have a 2D list, the 1st outer list containings rows. The inner lists (which are the rows) storing the column values for a given row.
<br><br>
- We'll start with the diabetes dataset first. Load the dataset and check out the 1st few rows:
```python
#load dataset
pima_data = "pima-indians-diabetes.data.csv"
pima_dataset = load_csv(pima_data)
def peak(dataset, num_rows):
    print(dataset[:num_rows])

print("Loaded data %s with %d rows and %d columns." % (pima_data, len(dataset), len(dataset[0])))
peak(pima_dataset, 5)
```
```Loaded data pima-indians-diabetes.data.csv with 768 rows and 9 columns.  [['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1'], ['1', '85', '66', '29', '0', '26.6', '0.351', '31', '0'], ['8', '183', '64', '0', '0', '23.3', '0.672', '32', '1'], ['1', '89', '66', '23', '94', '28.1', '0.167', '21', '0'], ['0', '137', '40', '35', '168', '43.1', '2.288', '33', '1']]```
<br><br>
#### Convert string to float
- Notice the data we output earlier is actually string. ML works/understands better numerical value so we will have to convert all the data to float.
```python
def to_float(dataset, col):
    for row in dataset:
        row[col] = float(row[col])

for i in range(len(pima_dataset[0])):
    to_float(pima_dataset, i)
```
- We decompose into 2 parts: loop through the column values and at each index convert the data to float. Then do so for every row in the entire dataset in another loop/function.
- `peak()` through the first few rows and the data should now be in float (the value output is not in `''` mark anymore).

#### Convert class value to integer.
- If we're working with the Iris dataset now, notice that the last column contains string. They are the class values that indicating what flower species is that in a specific row: Iris-setosa, Iris-versicolor and Iris-virginica.
<br><br>
- We mentioned before that ML prefers to work with numbers so we represent the 3 classes above from number 0-2:
```python
iris_data = "iris.csv"
def encode_to_int(dataset, col):
    class_encoding = {}
    value_encoding = 0
    for row in dataset:
        if row[col] in class_encoding:
            continue

        class_encoding[row[col]] = value_encoding
        value_encoding += 1

    for row in dataset:
        row[col] = class_encoding[row[col]]

    return class_encoding


encoding = encode_to_int(dataset, 4)
print(encoding)
```
```{'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}```
- Create a dictionary to store the class value and its corresponding integer value.
- We know the 4th columns is where the class values are, so for each row only look at that column value. If the class value is already encoded (it's already in our dictionary) then skip it. Otherwise add it to the dictionary and update the integer value.
- In the end, loop back the dataset and change the class value into integer value.


