"""
I. Load a csv file and get dataset
II. Convert data to float
III. Convert data to preferred numerical value (encoding)
"""

#use a csv module to read in csv file
from csv import reader

pima_data = "pima-indians-diabetes.data.csv"
iris_data = "iris.csv"

#load a csv file
##a wrapper function that wrap around this behaviour!
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as f:
        lines = reader(f)           #return an (iterator - list-like) object
        for line in lines:
            if not line:
                continue
            dataset.append(line)       #make it into a 2D list (more organizable/accessible)
    return dataset


#load dataset
pima_dataset = load_csv(pima_data)
def peak(dataset, num_rows):
    print(dataset[:num_rows])

print("Loaded data %s with %d rows and %d columns." % (pima_data, len(pima_dataset), len(pima_dataset[0])))
#peak(-1)
peak(pima_dataset, 5)

#notice that each list element is actually a string
def to_float(dataset, col):
    for row in dataset:
        row[col] = float(row[col])

for i in range(len(pima_dataset[0])):
    to_float(pima_dataset, i)

peak(pima_dataset, 5)


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

iris_dataset = load_csv(iris_data)
encoding = encode_to_int(iris_dataset, 4)
print(encoding)
peak(iris_dataset, 5)


