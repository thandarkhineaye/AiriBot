# importing the module
import csv

# open the file in read mode
filename = open("DataSets/QuesListOfHR&TRDataSet.csv", "r", encoding="utf-8-sig")

# creating dictreader object
file = csv.reader(filename)

# creating empty lists
data = []


# iterating over each row and append
# values to empty list
for col in file:
    data.append(col)

# printing lists
print(data)
