from packages.modules import *


#load data & data cleaning
data = pd.read_csv("data.csv", delimiter=",")
data = data.drop(["Unnamed: 32"], axis= 1)
data["diagnosis"] = data["diagnosis"].replace(["B", "M"],[1, 0])

# split dataset
X_train, Y_train, X_test, Y_test = train_test_split(data)#load data & data cleaning
data = pd.read_csv("data.csv", delimiter=",")
data = data.drop(["Unnamed: 32"], axis= 1)
data["diagnosis"] = data["diagnosis"].replace(["B", "M"],[1, 0])

# split dataset
X_train, Y_train, X_test, Y_test = train_test_split(data)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
