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

layers_dims = [X_train.shape[1], 20, 7, 5, 1]
costs = []
parameters = initialize_parameters_deep(layers_dims)
num_iterations = 3000
learning_rate = 0.0075

for i in range(0, num_iterations):
    AL, caches = L_model_forward(X_train.T, parameters)
    cost = compute_cost(AL, Y_train)
    grads = L_model_backward(AL, Y_train, caches)
    parameters = update_parameters(parameters, grads, learning_rate)

    if i % 100 == 0 or i == num_iterations - 1:
        print(f"Cost after iteration {i}: {np.squeeze(cost)}")
    if i % 100 == 0 or i == num_iterations:
        costs.append(cost)
    