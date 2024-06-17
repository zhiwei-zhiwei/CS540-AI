# most of code are from TA's solution AND code in line 150 and 165 are from Tianwei Bao
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
from scipy.special import expit

# Todo: you need to change the activation function from relu (current) version to logistic, remember, not only the activation function, but the weight update part as well.

def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')
x_test, y_test = data_loader('mnist_test.csv')

test_labels = [0, 4] # set the training numbers
indices = np.where(np.isin(y_train, test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]

test_labels = [0, 4]
indices = np.where(np.isin(y_train, test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]

y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1
y_t[y_t == test_labels[0]] = 0
y_t[y_t == test_labels[1]] = 1
num_hidden_uints = 28 # number of hidden layers


def relu(x):
    y = x
    y[y < 0] = 0
    return y


def diff_relu(x):
    y = x
    y[x > 0] = 1
    y[x <= 0] = 0
    return y


def nnet(train_x, train_y, test_x, test_y, lr, num_epochs):
    num_train = len(train_y)
    num_test = len(test_y)

    train_x = np.hstack((train_x, np.ones(num_train).reshape(-1, 1)))
    test_x = np.hstack((test_x, np.ones(num_test).reshape(-1, 1)))

    num_input_uints = train_x.shape[1]  # 785

    wih = np.random.uniform(low=-1, high=1, size=(num_hidden_uints, num_input_uints))  # 392*785

    who = np.random.uniform(low=-1, high=1, size=(1, num_hidden_uints + 1))  # 1 * 393

    for epoch in range(1, num_epochs + 1):
        out_o = np.zeros(num_train)
        out_h = np.zeros((num_train, num_hidden_uints + 1))  # num_train * 393
        out_h[:, -1] = 1
        for ind in range(num_train):
            row = train_x[ind]  # len = 785
            out_h[ind, :-1] = relu(np.matmul(wih, row))
            out_o[ind] = 1 / (1 + np.exp(-sum(out_h[ind] @ who.T)))

            delta = np.multiply(diff_relu(out_h[ind]), (train_y[ind] - out_o[ind]) * np.squeeze(who))
            wih += lr * np.matmul(np.expand_dims(delta[:-1], axis=1), np.expand_dims(row, axis=0))
            who += np.expand_dims(lr * (train_y[ind] - out_o[ind]) * out_h[ind, :], axis=0)
        error = sum(- train_y * np.log(out_o) - (1 - train_y) * np.log(1 - out_o))
        num_correct = sum((out_o > 0.5).astype(int) == train_y)

        print('epoch = ', epoch, ' error = {:.7}'.format(error),
              'correctly classified = {:.4%}'.format(num_correct / num_train))

    return wih.T, who


# Todo: change these hyper parameters
lr = 0.01
num_epochs = 10

W1, W2 = nnet(x, y, x_t, y_t, lr, num_epochs)
b1 = W1[-1:, :] # bias for first layer
b2 = W2[0, -1] # bias for second layer


# print(W1) # array
# print(W1.shape) # 785
# print(W1.shape[1])
# print(W1[1])
# print(len(W1[0]))
# print(W2) # array
# print(len((W2))) # 1
question5 = ""
for i in range(785):
    for j in range(28):
        if j < 28:
            question5 += str(np.round(W1[i][j], 4))
            if j != 27:
                question5 += ","
    if i < 784:
        question5 += "\n"
file = open("question5.txt", "w+")
file.write(question5)
file.close()

# print(W2)
# print(W2.shape)
for6 = 0
question6 = ""
for i in range(29):
    if i < 29:
        question6 += str(np.round(W2[0][i], 4))
        for6 += 1
        if for6 < 29:
            question6 += ","
file = open("question6.txt", "w+")
file.write(question6)
file.close()
# Todo: new test

new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0

# print(wleayer2)
# print(type(W1))
# print(W1.tolist())
# print(len(W1.tolist()))
# print(W1[:-1])
# print(len(W1[:-1]))
# print(type(b1))
# tempVal = list()
# activationVal = list()
# aaa = 0
# print(W1[:-1].shape)
# print(new_x.shape)
activationVal_One = expit(np.dot(new_x, W1[:-1, :]) + b1) # calcuate the value as a matrix expect the first number
q6 = ""
j = 0
# print(len(activationVal[1]))
for j in range(len(activationVal_One)):
    if j < len(activationVal_One):
        q6 += str(np.round(activationVal_One[j], 4))
        j += 1
        if j < 200:
            q6 += ","

file = open("q6.txt", "w+")
file.write(q6)
file.close()

activationVal_Two = expit(np.dot(activationVal_One, W2[0, :-1]) + b2)
# for x in new_x:
#     aaa += 1
    # temp_Val = expit(np.matmul(np.transpose(x), W1) + b1)
    # temp_Val = 1 / (1 + np.exp(-np.matmul(np.transpose(x), W1[:-1]) + b1))
    # print(temp_Val)
    # activation_Val = 1 / (1 + np.exp(-(np.matmul(temp_Val, np.transpose(W2)) + b2 )))
    # activation_Val = expit(np.matmul(np.transpose(W2), x) + b2)
    # activationVal.append((activation_Val))
    # tempVal.append(temp_Val)
# print(aaa)
# for x in new_x:
    # activation_Val = 1 / (1 + np.exp(-np.matmul()))


question7 = ""
question8 = ""
j = 0
# print(len(activationVal[1]))
for j in range(len(activationVal_Two)):
    if j < len(activationVal_One):
        question7 += str(np.round(activationVal_Two[j], 2))
        question8 += str(np.round(activationVal_Two[j]))
        j+=1
        if j < 200:
            question7 += ","
            question8 += ","

file = open("question7.txt", "w+")
file.write(question7)
file.close()

file = open("question8.txt", "w+")
file.write(question8)
file.close()