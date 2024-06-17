# most of code are from TA's solution
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# read file
def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')
x_test, y_test = data_loader('mnist_test.csv')

print('data loading done')

test_labels = [0, 4] # set test number
indices = np.where(np.isin(y_train, test_labels))[0]
indices_t = np.where(np.isin(y_test, test_labels))[0]

x = x_train[indices]
# print(len(x[1]))
# print(round(x[1],2))
y = y_train[indices]
x_t = x_test[indices_t]
y_t = y_test[indices_t]

# question1 = ""
# for i in range(len(x[1])):
#     # print(np.round(x[1], 2))
#     question1 += round(x[1][i], 2)
#     if i < len(x[1]):
#         print(",")
#         question1 += ","
# # print(question2)
# file = open("question1.txt", "w+")
# file.write(question1)
# file.close()

y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1
y_t[y_t == test_labels[0]] = 0
y_t[y_t == test_labels[1]] = 1

# Todo: you may need to change some hyper-paramter like num_epochs and alpha, etc
num_epochs = 10
m = x.shape[1]
n = x.shape[0]
alpha = 0.01
print(m)
large_num = 1e8
epsilon = 1e-6
thresh = 1e-4

w = np.random.rand(m)
b = np.random.rand()

c = np.zeros(num_epochs)
print(str(len(w)))



for epoch in range(num_epochs):
    a =expit(np.matmul(w, np.transpose(x)) + b)
    # expit(np.matmul(np.transpose(w), x) + b)

    w -= alpha * np.matmul(a - y, x)

    b -= alpha * (a - y).sum()

    cost = np.zeros(len(y))
    idx = (y == 0) & (a > 1 - thresh) | (y == 1) & (a < thresh)
    cost[idx] = large_num

    a[a < thresh] = thresh
    a[a > 1 - thresh] = thresh

    inv_idx = np.invert(idx)
    cost[inv_idx] = - y[inv_idx] * np.log(a[inv_idx]) - (1 - y[inv_idx]) * np.log(1 - a[inv_idx])
    c[epoch] = cost.sum()

    # if epoch % 3 == 0:
    #     print('epoch = ', epoch + 1, 'cost = ', c[epoch])

    if epoch > 0 and abs(c[epoch - 1] - c[epoch]) < epsilon:
        break

question2 = ""
for i in range(len(w)):
    if i < len(w):
        question2 += str(round(w[i], 4))
        question2 += ","
question2 += str(b)
file = open("question2.txt", "w+")
file.write(question2)
file.close()

# Todo: new test
new_test = np.loadtxt('test.txt', delimiter=',')
new_x = new_test / 255.0

# print(activation = tf.nn.relu)
# print(type(new_test))

print(w)
predictedVal = list()  #init an list to store the numbers
activationVal = list()
for x in new_x:
    # print(x)
    # activation_Val = 1 / (1 + np.exp(-(np.matmul(np.transpose(w), x) + b)))
    predicted_Val = expit(np.matmul(np.transpose(w), x) + b) # same like line 112, but it can prevent the worning
    predictedVal.append(predicted_Val)
    activation_Val = expit(x)
    activationVal.append((activation_Val))
# print(activationVal)

question3 = ""
question4 = ""

j = 0
# print(len(activationVal[1]))
for j in range(len(activationVal)):
    if j < len(activationVal):
        question3 += str(np.round(activationVal[j]))
        # question3 += ","
        j+=1
        if j < 200:
            question3 += ","

i = 0
for i in range(len(predictedVal)):
    if i < len(predictedVal):
        if predictedVal[i] >= 0.5:
            question4 += str(1.0)
        else:
            question4 += str(0.0)
        i+=1
        if i < 200:
            question4 += ","

file = open("question4.txt", "w+")
file.write(question4)
file.close()