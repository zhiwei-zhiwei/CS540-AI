# Zhiwei Cao
# most of codes are from TA's solution and "def tree_to_code" is from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
import numpy as np
from math import log2
import copy
from sklearn.tree import _tree

'''
This script is using all 9 features (2,3,...,10) to create a tree, which serves as a template.
Todo: you need to modify this by using the several specified features to create your own tree 
Todo: you need to do the pruning yourself
Todo: you need to get all the output including the test results.
Todo: you also need to generate the tree of such the format in the writeup: 'if (x3 <= 6) return 2 .......'
'''

with open('breast-cancer-wisconsin.data', 'r') as f:
    a = [l.strip('\n').split(',') for l in f if '?' not in l]
a = np.array(a).astype(int)   # training data

with open('test.txt', 'r') as f:
    testSet = [l.strip('\n').split(',') for l in f if '?' not in l]
testSet = np.array(testSet).astype(int)   # training data


two = 0
four = 0
print("shape of a is ", a.shape)
for x in range(683):
    for y in range(11):
        if a[x][y] == 2:
            two += 1
        if a[x][y] == 4:
            four += 1
print("2's in total = ", two, "\n4's in total = ", four)

n = two + four
hY = -(two/n)*np.log2(two/n) - (four/n)*np.log2(four/n)
print("initial entropy is ", round(hY, 4))


def entropy(data):
    count = len(data)
    p0 = sum(b[-1] == 2 for b in data) / count
    if p0 == 0 or p0 == 1: return 0
    p1 = 1 - p0
    return -p0 * log2(p0) - p1 * log2(p1)


def infogain(data, fea, threshold):  # x_fea <= threshold;  fea = 2,3,4,..., 10; threshold = 1,..., 9
    count = len(data)
    d1 = data[data[:, fea - 1] <= threshold]
    d2 = data[data[:, fea - 1] > threshold]
    if len(d1) == 0 or len(d2) == 0: return 0
    return entropy(data) - (len(d1) / count * entropy(d1) + len(d2) / count * entropy(d2))


def find_best_split(data):
    c = len(data)
    c0 = sum(b[-1] == 2 for b in data)
    if c0 == c: return (2, None)
    if c0 == 0: return (4, None)
    ig = [[infogain(data, f, t) for t in range(1, 10)] for f in range(2, 8)]
    ig = np.array(ig)
    max_ig = max(max(i) for i in ig)
    if max_ig == 0:
        if c0 >= c - c0:
            return (2, None)
        else:
            return (4, None)
    ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
    fea, threshold = ind[0] + 2, ind[1] + 1
    return (fea, threshold)


def split(data, node):
    fea, threshold = node.fea, node.threshold
    d1 = data[data[:,fea-1] <= threshold]
    d2 = data[data[:, fea-1] > threshold]
    return (d1,d2)


class Node:
    def __init__(self, fea, threshold):
        self.fea = fea
        self.threshold = threshold
        self.left = None
        self.right = None



ig = [[infogain(a, fea, t) for t in range(1,10)] for fea in range(2, 8)]
ig = np.array(ig)
ind = np.unravel_index(np.argmax(ig, axis=None), ig.shape)
root = Node(ind[0] + 2, ind[1] + 1)

igg = [[infogain(testSet, fea, t) for t in range(1,10)] for fea in range(2, 8)]
igg = np.array(igg)
ind = np.unravel_index(np.argmax(igg, axis=None), igg.shape)
setRoot = Node(ind[0] + 2, ind[1] + 1)


def create_tree(data, node):
    d1, d2 = split(data, node)
    f1, t1 = find_best_split(d1)
    f2, t2 = find_best_split(d2)
    if t1 == None:
        node.left = f1
    else:
        node.left = Node(f1,t1)
        create_tree(d1, node.left)

    if t2 == None:
        node.right = f2
    else:
        node.right = Node(f2, t2)
        create_tree(d2, node.right)

create_tree(a, root)

create_tree(testSet,setRoot)


s1 = [setRoot]
s2 = []

while s1:
    s2 = copy.deepcopy(s1)
    s1 = []
    for n in s2:
        if n != 2 and n != 4:
            # print(f"{n.fea} < {n.threshold}")
            if n.left != None:
                s1 += [n.left]
            if n.right != None:
                s1 += [n.right]
        # else:
        #     print("else ", n)
    # print()
# with open("tree.txt","r") as f:
#     tree = f.read();
#     print(tree)

n2Minus = 0
n2Plus = 0
n4Minus = 0
n4Plus = 0
# t = 7
for i in range(a.shape[0]):
    if a[i][1] <= 7:
        if a[i][10] == 2:
            n2Minus += 1
        else:
            n4Minus += 1
    else:
        if a[i][10] == 2:
            n2Plus += 1
        else:
            n4Plus += 1
print("n2Minus ", n2Minus,
      "\nn2Plus ", n2Plus,
      "\nn4Minus ", n4Minus,
      "\nn4Plus ", n4Plus)

nMinus = n2Minus + n4Minus
nPlus = n2Plus + n4Plus

hYx = - (n2Minus/n)*np.log2(n2Minus/nMinus) - (n2Plus/n)*np.log2(n2Plus/nPlus) - (n4Minus/n)*np.log2(n4Minus/nMinus) - (n4Plus/n)*np.log2(n4Plus/nPlus)

def tree_to_code(root, depth):
    temp = " " * depth
    current = root
    if current != None and current != 2 and current != 4:
        fea = "x"+ str(current.fea)
        threshold = current.threshold
        print(f"{temp}if {fea} <= {threshold}")
        tree_to_code(current.left, depth + 1)
        print(f"{temp}else")
        tree_to_code(current.right, depth + 1)
    else:
        print(f"{temp}return {current}")

tree_to_code(root,0)
print("-------------------------------------------------------------------------")
tree_to_code(setRoot,0)


q7 = ""
for i in range(testSet.shape[0]):
    if testSet[i][2] <= 2:
        if testSet[i][6] <= 3:
            if testSet[i][1] <= 7: q7 += str(2)
            else:
                if testSet[i][3] <= 1: q7 += str(2)
                else: q7 += str(4)
        else:
            if testSet[i][1] <= 3: q7 += str(2)
            else:
                if testSet[i][4] <= 6:
                    if testSet[i][6] <= 4:
                        if testSet[i][1] <= 5: q7 += str(2)
                        else: q7 += str(4)
                    else: q7 += str(4)
                else: q7 += str(2)
    else:
        if testSet[i][2] <= 4:
            if testSet[i][6] <= 2:
                if testSet[i][2] <= 3:
                    if testSet[i][5] <= 4: q7 += str(2)
                    else:
                        if testSet[i][1] <= 2: q7 += str(2)
                        else: q7 += str(4)
                else:
                    if testSet[i][4] <= 3:
                        if testSet[i][1] <= 8: q7 += str(2)
                        else: q7 += str(4)
                    else: q7 += str(4)
            else:
                if testSet[i][1] <= 6:
                    if testSet[i][6] <= 5:
                        if testSet[i][1] <= 4:
                            if testSet[i][1] <= 2: q7 += str(4)
                            else: q7 += str(2)
                        else:
                            if testSet[i][5] <= 3: q7 += str(4)
                            else: q7 += str(2)
                    else:
                        if testSet[i][1] <= 5:
                            if testSet[i][5] <= 6: q7 += str(4)
                            else:
                                if testSet[i][5] <= 7: q7 += str(2)
                                else: q7 += str(4)
                        else: q7 += str(2)
                else:
                    if testSet[i][6] <= 7:
                        if testSet[i][4] <= 4: q7 += str(4)
                        else:
                            if testSet[i][3] <= 4: q7 += str(2)
                            else: q7 += str(4)
                    else: q7 += str(4)
        else:
            if testSet[i][4] <= 1:
                if testSet[i][1] <= 6:
                    if testSet[i][3] <= 6: q7 += str(4)
                    else: q7 += str(2)
                else: q7 += str(4)
            else:
                if testSet[i][6] <= 8:
                    if testSet[i][6] <= 7: q7 += str(4)
                    else:
                        if testSet[i][2] <= 8: q7 += str(4)
                        else:
                            if testSet[i][2] <= 9: q7 += str(2)
                            else: q7 += str(4)
                else: q7 += str(4)
    if i != testSet.shape[0] - 1:
        q7 += ","

file = open("q7.txt", "w+")
file.write(q7)
file.close()

q9 = ""
for i in range(testSet.shape[0]):
    if testSet[i][2] <= 1:
        if testSet[i][1] <= 5: q9 += str(2)
        else: q9 += str(4)
    else:
        if testSet[i][3] <= 4:
            if testSet[i][6] <= 1:
                if testSet[i][5] <= 4: q9 += str(2)
                else: q9 += str(4)
            else:
                if testSet[i][2] <= 4:
                    if testSet[i][3] <= 3:
                        if testSet[i][3] <= 1: q9 += str(2)
                        else: q9 += str(4)
                    else:
                        if testSet[i][1] <= 5: q9 += str(2)
                        else:
                            if testSet[i][5] <= 3: q9 += str(4)
                            else: q9 += str(2)
                else: q9 += str(4)
        else:
            if testSet[i][6] <= 4:
                if testSet[i][1] <= 6: q9 += str(2)
                else: q9 += str(4)
            else: q9 += str(4)

    if i != testSet.shape[0] - 1:
        q9 += ","

file = open("q9.txt", "w+")
file.write(q9)
file.close()