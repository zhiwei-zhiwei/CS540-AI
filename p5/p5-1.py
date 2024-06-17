# most of code are from TA's solution 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import inline as inline

np.set_printoptions(threshold=np.inf)
# %matplotlib inline
import copy
import math
import heapq

'''
The below script is based on a 55 * 57 maze. 
Todo:
    1. Plot the maze and solution in the required format.
    2. Implement DFS algorithm. (I've given you the BFS below)
    3. Implement A* with Euclidean distance. (I've given you the one with Manhattan distance)

'''

width, height = 41, 41
X, Y = 14, 2

ori_img = mpimg.imread('maze.png')
img = ori_img[:, :, 0]

M = np.zeros([height * 2 + 1, width * 3 + 1])

for h in range(height * 2 + 1):
    for w in range(width * 3 + 1):
        if (h % 2 == 0) and (w % 3 == 0):
            M[h, w] = 1
        if (h % 2 == 0) and (w % 3 != 0):
            i = int(h / 2)
            j = int(np.floor(w / 3))
            if np.sum(img[16 * i + 0:16 * i + 2, 16 * j + 2:16 * j + 16]) == 0:
                M[h, w] = 2
        if (h % 2 != 0) and (w % 3 == 0):
            i = int(np.floor(h / 2))
            j = int(w / 3)
            if np.sum(img[16 * i + 2:16 * i + 16, 16 * j + 0:16 * j + 2]) == 0:
                M[h, w] = 3

f = open("q1.txt", 'a')
for h in range(height * 2 + 1):
    for w in range(width * 3 + 1):
        if M[h, w] == 0:
            f.write(' ')
        if M[h, w] == 1:
            f.write('+')
        if M[h, w] == 2:
            f.write('-')
        if M[h, w] == 3:
            f.write('|')
    f.write('\n')
f.close()

# file = open("question1.txt","w+")
# file.write(str(question1))
# file.close()

class Cell:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.succ = ''
        self.action = ''  # which action the parent takes to get this cell

cells = [[Cell(i, j) for i in range(width)] for j in range(height)]
# print(cells)

f2 = open("q2.txt", 'a')
for i in range(height):
    succ = []
    for j in range(width):
        s = ''
        c1, c2 = i * 16 + 8, j * 16 + 8
        if img[c1 - 8, c2] == 1:
            f2.write('U')
            s += 'U'
        if img[c1 + 8, c2] == 1:
            f2.write('D')
            s += 'D'
        if img[c1, c2 - 8] == 1:
            f2.write('L')
            s += 'L'
        if img[c1, c2 + 8] == 1:
            f2.write('R')
            s += 'R'
        cells[i][j].succ = s
        succ.append(s)
        if j != width - 1:
            f2.write(',')
    f2.write('\n')
f2.close()

# 2
cells[0][20].succ = cells[0][20].succ.replace('U', '')
cells[40][20].succ = cells[40][20].succ.replace('D', '')

# bfs
f3 = open("q5.txt", 'a')
visited = set()
s1 = {(0, 20)}
s2 = set()
queue = []
while (40, 20) not in visited:
    for a in s1:
        queue.append(a)
        visited.add(a)
        i, j = a[0], a[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i - 1, j) not in (s1 | s2 | visited):
            f3.write('1,')
            queue.pop(-1)
            queue.append("1")
            s2.add((i - 1, j))
            cells[i - 1][j].action = 'U'
        if 'D' in succ and (i + 1, j) not in (s1 | s2 | visited):
            f3.write('1,')
            queue.pop(-1)
            queue.append("1")
            s2.add((i + 1, j))
            cells[i + 1][j].action = 'D'
        if 'L' in succ and (i, j - 1) not in (s1 | s2 | visited):
            f3.write('1,')
            queue.pop(-1)
            queue.append("1")
            s2.add((i, j - 1))
            cells[i][j - 1].action = 'L'
        if 'R' in succ and (i, j + 1) not in (s1 | s2 | visited):
            f3.write('1,')
            queue.pop(-1)
            queue.append("1")
            s2.add((i, j + 1))
            cells[i][j + 1].action = 'R'
        else:
            f3.write('0,')
            queue.pop(-1)
            queue.append("0")
    s1 = s2
    s2 = set()
    f3.write('\n')
f3.close()
# print(queue)
# print(len(queue))


# dfs
dvisited = set()
ds1 = {(0, 20)}
ds2 = set()
dqueue = []
while (40, 20) not in dvisited:
    for a in ds1:
        dqueue.append(a)
        dvisited.add(a)
        i, j = a[0], a[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i - 1, j) not in (ds1 | ds2 | dvisited):
            dqueue.pop(-1)
            dqueue.append("1")
            ds2.add((i - 1, j))
            cells[i - 1][j].action = 'U'
        if 'D' in succ and (i + 1, j) not in (ds1 | ds2 | dvisited):
            dqueue.pop(-1)
            dqueue.append("1")
            ds2.add((i + 1, j))
            cells[i + 1][j].action = 'D'
        if 'L' in succ and (i, j - 1) not in (ds1 | ds2 | dvisited):
            dqueue.pop(-1)
            dqueue.append("1")
            ds2.add((i, j - 1))
            cells[i][j - 1].action = 'L'
        if 'R' in succ and (i, j + 1) not in (ds1 | ds2 | dvisited):
            dqueue.pop(-1)
            dqueue.append("1")
            ds2.add((i, j + 1))
            cells[i][j + 1].action = 'R'
        else:
            dqueue.pop(-1)
            dqueue.append("0")
    ds1 = ds2
    ds2 = set()


cur = (40, 20)
s = ''
seq = []

while cur != (0, 20):
    # seq.append()
    # print(cur)
    seq.append(cur)
    i, j = cur[0], cur[1]
    t = cells[i][j].action
    # print(i,j)
    s += t
    # print(s)
    if t == 'U':
        cur = (i + 1, j)
    if t == 'D':
        cur = (i - 1, j)
    if t == 'L':
        cur = (i, j + 1)
    if t == 'R':
        cur = (i, j - 1)
seq.append((0,20))
action = s[::-1]
seq = seq[::-1]
# 3
#
for k in range(len(action)):
    i, j = seq[k]
    a = action[k]
    # if k == 0:
    #     j = seq[k][1]
    #     j = j + 1
    #     seq[k] = i,j
    #     print(i,j,seq[k])
    M[2 * i + 1, 3 * j + 1:3 * j + 3] = 4
    if a == 'U':
        M[2 * i + 0, 3 * j + 1:3 * j + 3] = 4
    if a == 'D':
        M[2 * i + 2, 3 * j + 1:3 * j + 3] = 4
    if a == 'L':
        M[2 * i + 1, 3 * j + 0] = 4
    if a == 'R':
        M[2 * i + 1, 3 * j + 3] = 4
i, j = seq[k+1]
M[2 * i + 1, 3 * j + 1:3 * j + 3] = 4

M[0, 3 * int((width - 1) / 2) + 1:3 * int((width - 1) / 2) + 3] = 4
M[-1, 3 * int((width - 1) / 2) + 1:3 * int((width - 1) / 2) + 3] = 4
# print(seq)
f = open("q4.txt", 'a')
for h in range(height * 2 + 1):
    for w in range(width * 3 + 1):
        if M[h, w] == 0:
            f.write(' ')
        if M[h, w] == 1:
            f.write('+')
        if M[h, w] == 2:
            f.write('-')
        if M[h, w] == 3:
            f.write('|')
        if M[h, w] == 4:
            f.write('@')
    f.write('\n')
f.close()

## Part2
man = {(i, j): abs(i - 40) + abs(j - 20) for i in range(width) for j in range(height)}
euc = {(i, j): math.sqrt((i - 40) ** 2 + (j - 20) ** 2) for i in range(width) for j in range(height)}

question7 = ""
cout = 0
for i in man.keys():
    cout += 1
    question7 += str(man.get(i))
    if cout % 41 == 0:
        question7 += str("\n")
    else:
        question7 += str(",")
file = open("question7.txt","w+")
file.write(str(question7))
file.close()

# manhattan   use man
g = {(i, j): float('inf') for i in range(width) for j in range(height)}
g[(0, 20)] = 0
queue = [(0, 20)]
visited = set()

# man
f8 = open("q8.txt", 'a')
while queue and (40, 20) not in visited:
    queue.sort(key=lambda x: g[x] + man[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        i, j = point[0], point[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i - 1, j) not in visited:
            if (i - 1, j) not in queue:
                f8.write('1,')
                # queue.pop(-1)
                # queue.append("1")
                queue += [(i - 1, j)]
            g[(i - 1, j)] = min(g[(i - 1, j)], g[(i, j)] + 1)
            queue.append((i - 1, j))
        if 'D' in succ and (i + 1, j) not in visited:
            if (i + 1, j) not in queue:
                f8.write('1,')
                queue += [(i + 1, j)]
            g[(i + 1, j)] = min(g[(i + 1, j)], g[(i, j)] + 1)
            # queue.append(i + 1, j)
        if 'L' in succ and (i, j - 1) not in visited:
            if (i, j - 1) not in queue:
                f8.write('1,')
                queue += [(i, j - 1)]
            g[(i, j - 1)] = min(g[(i, j - 1)], g[(i, j)] + 1)
            # queue.append(i, j - 1)
        if 'R' in succ and (i, j + 1) not in visited:
            if (i, j + 1) not in queue:
                f8.write('1,')
                queue += [(i, j + 1)]
            g[(i, j + 1)] = min(g[(i, j + 1)], g[(i, j)] + 1)
            # queue.append(i, j + 1)
    else:
        f8.write('0,')
    # print(queue)
    # f8.write('\n')
f8.close()

f9 = open("q9.txt", 'a')
while queue and (40, 20) not in visited:
    queue.sort(key=lambda x: g[x] + euc[x])
    point = queue.pop(0)
    if point not in visited:
        visited.add(point)
        i, j = point[0], point[1]
        succ = cells[i][j].succ
        if 'U' in succ and (i - 1, j) not in visited:
            if (i - 1, j) not in queue:
                f9.write('1,')
                queue += [(i - 1, j)]
            g[(i - 1, j)] = min(g[(i - 1, j)], g[(i, j)] + 1)
            queue.append((i - 1, j))
        if 'D' in succ and (i + 1, j) not in visited:
            if (i + 1, j) not in queue:
                f9.write('1,')
                queue += [(i + 1, j)]
            g[(i + 1, j)] = min(g[(i + 1, j)], g[(i, j)] + 1)
        if 'L' in succ and (i, j - 1) not in visited:
            if (i, j - 1) not in queue:
                f9.write('1,')
                queue += [(i, j - 1)]
            g[(i, j - 1)] = min(g[(i, j - 1)], g[(i, j)] + 1)
        if 'R' in succ and (i, j + 1) not in visited:
            if (i, j + 1) not in queue:
                f9.write('1,')
                queue += [(i, j + 1)]
            g[(i, j + 1)] = min(g[(i, j + 1)], g[(i, j)] + 1)
    else:
        f9.write('0,')
f9.close()
