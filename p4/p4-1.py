# most of code are from TA's solution and line 108-135 are from Tianwei Bao's code

import math

import numpy as np
from numpy.linalg import norm
import pandas as pd
'''
Todo: 
1. Part 1 in P4.
2. Euclidean distance (currently are all manhattan in my code below)
3. Complete linkage distance
4. Total distortion
5. Output all required information in correct format

PS: Currently, I choose 
	n = num of all distinct countries, and
	m = 3 (latitude, longitude, total deaths until Jun27, 
		  i.e., 1st, 2nd, last number for each country as parameters).
	Also, for countries that have several rows, I average the latitude, longitude and sum up the deaths.

	You may need to change some of that based on your part 1 results.

'''



# For 'South Korea', and "Bonaire Sint Eustatius and Saba" (line 145 and 257), I removed the ',' in name manually
with open('time_series_covid19_deaths_US.csv') as f:
    data = list(f)[1:]
state_list = ["Alabama", "Alaska", "American Samoa", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Diamond Princess", "District of Columbia", "Florida", "Georgia", "Grand Princess", "Guam", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Northern Mariana Islands", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virgin Islands", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
list = []
listS = []
d_dict = {}
for d in data:
    l = d.strip('\n').split(',')
    c = l[6]  # state
    if c == "Wisconsin":
        list.append(l[14:-1])
    if c == "South Carolina":
        listS.append(l[14:-1])
# **************************************wisc*************************
temp_store = []
f=open('temp_store.txt','w+')
for i in range(74):
    jointsFrame = list[i] #each line
    temp_store.append(jointsFrame)
    for Ji in range(550):
        strNum = str(jointsFrame[Ji])
        f.write(strNum)
        if Ji != 549:
            f.write(',')
    f.write('\n')
f.close()
with open('temp_store.txt', 'r') as f:
    q1 = [l.strip('\n').split(',') for l in f if '?' not in l]
q1 = np.array(q1).astype(int)
store1 = []
for x in range(550):
    storeNum = 0
    for y in range(73):
        storeNum += q1[y][x]
    store1.append(storeNum)
# ************************************sc**********************************
temp_store_S = []
f=open('temp_store_S.txt','w+')
for i in range(48):
    jointsFrame = listS[i] #each line
    temp_store_S.append(jointsFrame)
    for Ji in range(550):
        strNum = str(jointsFrame[Ji])
        f.write(strNum)
        if Ji != 549:
            f.write(',')
    f.write('\n')
f.close()
with open('temp_store_S.txt', 'r') as f:
    q1_S = [l.strip('\n').split(',') for l in f if '?' not in l]
q1_S = np.array(q1_S).astype(int)
store2 = []
for x in range(550):
    storeNum2 = 0
    for y in range(48):
        storeNum2 += q1_S[y][x]
    store2.append(storeNum2)
question1 = ""
question1 += str(store1)
question1 += "\n"
question1 += str(store2)
file = open("question1.txt", "w+")
file.write(question1)
file.close()

q2W = []
q2S = []
for a in range(549):
    q2S.append(store2[a+1] - store2[a])
    q2W.append(store1[a+1] - store1[a])
question2 = ""
question2 += str(q2W)
question2 += "\n"
question2 += str(q2S)
file = open("question2.txt", "w+")
file.write(question2)
file.close()
# ***************************question 4**************************
# inorder to make sure the code run successfully, comment line 108-135 when you want to test q5-9
def get_cumulative_time_series(state):
    time_series = np.empty((0, 551), int)
    for line in data:
        line = line.strip('\n').split(',')
        if line[6] == state:
            append = [int(ele) for ele in line[14:]]
            time_series = np.append(time_series, [append], axis=0)
    return np.sum(time_series, axis=0)
states = []
for a in data:
    a = a.strip('\n').split(',')
    s = a[6] # state
    # only get the main states
    if s not in states and s != "American Samoa" and s != "Diamond Princess" and s != "Grand Princess" and s != "Guam" \
            and s != "Northern Mariana Islands" and s != "Virgin Islands" \
            and s != "District of Columbia" and s != "Rhode Island":
        states.append(s)
all_state = ""
for s in states:
    temp = get_cumulative_time_series(s)
    df = pd.DataFrame(temp)
    var = str(int(df.var()[0]))
    mean = str(int(df.mean()[0]))
    max = str(int(df.max()[0]))
    med = str(int(df.median()[0]))
    sum = str(int(df.sum()[0]))
    all_state += var + "," + mean + "," + max + "," + med + "," + sum + "\n"
    # print(all_state)


d_dict = {}
test_list = []
jlgl = 0
for d in data:
    l = d.strip('\n').split(',')
    c = l[6]  # state
    d_dict[c] = [float(l[0])], [float(l[4])], [float(l[8])], [float(l[9])], [float(l[-1])]
d_dict = {k:np.array([(sum(v[0])/len(v[0])), (sum(v[1])/len(v[1])), (sum(v[2])/len(v[2])), (sum(v[3])/len(v[3])), (sum(v[4]))]) for k,v in d_dict.items()}
state = sorted([c for c in d_dict.keys()])

def manhattan(a,b):
    return sum(abs(a[i]-b[i]) for i in range(len(a)))

def Euclidean(a,b):
    # return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))
    # return sum(abs(np.linalg.norm(a[i]-b[i])) for i in range(len(a)))
    # a = np.array(a)
    # b = np.array(b)
    # return norm(a-b)
    return np.linalg.norm(a - b)


 # single linkage distance
def sld(cluster1, cluster2):
    res = float('inf')
    # c1, c2 each is a country in the corresponding cluster
    for c1 in cluster1:
        for c2 in cluster2:
            dist = Euclidean(d_dict[c1], d_dict[c2])
            if dist < res:
                res = dist
    return res

k = 5
# hierarchical clustering (sld, 'manhattan')
n = len(d_dict)
clusters = [{d} for d in d_dict.keys()]
for _ in range(n-k):
    dist = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters)-1):
        for j in range(i+1, len(clusters)):
            if sld(clusters[i], clusters[j]) < dist:
                dist = sld(clusters[i], clusters[j])
                best_pair = (i, j)
    new_clu = clusters[best_pair[0]] | clusters[best_pair[1]]
    clusters = [clusters[i] for i in range(len(clusters)) if i not in best_pair]
    clusters.append(new_clu)
question5 = ""
for i in range(len(state)):
    s = state[i]
    if s in clusters[0]:
        question5 += "0"
    if s in clusters[1]:
        question5 += "1"
    if s in clusters[2]:
        question5 += "2"
    if s in clusters[3]:
        question5 += "3"
    else:
        question5 += "4"
    if i != len(state) - 1:
        question5 += ","

# file = open("question5.txt", "w+")
# file.write(question5)
# file.close()
# 3,3,4,3,3,3,3,3,4,4,4,3,3,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3,3,3,3,4,3,3,3,3,4,3,3,3,3,3,4,3,4,3,3,3,3,3

 # single linkage distance
def cld(cluster1, cluster2):
    res = float('-inf')
    # c1, c2 each is a country in the corresponding cluster
    for c1 in cluster1:
        for c2 in cluster2:
            dist = Euclidean(d_dict[c1], d_dict[c2])
            if dist > res:
                res = dist
    return res

clusters2 = [{d} for d in d_dict.keys()]
for _ in range(n-k):
    dist = float('inf')
    best_pair = (None, None)
    for i in range(len(clusters2)-1):
        for j in range(i+1, len(clusters2)):
            if sld(clusters2[i], clusters2[j]) < dist:
                dist = sld(clusters2[i], clusters2[j])
                best_pair = (i, j)
    new_clu = clusters2[best_pair[0]] | clusters2[best_pair[1]]
    clusters2 = [clusters2[i] for i in range(len(clusters2)) if i not in best_pair]
    clusters2.append(new_clu)

question6 = ""
for i in range(len(state)):
    s = state[i]
    if s in clusters2[0]:
        question6 += "0"
    if s in clusters2[1]:
        question6 += "1"
    if s in clusters2[2]:
        question6 += "2"
    if s in clusters2[3]:
        question6 += "3"
    else:
        question6 += "4"
    if i != len(state) - 1:
        question6 += ","

## k-means (manhattan)
import copy
def center(cluster):
    return np.average([d_dict[c] for c in cluster], axis=0)
init_num = np.random.choice(len(state) - 1, k)
clusters = [{state[i]} for i in init_num]
centers = []
while True:
    new_clusters = [set() for _ in range(k)]
    centers = [center(cluster) for cluster in clusters]
    for c in state:
        clu_ind = np.argmin([Euclidean(d_dict[c], centers[i]) for i in range(k)])
        new_clusters[clu_ind].add(c)
    if all(new_clusters[i] == clusters[i] for i in range(k)):
        break
    else:
        clusters = copy.deepcopy(new_clusters)

cluster1 = clusters[0]
cluster2 = clusters[1]
cluster3 = clusters[2]
cluster4 = clusters[3]
cluster5 = clusters[4]
question7 = ""
for index in range(len(state)):
    s = state[index]
    if s in cluster1:
        question7 += "0"
    elif s in cluster2:
        question7 += "1"
    elif s in cluster3:
        question7 += "2"
    elif s in cluster4:
        question7 += "3"
    else:
        question7 += "4"
    if index != len(state) - 1:
        question7 += ","

q7 = []
for index in range(len(state)):
    s = state[index]
    if s in cluster1:
        q7.append(0)
    elif s in cluster2:
        q7.append(1)
    elif s in cluster3:
        q7.append(2)
    elif s in cluster4:
        q7.append(3)
    else:
        q7.append(4)


iii = 0
question8 = ""
for i in centers:
    for j in i:
        iii += 1
        question8 += str(round(j, 4))
        if iii % 5 != 0:
            question8 += ","
    question8 += "\n"
# file = open("question8.txt", "w+")
# file.write(str(question8))
# file.close()
q7 = []
q7.append(question7)

with open('question8.txt', 'r') as f:
    q8 = [l.strip('\n').split(',') for l in f if '?' not in l]
q8 = np.array(q8).astype(float)

# question9 = 0
# for x in range(5):
#     for y in range(5):
#         # print(q8[x][y])
#             for z in range(115):
#                 if z % 2 == 0:
#                     # print(q7[0][z])
#                     question9 += abs(float(q8[x][y]) - float(q7[0][z]))**2
# print(question9)
