# most of codes are from TA's solution
# written by Zhiwei Cao
from collections import Counter, OrderedDict
from decimal import Decimal
from itertools import product
import matplotlib.pyplot as plt
from random import choices

import numpy as np
import string
import sys
import re
import math

# in this piece of code, I leave out a bunch of thing for you to fill up modify.
# The current code may run into a ZeroDivisionError. Thus, you need to add Laplace first.
'''
Todo: 
1. Laplace smoothing
2. Naive Bayes prediction
3. All the output.

'''

with open('Thor.txt', encoding='utf-8') as f:
    data = f.read()
# len(data)

data = data.lower()
data = data.translate(str.maketrans('', '', string.punctuation))
data = re.sub('[^a-z]+', ' ', data)
data = ' '.join(data.split(' '))

allchar = ' ' + string.ascii_lowercase

unigram = Counter(data)
unigram_prob = {ch: round((unigram[ch]) / (len(data)), 4) for ch in allchar}
# ***********************************Question 2************************************
question2 = ""
temp2 = 0
for i in unigram_prob:
    question2 += str(unigram_prob[i])
    temp2 += 1
    if temp2 < 27:
        question2 += ","
file = open("question2.txt","w+")
file.write(question2)
file.close()
# ************************* store value of unigram in a file for question 4 **************************************
temp_Unigram = ""
temp_Unigram += str(unigram.get(' '))
temp_Unigram += ","
temp_Unigram += str(unigram.get('a'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('b'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('c'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('d'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('e'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('f'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('g'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('h'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('i'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('j'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('k'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('l'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('m'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('n'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('o'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('p'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('q'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('r'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('s'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('t'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('u'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('v'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('w'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('x'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('y'))
temp_Unigram += ","
temp_Unigram += str(unigram.get('z'))
file = open("temp_Unigram.txt","w+")
file.write(temp_Unigram)
file.close()
# ************************************ read unigram file and store value in to a ndarray ***********************************
with open('temp_Unigram.txt','r')as f:
    un = [l.strip('\n').split(',') for l in f if '?' not in l]
un = np.array(un).astype(int)

# print(un.shape)

uni_list = [unigram_prob[c] for c in allchar]
# print(uni_list)

def ngram(n):
    # all possible n-grams
    d = dict.fromkeys([''.join(i) for i in product(allchar, repeat=n)], 0)
    # update counts
    d.update(Counter([''.join(j) for j in zip(*[data[i:] for i in range(n)])]))
    return d


bigram = ngram(2)  # c(ab)
bigram_prob = {c: bigram[c] / unigram[c[0]] for c in bigram}  # p(b|a)

q3 = {c: Decimal(bigram[c] / unigram[c[0]]).quantize(Decimal("0.0001"), rounding = "ROUND_UP") for c in bigram}

# ***************************************** Question 3 *****************************************************
question3 = ""
temp3 = 0
for i in q3:
    # print(i,q3[i])
    question3 += str(q3[i])
    temp3 += 1
    if temp3 < 27*27:
        question3 += ","
    if temp3 %27 == 0:
        question3 += "\n"

file = open("question3.txt","w+")
file.write(question3)
file.close()

# ******************************* question 4 ****************************************************************
q4 = {c: Decimal(bigram[c] / unigram[c[0]]).quantize(Decimal("0.0001"), rounding = "ROUND_UP") for c in bigram}
# set a temper file to store count of bigram
temp_Bigram = ""
tempB = 0
for i in bigram:
    tempB += 1
    # print(i,bigram[i])
    temp_Bigram += str(bigram[i])
    if tempB < 27 * 27 and tempB % 27 != 0:
        temp_Bigram += ","
    if tempB % 27 == 0:
        temp_Bigram += "\n"
file = open("temp_Bigram.txt","w+")
file.write(temp_Bigram)
file.close()

with open('temp_Bigram.txt','r')as f:
    bi = [l.strip('\n').split(',') for l in f if '?' not in l]
bi = np.array(bi).astype(int)

question4 = ""
temp4 = 0
for a in range(27):
    for b in range(27):
        # print(bi[a][b],un[0][a])
        temp4 += 1
        q4_value = (bi[a][b] + 1) / (un[0][a] + 27)
        question4 += str(Decimal(q4_value).quantize(Decimal("0.0001"), rounding = "ROUND_UP"))
        if temp4 % 27 != 0:
            question4 += ","
    question4 += "\n"
file = open("question4.txt","w+")
file.write(question4)
file.close()

trigram = ngram(3)
trigram_prob = {c: (trigram[c] + 1) / (bigram[c[:2]] + 27) for c in trigram}

def gen_bi(c):
    w = [bigram_prob[c + i] for i in allchar]
    return choices(allchar, weights=w)[0]


def gen_tri(ab):
    w_tri = [trigram_prob[ab + i] for i in allchar]
    return choices(allchar, weights=w_tri)[0]


def gen_sen(c, num):
    res = c + gen_bi(c)
    for i in range(num - 2):
        if bigram[res[-2:]] == 0:
            t = gen_bi(res[-1])
        else:
            t = gen_tri(res[-2:])
        res += t
    return res

# ******************************************* Question 5 *********************************
question5 = ""
alphabet = string.ascii_lowercase
for i in range(26):
    # print(alphabet[i])
    question5 += gen_sen(alphabet[i],1000)
    question5 += "\n"
file = open("question5.txt","w+")
file.write(question5)
file.close()

with open('script.txt', encoding='utf-8') as f:
    young = f.read()

dict2 = Counter(young)
# ***********************question 7 *****************************************
likeli = [dict2[c] / len(young) for c in allchar]
# print(likeli)
# ***********************question 8 *****************************************
post_young = [round(likeli[i] * 0.43 / (likeli[i] * 0.43 + uni_list[i] * 0.57), 4) for i in range(27)]
# print(post_young)
# ***********************question 9 *****************************************
post_hugh = [1 - post_young[i] for i in range(27)]
# print(post_hugh)
for i in range(26):
    for a in range(26):
        t1 = math.log(post_young[i])
        t2 = math.log(1 - post_young[i])