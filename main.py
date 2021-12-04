# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:37:37 2021

@author: Bhavdeep Singh
"""
import numpy as np
import pandas as pd
import pdb

data = pd.read_csv('datavalue.csv')
print(data.to_string())
data['salary'] = data['salary'].map({'low': 1, 'medium': 2, 'high': 3})
data['sales'] = data['sales'].map(
    {'sales': 1, 'accounting': 2, 'technical': 3, 'support': 4, 'management': 5, 'IT': 6, 'product_mng': 7,
     'marketing': 8, 'hr': 9, 'RandD': 10})
training_per = .70
[r, c] = data.shape
data = data.values.tolist()
# pdb.set_trace();
for i in range(0, len(data) - 2):
    kc = data[i]
    #    satisfaction level
    if kc[0] <= .40:
        kc[0] = .40
    elif kc[0] > .40 and kc[0] <= .60:
        kc[0] = .60
    else:
        kc[0] = .80

    data[i] = kc

# for i in range(i,len(data)-2):
# pdb.set_trace();
cat1 = []
cat2 = []
cat3 = []
catlabels = []
for i in range(1, len(data) - 2):
    catvalue = []
    kc = data[i]
    for j in range(1, c - 3):
        catvalue.append(kc[j])

    if kc[0] == .40:
        cat1.append(kc)
        catlabels.append(1)
    elif kc[0] == .60:
        cat2.append(catvalue)
        catlabels.append(2)
    else:
        cat3.append(catvalue)
        catlabels.append(3)

import random
def mean_values(input_data):
    return sum(input_data) / len(input_data)


def genetic_index_selection(data):
    total_population = data
    index = []
    #    pdb.set_trace();
    crawlvalue = 0
    ktt = 50
    for i in total_population:

        colvalue = 0
        selectv = 0
        rejectv = 0
        for cvx in i:
            indexcount = 10
            indexvalues = []
            comparisonvalue = []
            #            pdb.set_trace();
            for jj in range(0, indexcount - 1):
                indexr = (round(ktt * random.random()))
                cv = data[indexr]
                comparisonvalue.append(cv[colvalue])
            crossovervalue = mean_values(comparisonvalue)
            mutation_rate = random.random()
            #            fitness function
            #            pdb.set_trace();
            if cvx * mutation_rate >= crossovervalue * mutation_rate:
                selectv += 1
            else:
                rejectv = rejectv + 1
            colvalue = colvalue + 1
        if selectv > rejectv:
            index.append(crawlvalue)
        crawlvalue = crawlvalue + 1
    return index


# pdb.set_trace()
import matplotlib.pyplot as plt

plt.figure()
plt.plot(cat1)
plt.plot(cat2)
plt.plot(cat3)
plt.xlabel('Number of Records')
plt.ylabel('Attribute Value')
plt.legend(labels=["Low S-Level", "Average S-Level", "High S-Level"])
plt.show()
# pdb.set_trace()
index1 = genetic_index_selection(cat1)
index2 = genetic_index_selection(cat2)
index3 = genetic_index_selection(cat3)

cat1u = []
cat2u = []
cat3u = []
# pdb.set_trace()
for kg in index1:
    cat1u.append(cat1[kg])
for kg in index2:
    cat2u.append(cat2[kg])
for kg in index3:
    cat3u.append(cat3[kg])
plt.figure()
plt.plot(index1, cat1u)
plt.plot(index2, cat2u)
plt.plot(index3, cat3u)
plt.xlabel('Selected Indexes')
plt.ylabel('Attribute Value')
plt.legend(labels=["Low S-Level-U", "Average S-Level-U", "High S-Level-U"])
plt.show()

# pdb.set_trace();
# def newmean(inputd):
#    for i in inputd:

def genepochs():
    return 15


def genab():
    #    double[] ab=new double[2];
    ab = [.1,
          .02]  # The applied Neural Network follows a linear regression model whose equation is ax+b=0  where a and b are arb. constants

    return ab


def gradient_value(input_v, epochcounter):
    try:
        ff = mean_values(input_v)
    except:
        ff = 100 * random.random()
    gradientv = ff + epochcounter / len(input_v)
    #    Gradient value is the threshold over which the data processing is dependent
    return gradientv


def updateneuronvalue(inputd):
    counterx = 0
    outputd = inputd
    for i in inputd:

        abv = genab()
        try:

            outputd[counterx] = abv[0] * i + abv[1]  # Generating Output of the Propogation
        except:
            outputd[counterx] = 10 * random.random()
        counterx = counterx + 1
    return outputd


def train_neural(inputvector):
    # Layer 2 Processing .......
    totalepoch = genepochs()

    gradient = gradient_value(inputvector, totalepoch)
    epochcounter = 0
    gradientsatisfied = 0
    #    pdb.set_trace();
    newweight = inputvector
    # Satisfying the layer condition
    # epoch is the total number of processing iterations , either the gradient of the Network should be satisfied or the total processing iteration should be less than that of the supplied iteration
    while epochcounter < totalepoch and gradientsatisfied == 0:
        #        pdb.set_trace();
        newweight = updateneuronvalue(newweight)
        gvalue = mean_values(newweight)
        if gvalue > gradient:
            gradientsatisfied = 1
        epochcounter = epochcounter + 1
    #        Output Layer
    return newweight


training_cat1 = []
training_cat2 = []
training_cat3 = []
tr1 = round(len(cat1u) * .70)
tr2 = round(len(cat2u) * .70)
tr3 = round(len(cat3u) * .70)
cat1uu = []
cat2uu = []
cat3uu = []
for i in range(0, tr1 - 1):
    cat1uu.append(cat1u[i])

for i in range(0, tr2 - 1):
    cat2uu.append(cat2u[i])

for i in range(0, tr3 - 1):
    cat3uu.append(cat3u[i])
training_cat1 = train_neural(cat1uu)
training_cat2 = train_neural(cat2uu)
training_cat3 = train_neural(cat3uu)
test_cat1 = train_neural(cat1u)
test_cat2 = train_neural(cat2u)
test_cat3 = train_neural(cat3u)
results = []
gt = []
for i in test_cat1:
    mv1 = abs(i - mean_values(training_cat1))
    mv2 = abs(i - mean_values(training_cat2))
    mv3 = abs(i - mean_values(training_cat3))
    mylist = [mv1, mv2, mv3]
    indexv = mylist.index(min(mylist))
    gt.append(0)
    results.append(indexv)
for i in test_cat2:
    mv1 = abs(i - mean_values(training_cat1))
    mv2 = abs(i - mean_values(training_cat2))
    mv3 = abs(i - mean_values(training_cat3))
    mylist = [mv1, mv2, mv3]
    indexv = mylist.index(min(mylist))
    results.append(indexv)
    gt.append(1)
for i in test_cat3:
    mv1 = abs(i - mean_values(training_cat1))
    mv2 = abs(i - mean_values(training_cat2))
    mv3 = abs(i - mean_values(training_cat3))
    mylist = [mv1, mv2, mv3]
    indexv = mylist.index(min(mylist))
    results.append(indexv)
    gt.append(2)


def calculate_p_r(results, gt):
    counter = 0
    tp = 0
    fp = 0
    for i in results:
        if i == gt[counter]:
            tp = tp + 1
        else:
            fp = fp + 1
        counter = counter + 1
    tpr = tp / counter
    fpr = fp / counter
    fmeasure = (2 * tpr * fpr) / (tpr + fpr)
    return tpr, fpr, fmeasure


import matplotlib.pyplot as pltx

pltx.figure()
pltx.plot(cat1u)
pltx.plot(cat2u)
pltx.plot(cat3u)
pltx.xlabel('Number of Selected Rows for normal')
pltx.ylabel('Trained Neural Value')
pltx.legend(labels=["Low Satisfaction", "Average Satisfaction", "High Satisfaction"])
pltx.show()
[precision, recall, fmeasure] = calculate_p_r(results, gt)
print('Precision ' + str(precision))
print('RECALL ' + str(recall))
print("F-measure " + str(fmeasure))