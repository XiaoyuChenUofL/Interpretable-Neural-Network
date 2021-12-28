import numpy as np
import os
from sklearn import metrics

def sigmoid(x):
    return 1/(1 + np.exp(-x))
#%% Simualtion setting with correlated feature data
### Weighted Average


#%% RULE WITH INN THRESHOLD - 64
num_feature = 64
num_rules = 8

w_1 = np.asarray([[0.9, 0.2, 0.6, 0.8, 0.5, 0.9, 0.2, 0.6],]*num_rules)#
# b_1 = [-3.6, -3.7, -3.6, -3.7,-4, -4, -4, -4]
# b_1 = [-3.5, -3.7, -3.4, -3.7,-3.5, -4, -3.7, -3.3]
b_1 = [-3.7, -3.7, -4, -4,-3.7, -3.7, -4, -4]
w_2 = np.ones(num_rules)/num_rules
l2valve = 0.5
num_sample = 10000
sparsity = 50
rou = 0.5
sparse_list = list(range(0,num_feature,2))
# sparse_list = [0, 2, 8, 10, 16, 18, 24, 26, 32, 34, 40, 48, 56 ]
cov_list = np.random.randint(8,9,size=num_feature) # Feature distribution

X = np.load(os.path.join('simu_data','WA{}_X_{}_{}.npy'.format(num_sample, num_feature,sparsity)))
y_true = np.load(os.path.join('simu_data','WA{}_Y_{}_{}.npy'.format(num_sample, num_feature,sparsity)))

alpha = np.asarray(cov_list, dtype=np.float32)*(-6)
sparse_list = np.sort(sparse_list)
alpha[sparse_list] = [0.574647626,0.390761924,0.771036312,0.586504598,0.173612505,0.530871877,0.26519231,
                      0.825404983,0.365754837,0.880540778,0.58654933,0.424116803,0.722942567,0.601137857,
                      0.228923273,0.344070102,0.578299325,0.385088502,0.791256933,0.359282344,0.161798579,
                      0.480870889,0.256273179,0.775649625,0.39453523,0.901951461,0.492515519,0.257228702,
                      0.764680937,0.632448061,0.18587887,0.314787163]

# alpha[sparse_list] = [0.6, 0.4, 0.8, 0.6, 0.2, 0.5, 0.3, 0.7, 0.4, 0.9, 0.6, 0.4, 0.8, 0.6, 0.2, 0.3,
#                       0.6, 0.4, 0.8, 0.6, 0.2, 0.5, 0.3, 0.7, 0.4, 0.9, 0.6, 0.4, 0.8, 0.6, 0.2, 0.3] #
alpha = alpha.reshape((num_rules,np.int(num_feature/num_rules))) # decided by rules
X = X.reshape((num_sample,num_rules,np.int(num_feature/num_rules)))
Y = np.zeros(num_sample)
z1 = np.zeros((num_sample,num_rules,np.int(num_feature/num_rules)))
z2 = np.zeros((num_sample,num_rules))
### Rule Classification Results
for k in range(X.shape[0]):
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if X[k][i][j] >= alpha[i][j]:
                z1[k][i][j] = 1
        z2[k][i] = sigmoid(np.dot(w_1[i],z1[k][i]) + b_1[i])
        if z2[k][i]>=0.5:
            z2[k][i] = 1
        else:
            z2[k][i] = 0
    # Y[k] = sigmoid(np.dot(w_2,z2[k]) + b_2)
    Y[k] = np.dot(w_2, z2[k])
    if Y[k] >= l2valve:
        Y[k] = 1
    else:
        Y[k] = 0

print(metrics.confusion_matrix(y_true,Y))
print(metrics.accuracy_score(y_true,Y))

#%% RULE WITH INN THRESHOLD - 32
num_feature = 32
num_rules = 4

w_1 = np.asarray([[0.9, 0.2, 0.6, 0.8, 0.5, 0.9, 0.2, 0.6],]*num_rules)#
b_1 = [-3.2, -3.6, -3.2, -3.6]
w_2 = np.ones(num_rules)/num_rules
l2valve = 0.5
num_sample = 10000
sparsity = 50

sparse_list = list(range(0,num_feature,2))
# sparse_list = [0, 2, 8, 16, 24, 26 ]
cov_list = np.random.randint(8,9,size=num_feature) # Feature distribution

X = np.load(os.path.join('simu_data','WA{}_X_{}_{}.npy'.format(num_sample, num_feature,sparsity)))
y_true = np.load(os.path.join('simu_data','WA{}_Y_{}_{}.npy'.format(num_sample, num_feature,sparsity)))

alpha = np.asarray(cov_list, dtype=np.float32)*(-6)
sparse_list = np.sort(sparse_list)
alpha[sparse_list] = [0.609136482,0.39730603,0.74125178,0.555692655,0.243446829,0.527634331,
0.328491102,0.40097924,0.427553125,0.893825719,0.633354501,0.389677165,0.805330391,0.615144203,0.22196668,0.280992105]

# alpha[sparse_list] = [0.6, 0.4, 0.8, 0.6, 0.2, 0.5] #
alpha = alpha.reshape((num_rules,np.int(num_feature/num_rules))) # decided by rules
X = X.reshape((num_sample,num_rules,np.int(num_feature/num_rules)))
Y = np.zeros(num_sample)
z1 = np.zeros((num_sample,num_rules,np.int(num_feature/num_rules)))
z2 = np.zeros((num_sample,num_rules))
### Rule Classification Results
for k in range(X.shape[0]):
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if X[k][i][j] >= alpha[i][j]:
                z1[k][i][j] = 1
        z2[k][i] = sigmoid(np.dot(w_1[i],z1[k][i]) + b_1[i])
        if z2[k][i]>=0.5:
            z2[k][i] = 1
        else:
            z2[k][i] = 0
    # Y[k] = sigmoid(np.dot(w_2,z2[k]) + b_2)
    Y[k] = np.dot(w_2, z2[k])
    if Y[k] >= l2valve:
        Y[k] = 1
    else:
        Y[k] = 0

print(metrics.confusion_matrix(y_true,Y))
print(metrics.accuracy_score(y_true,Y))

#%% RULE WITH INN THRESHOLD - 8
num_feature = 8
num_rules = 4

w_1 = np.asarray([[0.9, 0.2],]*num_rules)#
b_1 = [-0.6,-0.6,-0.6,-0.6]
w_2 = np.ones(num_rules)/num_rules
l2valve = 0.5
num_sample = 10000
sparsity = 50

sparse_list = list(range(0,num_feature,2))
# sparse_list = [0, 4 ]
cov_list = np.random.randint(8,9,size=num_feature) # Feature distribution

X = np.load(os.path.join('simu_data','WA{}_X_{}_{}.npy'.format(num_sample, num_feature,sparsity)))
y_true = np.load(os.path.join('simu_data','WA{}_Y_{}_{}.npy'.format(num_sample, num_feature,sparsity)))

alpha = np.asarray(cov_list, dtype=np.float32)*(-6)
sparse_list = np.sort(sparse_list)
alpha[sparse_list] = [0.605048696,0.398168598,0.801407115,0.605760121]

# alpha[sparse_list] = [0.6, 0.4] #
alpha = alpha.reshape((num_rules,np.int(num_feature/num_rules))) # decided by rules
X = X.reshape((num_sample,num_rules,np.int(num_feature/num_rules)))
Y = np.zeros(num_sample)
z1 = np.zeros((num_sample,num_rules,np.int(num_feature/num_rules)))
z2 = np.zeros((num_sample,num_rules))
### Rule Classification Results
for k in range(X.shape[0]):
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if X[k][i][j] >= alpha[i][j]:
                z1[k][i][j] = 1
        z2[k][i] = sigmoid(np.dot(w_1[i],z1[k][i]) + b_1[i])
        if z2[k][i]>=0.5:
            z2[k][i] = 1
        else:
            z2[k][i] = 0
    # Y[k] = sigmoid(np.dot(w_2,z2[k]) + b_2)
    Y[k] = np.dot(w_2, z2[k])
    if Y[k] >= l2valve:
        Y[k] = 1
    else:
        Y[k] = 0

print(metrics.confusion_matrix(y_true,Y))
print(metrics.accuracy_score(y_true,Y))


