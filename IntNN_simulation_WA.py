from __future__ import absolute_import, division, print_function, unicode_literals
import os
from tqdm import tqdm
import datetime
import tensorflow as tf
os.environ['TF_KERAS'] = '1'
from tensorflow import keras
from keras_bert import AdamWarmup, calc_train_steps
from scipy import stats
from tensorflow.keras import optimizers
import random
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from keras_radam.training import RAdamOptimizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from simulation_WA_NN import Simulation_8, Simulation_32, Simulation_64, Simulation_64_2

#%%
num_feature =32
rules = 4
num_sample = 10000
sparsity = 20
if num_sample == 200:
    batch = 32
elif num_sample == 500:
    batch = 50
elif num_sample == 1000:
    batch = 100
elif num_sample == 10000:
    batch = 1000

X = np.load(os.path.join('simu_data','WA{}_X_{}_{}.npy'.format(num_sample, num_feature,sparsity)))
y = np.load(os.path.join('simu_data','WA{}_Y_{}_{}.npy'.format(num_sample, num_feature,sparsity)))

column_list = ['Feature ' + str(x) for x in range(1, num_feature + 1)]
rule_length = np.int(X.shape[1]/rules)
#%% 5-Fold CV
k_num = 5
kf = KFold(n_splits=k_num, shuffle=False)

write = pd.ExcelWriter('Simulation_WA_{}_{}_{}_5CV.xlsx'.format(num_feature,sparsity,num_sample))
accuracy = np.zeros(k_num)
type_one = np.zeros(k_num)
type_two = np.zeros(k_num)
alpha = np.zeros([rules,X.shape[1]])
result_index = []
result = []
k = 0
plt.close('all')


for train_index, test_index in tqdm(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scalar = preprocessing.StandardScaler().fit(X_train)
    X_train = scalar.transform(X_train)
    X_test = scalar.transform(X_test)

    if num_feature == 8:
        model = Simulation_8()
    elif num_feature == 32:
        model = Simulation_32()
    elif num_feature == 64:
        model = Simulation_64()

    total_steps, warmup_steps = calc_train_steps(
        num_example=X_train.shape[0],
        batch_size=batch,
        epochs=1000,
        warmup_proportion=0.1,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-1, min_lr=1e-3)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch, epochs=3000)

    loss = model.history.history['loss']

    alpha_list = [model.rule1_0.alpha.value().numpy().flatten(),
                      model.rule2_0.alpha.value().numpy().flatten(),
                      model.rule3_0.alpha.value().numpy().flatten(),
                      model.rule4_0.alpha.value().numpy().flatten(),
                      # model.rule5_0.alpha.value().numpy().flatten(),
                      # model.rule6_0.alpha.value().numpy().flatten(),
                      # model.rule7_0.alpha.value().numpy().flatten(),
                      # model.rule8_0.alpha.value().numpy().flatten()
                  ]

    for r in range(rules):
        alpha[r, np.int(r * rule_length):np.int((r + 1) * rule_length)] = alpha_list[r]

    alpha_re = scalar.inverse_transform(alpha)
    alpha_re[np.where(alpha == 0)] = 0

    weights = np.asarray(model.outlayer.get_weights()).flatten()
    alpha_all = np.concatenate((alpha, alpha_re), axis=0)
    df = pd.DataFrame(alpha_all,columns=column_list)


    df.to_excel(write, sheet_name='Alpha Value Fold{}'.format(k + 1))


    plt.title('Loss in Traning for Full samples')
    plt.plot(np.array(range(len(loss))), loss)
    plt.show()

    print(model.evaluate(X_test, y_test))
    # train_accuracy[k] = model.evaluate(X_train, y_train)[1]
    accuracy[k] = model.evaluate(X_test, y_test)[1]
    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5)
    cnf_matrix = metrics.confusion_matrix(y[test_index], y_pred)
    result_index.append(test_index)
    result.append(y_pred)

    k += 1

result_index = list(itertools.chain.from_iterable(result_index))
result = list(itertools.chain.from_iterable(result))

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z

result = sort_list(result, result_index)
write.save()
print('5-Fold Cross Validation Result', '\n',metrics.confusion_matrix(y, result))
accuracy_all = np.average(accuracy)
print('Average accuracy for 5 fold cross validation is {0:.3f}'.
        format(accuracy_all))#,type_one_all,type_two_all
print('SE for 5 fold cross validation is{0:.3f}'.
        format(stats.sem(accuracy)))

#%% Iterative estimation for all samples

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
iter = 10
write2 = pd.ExcelWriter('Simulation_WA_{}_{}_{}_all_fn.xlsx'.format(num_feature,sparsity, num_sample))
RAdam = RAdamOptimizer(learning_rate=1e-3)
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)  # , decay=1e-6
alpha_all_org = np.zeros((iter,rules,X.shape[1]))
alpha_all_5 = np.zeros((iter,rules,X.shape[1]))

for i in tqdm(range(iter)):

    if num_feature == 8:
        model = Simulation_8()
    elif num_feature == 32:
        model = Simulation_32()
    elif num_feature == 64:
        model = Simulation_64()
    alpha = np.zeros([rules,X.shape[1]])

    total_steps, warmup_steps = calc_train_steps(
        num_example=X.shape[0],
        batch_size=batch,
        epochs=500,
        warmup_proportion=0.1,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-1, min_lr=1e-3)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=batch, epochs=3000)

    alpha_list = [model.rule1_0.alpha.value().numpy().flatten(),
                  model.rule2_0.alpha.value().numpy().flatten(),
                  model.rule3_0.alpha.value().numpy().flatten(),
                  model.rule4_0.alpha.value().numpy().flatten(),
                  # model.rule5_0.alpha.value().numpy().flatten(),
                  # model.rule6_0.alpha.value().numpy().flatten(),
                  # model.rule7_0.alpha.value().numpy().flatten(),
                  # model.rule8_0.alpha.value().numpy().flatten(),
                  ]

    for r in range(rules):
        # alpha[r, np.int(r*rule_length):np.int((r+1)*rule_length)] = model.rule1_0.alpha.value().numpy().flatten()
        alpha[r, np.int(r*rule_length):np.int((r+1)*rule_length)] = alpha_list[r]

    alpha_re = scaler.inverse_transform(alpha)
    alpha_re[np.where(alpha == 0)] = 0
    alpha_all = np.concatenate((alpha, alpha_re), axis=0)
    df = pd.DataFrame(alpha_all, columns=column_list
                                          )
    alpha_all_org[i] = alpha
    alpha_all_5[i] = alpha_re
    df.to_excel(write2, sheet_name='Alpha Value All Iter {}'.format(i+1))

alpha_mean_org = np.mean(alpha_all_org,axis=0)
alpha_std_org = np.std(alpha_all_org, axis=0)
alpha_cv_org = np.abs(np.divide(alpha_std_org, alpha_mean_org, out=np.zeros_like(alpha_std_org), where=alpha_mean_org!=0))
alpha_se_org = stats.sem(alpha_all_org, axis=0)
alpha_mean_iter = np.mean(alpha_all_5, axis=0)

df = pd.DataFrame(alpha_mean_org, columns=column_list)
df2 = pd.DataFrame(alpha_std_org, columns=column_list)
df3 = pd.DataFrame(alpha_cv_org, columns=column_list)
df4 = pd.DataFrame(alpha_se_org, columns=column_list)
df5 = pd.DataFrame(alpha_mean_iter, columns=column_list)
df = pd.concat([df, df2, df3, df4, df5], keys=['Org Mean', 'Org Std', 'Org CV', 'Org SE','All Mean'])

df.to_excel(write2, sheet_name='Summary')
write2.save()
