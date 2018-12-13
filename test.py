import numpy as np
import data_process as dp
import learning
import math

ftrain, ltrain = dp.getdata('scene_train')
Xtest, Ytest = dp.getdata('scene_test')

X = ftrain
Y = ltrain
d = ftrain.shape[0]
k = 20
lr = 1
loss = 0.3
ltrain = dp.unlabeldata(ltrain,loss)
W = np.random.rand(d,k)
total_episode = 200
# 'Hinge', 'Logistic', 'Linear'
method = 'Hinge'
totalLoss = np.empty(0)
test_correct = np.empty(0)
train_correct = np.empty(0)

for i in range(total_episode):
    print(i)
    H, lossArray = learning.minh(ltrain, ftrain, W, 0.001, k, method)
    if method != 'Linear':
    	print(lossArray)

    totalLoss = np.append(totalLoss, lossArray)
    W = learning.minW(ltrain, ftrain,H,W,0.001, k, 10, lr / math.ceil((i+1)/10), method)

    train_correct = np.append(train_correct,1 - learning.error_rate(Y,X,W,H))
    print("Train correct rate:\t" + str(train_correct[-1]))

    test_correct = np.append(test_correct,1 -learning.error_rate(Ytest,Xtest,W,H))
    print("Test correct rate:\t" + str(test_correct[-1]))


#if method != 'Linear':
#    totalLoss = np.reshape(totalLoss,(total_episode, 6))
#np.save("loss_hinge_k=20_lr=1.npy",totalLoss)
#np.save("train_hinge_k=20_lr=1_lost=03.npy",train_correct)
#np.save("test_hinge_k=20_lr=1_lost=03.npy",test_correct)
