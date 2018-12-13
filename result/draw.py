import numpy as np
import matplotlib.pyplot as plt

def test_k():
	x = list(range(200))
	rank = 100
	xprint = list(range(100))

	test_k1 = np.load('test_hinge_k=1_lr=1.npy')
	p_k1 = np.poly1d(np.polyfit(x, test_k1, rank))
	test_k3 = np.load('test_hinge_k=3_lr=1.npy')
	p_k3 = np.poly1d(np.polyfit(x, test_k3, rank))
	test_k5 = np.load('test_hinge_k=5_lr=1.npy')
	p_k5 = np.poly1d(np.polyfit(x, test_k5, rank))
	test_k10 = np.load('test_hinge_k=10_lr=1.npy')
	p_k10 = np.poly1d(np.polyfit(x, test_k10, rank))
	test_k20 = np.load('test_hinge_k=20_lr=1.npy')
	p_k20 = np.poly1d(np.polyfit(x, test_k20, rank))
	test_k50 = np.load('test_hinge_k=50_lr=1.npy')
	p_k50 = np.poly1d(np.polyfit(x, test_k50, rank))
	test_k100 = np.load('test_hinge_k=100_lr=1.npy')
	p_k100 = np.poly1d(np.polyfit(x, test_k100, rank))
	
	plt.plot(xprint, p_k1(xprint), label='k=1')
	plt.plot(xprint, p_k3(xprint), label='k=3')
	plt.plot(xprint, p_k5(xprint), label='k=5')
	plt.plot(xprint, p_k10(xprint),label='k=10')
	plt.plot(xprint, p_k20(xprint),label='k=20')
	plt.plot(xprint, p_k50(xprint),label='k=50')
	plt.plot(xprint, p_k100(xprint),label='k=100')
	# plt.plot(list(range(200)), test_k1)
	plt.xlabel("episodes")
	plt.ylabel("test accuracy")
	plt.legend()
	plt.show()

def train_k():
	x = list(range(200))
	rank = 100
	xprint = list(range(100))

	train_k1 = np.load('train_hinge_k=1_lr=1.npy')
	p_k1 = np.poly1d(np.polyfit(x, train_k1, rank))
	train_k3 = np.load('train_hinge_k=3_lr=1.npy')
	p_k3 = np.poly1d(np.polyfit(x, train_k3, rank))
	train_k5 = np.load('train_hinge_k=5_lr=1.npy')
	p_k5 = np.poly1d(np.polyfit(x, train_k5, rank))
	train_k10 = np.load('train_hinge_k=10_lr=1.npy')
	p_k10 = np.poly1d(np.polyfit(x, train_k10, rank))
	train_k20 = np.load('train_hinge_k=20_lr=1.npy')
	p_k20 = np.poly1d(np.polyfit(x, train_k20, rank))
	train_k50 = np.load('train_hinge_k=50_lr=1.npy')
	p_k50 = np.poly1d(np.polyfit(x, train_k50, rank))
	train_k100 = np.load('train_hinge_k=100_lr=1.npy')
	p_k100 = np.poly1d(np.polyfit(x, train_k100, rank))
	
	
	plt.plot(xprint, p_k1(xprint), label='k=1')
	plt.plot(xprint, p_k3(xprint), label='k=3')
	plt.plot(xprint, p_k5(xprint), label='k=5')
	plt.plot(xprint, p_k10(xprint),label='k=10')
	plt.plot(xprint, p_k20(xprint),label='k=20')
	plt.plot(xprint, p_k50(xprint),label='k=50')
	plt.plot(xprint, p_k100(xprint),label='k=100')
	plt.xlabel("episodes")
	plt.ylabel("train accuracy")
	plt.legend()
	plt.show()

def getPolyfit(input):
	x = list(range(200))
	rank = 100
	data = np.load(input)
	return np.poly1d(np.polyfit(x, data, rank))

def train_method():
	xprint = list(range(100))
	train_hinge = getPolyfit('train_hinge_k=20_lr=1.npy')
	train_logistic = getPolyfit('train_logistic_k=20_lr=1.npy')
	train_linear = getPolyfit('train_linear_k=20_lr=1.npy')

	plt.plot(xprint, train_hinge(xprint), label='hinge')
	plt.plot(xprint, train_logistic(xprint), label='logistic')
	plt.plot(xprint, train_linear(xprint), label='linear')
	plt.xlabel("episodes")
	plt.ylabel("train accuracy")
	plt.legend()
	plt.show()

def test_method():
	xprint = list(range(100))
	test_hinge = getPolyfit('test_hinge_k=20_lr=1.npy')
	test_logistic = getPolyfit('test_logistic_k=20_lr=1.npy')
	test_linear = getPolyfit('test_linear_k=20_lr=1.npy')

	plt.plot(xprint, test_hinge(xprint), label='hinge')
	plt.plot(xprint, test_logistic(xprint), label='logistic')
	plt.plot(xprint, test_linear(xprint), label='linear')
	plt.xlabel("episodes")
	plt.ylabel("test accuracy")
	plt.legend()
	plt.show()

def train_hinge_loss():
	x = list(range(200))
	rank = 100
	xprint = list(range(100))

	loss = np.load('loss_hinge_k=20_lr=1.npy')
	f1 = loss[:,0]
	f2 = loss[:,1]
	f3 = loss[:,2]
	f4 = loss[:,3]
	f5 = loss[:,4]
	f6 = loss[:,5]
	p1 = np.poly1d(np.polyfit(x, f1, rank))
	p2 = np.poly1d(np.polyfit(x, f2, rank))
	p3 = np.poly1d(np.polyfit(x, f3, rank))
	p4 = np.poly1d(np.polyfit(x, f4, rank))
	p5 = np.poly1d(np.polyfit(x, f5, rank))
	p6 = np.poly1d(np.polyfit(x, f6, rank))

	plt.plot(xprint,p1(xprint), label='label 1')
	plt.plot(xprint,p2(xprint), label='label 2')
	plt.plot(xprint,p3(xprint), label='label 3')
	plt.plot(xprint,p4(xprint), label='label 4')
	plt.plot(xprint,p5(xprint), label='label 5')
	plt.plot(xprint,p6(xprint), label='label 6')
	plt.xlabel("episodes")
	plt.ylabel("loss")
	plt.legend()
	plt.show()

def train_lr():
	xprint = list(range(100))
	train_lr1 = getPolyfit('train_hinge_k=20_lr=1.npy')
	train_lr05 = getPolyfit('train_hinge_k=20_lr=05.npy')
	train_lr01 = getPolyfit('train_hinge_k=20_lr=01.npy')
	train_lr005 = getPolyfit('train_hinge_k=20_lr=005.npy')
	plt.plot(xprint,train_lr1(xprint), label='lr = 1')
	plt.plot(xprint,train_lr05(xprint), label='lr = 0.5')
	plt.plot(xprint,train_lr01(xprint), label='lr = 0.1')
	plt.plot(xprint,train_lr005(xprint), label='lr = 0.05')
	plt.xlabel("episodes")
	plt.ylabel("train accuracy")
	plt.legend()
	plt.show()

def test_lr():
	xprint = list(range(100))
	test_lr1 = getPolyfit('test_hinge_k=20_lr=1.npy')
	test_lr05 = getPolyfit('test_hinge_k=20_lr=05.npy')
	test_lr01 = getPolyfit('test_hinge_k=20_lr=01.npy')
	test_lr005 = getPolyfit('test_hinge_k=20_lr=005.npy')
	plt.plot(xprint,test_lr1(xprint), label='lr = 1')
	plt.plot(xprint,test_lr05(xprint), label='lr = 0.5')
	plt.plot(xprint,test_lr01(xprint), label='lr = 0.1')
	plt.plot(xprint,test_lr005(xprint), label='lr = 0.05')
	plt.xlabel("episodes")
	plt.ylabel("test accuracy")
	plt.legend()
	plt.show()

def train_loss():
	xprint = list(range(100))
	train_loss0 = getPolyfit('train_hinge_k=20_lr=1.npy')
	train_loss005 = getPolyfit('train_hinge_k=20_lr=1_lost=005.npy')
	train_loss01 = getPolyfit('train_hinge_k=20_lr=1_lost=01.npy')
	train_loss02 = getPolyfit('train_hinge_k=20_lr=1_lost=02.npy')
	train_loss03 = getPolyfit('train_hinge_k=20_lr=1_lost=03.npy')
	train_loss05 = getPolyfit('train_hinge_k=20_lr=1_lost=05.npy')
	plt.plot(xprint,train_loss0(xprint), label='missing rate = 0')
	plt.plot(xprint,train_loss005(xprint), label='missing rate = 0.05')
	plt.plot(xprint,train_loss01(xprint), label='missing rate = 0.1')
	plt.plot(xprint,train_loss02(xprint), label='missing rate = 0.2')
	plt.plot(xprint,train_loss03(xprint), label='missing rate = 0.3')
	plt.plot(xprint,train_loss05(xprint), label='missing rate = 0.5')
	plt.xlabel("episodes")
	plt.ylabel("train accuracy")
	plt.legend()
	plt.show()

def test_loss():
	xprint = list(range(100))
	test_loss0 = getPolyfit('test_hinge_k=20_lr=1.npy')
	test_loss005 = getPolyfit('test_hinge_k=20_lr=1_lost=005.npy')
	test_loss01 = getPolyfit('test_hinge_k=20_lr=1_lost=01.npy')
	test_loss02 = getPolyfit('test_hinge_k=20_lr=1_lost=02.npy')
	test_loss03 = getPolyfit('test_hinge_k=20_lr=1_lost=03.npy')
	test_loss05 = getPolyfit('test_hinge_k=20_lr=1_lost=05.npy')
	plt.plot(xprint,test_loss0(xprint), label='missing rate = 0')
	plt.plot(xprint,test_loss005(xprint), label='missing rate = 0.05')
	plt.plot(xprint,test_loss01(xprint), label='missing rate = 0.1')
	plt.plot(xprint,test_loss02(xprint), label='missing rate = 0.2')
	plt.plot(xprint,test_loss03(xprint), label='missing rate = 0.3')
	plt.plot(xprint,test_loss05(xprint), label='missing rate = 0.5')
	plt.xlabel("episodes")
	plt.ylabel("test accuracy")
	plt.legend()
	plt.show()