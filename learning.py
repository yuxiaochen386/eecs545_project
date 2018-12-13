import numpy as np
import time

def square_h(A, y_input, lam):
	y = y_input.copy()
	y[y == 0] = -1   
	B = 2*(A.transpose().dot(A)+y.shape[0]*lam*np.identity(A.shape[1]))
	r = -2*A.transpose().dot(y)
	return -np.linalg.inv(B).dot(r)

def logistic_grad(Ai, y, h, n):
	Ai = Ai.reshape(1,Ai.shape[0])
	dJi = ((-y * np.exp(-y*np.dot(Ai,h.T))[0,0])/(1+np.exp(-y*np.dot(Ai,h.T))[0,0])) * Ai / n
	return dJi

def hinge_grad(Ai, y, h, n):
	if y * (Ai @ h.T) >= 1:
		dJi = np.zeros(h.shape)
	else:
		dJi =  - (1 / n * y) * Ai
	return dJi

def other_h(A, y_input, lam, h, isHinge):
	y = y_input.copy()
	y[y == 0] = -1
	d = 294
	n = y.shape[0]
	k = A.shape[1]
	L = 6
	h_old = h.copy()
	h_old = h_old.reshape((1,k))
	iteration = 0
	delta_h = float('inf')
	threshold = 1e-5
	J = []

#gradient descent
	while delta_h > threshold and iteration < 200:
		iteration = iteration + 1
		loss = 0.5*lam*(np.linalg.norm(h_old)**2)
		grad = lam * h_old
		for i in range (n):
			if not isHinge:
				Ji = np.log(1 + np.exp(-y[i] * np.dot(h_old, A[i,:].reshape((k,1)))))
			else:
				Ji = max(0, 1 - np.int64(y[i]) * A[i,:] @ h_old.T)
			loss = loss + Ji
			if not isHinge:
				dJi = logistic_grad(A[i,:], np.int64(y[i]), h_old, n)
			else:
				dJi = hinge_grad(A[i,:], np.int64(y[i]), h_old, n)
			grad = grad + dJi
		J.append(loss)
		h_new = h_old - 0.0001 / iteration * grad
		delta_h = np.linalg.norm(h_new - h_old)
		h_old = h_new
	return h_old, J[-1]
	
	
def minh(Y,X,W,lam,k, method):
	H = np.random.randn(Y.shape[1], k) / 1000
	lossArray = np.empty(0)
	for i in range(Y.shape[1]):
		y = Y[:,i]
		Xtemp = np.delete(X, np.argwhere(y == -1), 1)
		y = np.delete(y, np.argwhere(y==-1),0)
		A = np.transpose(Xtemp).dot(W)
		if method == 'Linear':
			H[i] = square_h(A, y, lam)
		else:
			isHinge = method == 'Hinge'
			newh, loss = other_h(A, y, lam, H[i], True)
			H[i] = newh
			lossArray = np.append(lossArray, loss)
	return H, lossArray

def square_W(Y_input, A, H, i, j):
	Y = Y_input.copy()
	Y[Y == 0] = -1
	return -2*(Y[i,j]-A[i,:].reshape(A[i,:].shape[0],1).transpose().dot(H[j,:].reshape(H[j,:].shape[0],1)))


def logistic_W(Y, A, H, i, j):
	y = Y[i,j]
	if y == 0:
		y= -1
	ah = A[i,:].reshape(A[i,:].shape[0],1).transpose().dot(H[j,:].reshape(H[j,:].shape[0],1))
	return (-y * np.exp(-y * ah) / (1 + np.exp(- y * ah))) 

def hinge_W(Y, A, H, i, j):
	y = Y[i,j]
	if y == 0:
		y = -1
	ah = A[i,:].reshape(A[i,:].shape[0],1).transpose().dot(H[j,:].reshape(H[j,:].shape[0],1))
	if 1 - y * ah > 0:
		return - y 
	else:
		return 0

def minW(Y,X,H,W,lam,k,itern,lr,method):
	A = X.transpose().dot(W)
	#w = W.transpose().reshape(W.size,1)
	D = np.empty((Y.shape[0],Y.shape[1]))
	for n in range(itern):
		for i in range(Y.shape[0]):
			for j in range(Y.shape[1]):
				if (Y[i,j]==-1):
					D[i,j] = 0
				else:
					if method == 'Linear':
						D[i,j] = square_W(Y,A,H,i,j)
					elif method == 'Logistic':
						D[i,j] = logistic_W(Y,A,H,i,j)
					elif method == 'Hinge':
						D[i,j] = hinge_W(Y,A,H,i,j)
		W = W-lr*(X.dot(D.dot(H))+lam*W)
	return W

def error(Y,X,W,H,lam):
	Z = W.dot(H.transpose())
	return (np.linalg.norm(Y-X.transpose().dot(Z))**2+(np.linalg.norm(W)**2+np.linalg.norm(H)**2)*lam/2)/(Y.shape[0]*Y.shape[1])

def error_rate(Y,X,W,H):
	Z = W.dot(H.transpose())
	Yp = X.transpose().dot(Z)
	for i in range(Yp.shape[0]):
		flag = False
		Yi = Yp[i,:]
		for j in range(Yi.shape[0]):
			if Yi[j] >= 0:
				flag = True
				Yp[i,j] = 1
		if flag == False:
			maxValue = max(Yp[i,:])
			for j in range(Yi.shape[0]):
				if Yp[i,j] == maxValue:
					Yp[i,j] = 1
				else:
					Yp[i,j] = 0
		if flag == True:
			for j in range(Yi.shape[0]):
				if Yp[i,j] < 0:
					Yp[i,j] = 0

	sum = 0
	for i in range(Yp.shape[0]):
		if np.array_equal(Yp[i,:], Y[i,:]):
			sum += 1
	return 1 - 1.0 * sum / Y.shape[0]