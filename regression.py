import numpy as np

def process_data(filename):
	data = np.genfromtxt(filename,dtype = "str",delimiter = ",")
	samples, features = data.shape
	Y = data[:,0]
	X = data[:,1:features]
	X = X.astype(np.float)
	return X, Y.astype(np.int)

def norm(X, mean, var):
	return (X - mean)/var

def regularizer(X,Y,l,type):
	samples, features = X.shape
	w = np.zeros(features)
	b = 0.0
	learningStep = 0.001
	loss = 0.1
	delta = 1
	while delta > 0.01:
		gradw = np.zeros(features)
		gradb = 0.0
		for x,y in zip(X,Y):
			exp = np.exp(np.dot(x, w) + b)
			single_b = 0.5 * (y + 1) - exp / (exp + 1)
			gradb += single_b
			gradw += single_b * x 

		if type == 2:
			gradw -= l * w
			gradb -= l * b
		else:
			gradw -= l / 2
			if b > 0:
				gradb -= l / 2
			else:
				gradb += l / 2
		w = w + gradw * learningStep
		b = b + gradb * learningStep
		cur_loss = get_loss(X,Y,w,b)
		delta = np.absolute(cur_loss - loss)
		loss = cur_loss
	return w, b

def get_loss(X,Y,w,b):
	loss = 0.0
	for x, y in zip(X,Y):
		product = np.dot(x, w) + b
		loss += 0.5 * (y + 1) * product - np.log(1 + np.exp(product))
	return loss

def grad_descent(X,Y):
	samples, features = X.shape
	w = np.zeros(features)
	b = 0.0
	learningStep = 0.001
	loss = 0.1
	delta = 1
	while delta > 0.01:
		gradw = np.zeros(features)
		gradb = 0.0
		for x,y in zip(X,Y):
			exp = np.exp(np.dot(x, w) + b)
			single_b = 0.5 * (y + 1) - exp / (exp + 1)
			gradb += single_b
			gradw += single_b * x 
		w = w + gradw * learningStep
		b = b + gradb * learningStep
		cur_loss = get_loss(X,Y,w,b)
		delta = np.absolute(cur_loss - loss)
		loss = cur_loss
	return w, b

def predict(w, b, X):
	label = []
	for x in X:
		if 0 < np.dot(x, w) + b:
			label.append(1)
		else:
			label.append(-1)
	return label

def accuracy(label, Y):
	err = 0.0
	for l, y in zip(label, Y):
		if l != y :
			err += 1.0
	return 1.0 - err / len(Y)

def penalty(lamda, train_X, train_Y, type):
	samples, features = train_X.shape
	best_acc = 0.0
	best_lamda = 0
	best_b = 0.0
	best_w = np.zeros(features)
	for l in lamda:
		w,b = regularizer(train_X, train_Y, l, type)
		label = predict(w,b, valid_X)
		acc = accuracy(label, valid_Y)
		if acc > best_acc:
			best_acc = acc
			best_lamda = l
			best_w = w
			best_b = b
	print("best lamda", best_lamda)
	print("best w", best_w)
	print("best b", best_b)
	return best_w, best_b

if __name__ == "__main__":
	train_X, train_Y = process_data("park_train.data")
	mean = np.mean(train_X,axis = 0)
	var = np.std(train_X, axis=0)
	train_X = norm(train_X, mean, var)
	valid_X, valid_Y = process_data("park_validation.data")
	valid_X = norm(valid_X, mean, var)
	test_X, test_Y = process_data("park_test.data")
	test_X = norm(test_X, mean, var)
	w, b = grad_descent(train_X, train_Y)
	print(accuracy(predict(w, b, test_X), test_Y))
	lamda = [0.0001,0.001,0.01,0.1]
	#l2 norm
	w, b = penalty(lamda, train_X, train_Y, 2)
	print("l2 norm")
	print(accuracy(predict(w, b, test_X), test_Y))
	#l1 norm
	w, b = penalty(lamda, train_X, train_Y, 1)
	print("l1 norm")
	print(accuracy(predict(w, b, test_X), test_Y))


