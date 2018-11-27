import numpy as np
class Node(object):
	def __init__(self, index):
		self.index = index
		self.left = None
		self.right = None
		self.label = 0.0

def process_labels(Y):
	process = []
	for y in Y:
		if y == '0':
			process.append(-1.0)
		else:
			process.append(1.0)
	return process

def build_one_attribute_tree(X, y):
	samples, features = X.shape
	treeList = []
	for i in range(features):
		root = get_Leaf_Labels(Node(i), X, y)
		treeList.append(root)
	return treeList

def get_prediction_matrix(line, treeList):
	pred = []
	for tree in treeList:
		pred.append(predict(line, tree))
	return pred 
def coordinate_descent(treeList, X, y):
	samples, features = X.shape
	alpha = np.full((features), 1.0/features)
	for time in range(400):
		exponential_loss = 0.0
		for i in range(samples):
			exponential_loss += np.exp(-y[i] * np.sum(np.multiply(alpha, get_prediction_matrix(X[i], treeList))))

		for i in range(features):
			correct = 0.0
			wrong = 0.0
			for j in range(samples):
				current_pred = predict(X[j], treeList[i])
				exp_sum = -y[j] * (np.sum(np.multiply(alpha, get_prediction_matrix(X[j], treeList))) - alpha[i] * current_pred)
				if current_pred == y[j]:
					correct += np.exp(exp_sum)
				else:
					wrong += np.exp(exp_sum)
			alpha[i] = 0.5 * np.log(correct / wrong)
	print("loss")
	print(exponential_loss)
	print("alpha")
	print(alpha)	
	return alpha

def predict(line, root):

	if root.index == -1:
		return np.sign(root.label)
	if line[root.index] == '0':
		return predict(line, root.left)
	else:
		return predict(line, root.right)
	return root.index

def get_Leaf_Labels(root, X, y):
	samples, features = X.shape
	for i in range(samples):
		node = root
		prev = root
		while node is not None and node.index != -1:
			prev = node
			if X[i][node.index] == '0':
				node = node.left
			else:
				node = node.right 
		if prev.left is None:
			prev.left = Node(-1)
			node = prev.left
		if prev.right is None:
			prev.right = Node(-1)
			node = prev.right
		node.label = node.label + y[i]
	return root

def getAccuracy(alpha, weakLearners, X, y):
	samples, features = X.shape
	correct = 0.0
	for i in range(samples):
		label = 0.0
		for j in range(len(weakLearners)):
			label += alpha[j] * predict(X[i], weakLearners[j])
		if label * y[i] > 0:
			correct += 1.0
	return correct / samples

train = np.genfromtxt("heart_train.data",dtype = "str",delimiter = ",")
test = np.genfromtxt("heart_test.data",dtype = "str",delimiter = ",")
train_X = train[:, 1:]
test_X = test[:, 1:]
test_y = process_labels(test[:, 0])
train_y = process_labels(train[:, 0])
weakLearners = build_one_attribute_tree(train_X, train_y)
samples, features = train_X.shape

alpha = coordinate_descent(weakLearners, train_X, train_y)#corrdinate
print(getAccuracy(alpha, weakLearners, train_X,train_y))
print(getAccuracy(alpha, weakLearners, test_X,test_y))