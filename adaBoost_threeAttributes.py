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

def five_structures(three_attr):
	treeList = []
	root = Node(three_attr[0])
	root.left = Node(three_attr[1])
	root.right = Node(three_attr[2])
	treeList.append(root)
	root1 = Node(three_attr[0])
	root1.left = Node(three_attr[1])
	root1.left.left = Node(three_attr[2])
	root2 = Node(three_attr[0])
	root2.left = Node(three_attr[1])
	root2.left.right = Node(three_attr[2])
	root3 = Node(three_attr[0])
	root3.right = Node(three_attr[1])
	root3.right.left = Node(three_attr[2])
	root4 = Node(three_attr[0])
	root4.right = Node(three_attr[1])
	root4.right.right = Node(three_attr[2])
	treeList.append(root1)
	treeList.append(root2)
	treeList.append(root3)
	treeList.append(root4)
	return treeList

def select_Min_Err_Tree(X, weight, y):
	samples, features = X.shape
	min_error = float("inf")
	for i in range(features):
		for j in range(features):
			for k in range(features):
				three_attr = [i,j,k]
				treeList = five_structures(three_attr)
				for root in treeList:
					train_error = cal_error(root, X, weight, y)
					if train_error < min_error:
						min_error = train_error
						min_error_tree = root
	return min_error, min_error_tree

def print_tree(root):
	if root is None:
		return
	print("root label")
	print(root.label)
	print("root index")
	print(root.index)
	print_tree(root.left)
	print_tree(root.right)

def cal_error(root, X, weight, y):
	samples, fea = X.shape
	err = 0.0
	root = get_Leaf_Labels(root, X, y)
	for i in range(samples):
		if predict(X[i], root) * y[i] <= 0:########
			err += weight[i]
	return err

def predict(line, root):
	if root.index == -1:
		return np.sign(root.label)
	if line[root.index] == '0':
		return predict(line, root.left)
	else:
		return predict(line, root.right)
	return np.sign(root.label)

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

def adaBoost(round, X, y):
	print("adaBoost preparing:")
	samples,features = X.shape
	alpha = []
	weakLearners = []
	weight = np.full((samples), 1.0/samples)
	for i in range(round):
		err, tree = select_Min_Err_Tree(X, weight, y)
		#print_tree(tree)
		a = 0.5 * (np.log((1 - err) / (float)(err)))
		alpha.append(a)
		weakLearners.append(tree)
		weight = updateWeight(err, a, weight, tree, X, y)
	return alpha, weakLearners

def updateWeight(err, alpha, weight, tree, X, y):
	for i in range(len(y)):
		weight[i] = weight[i] * np.exp(-y[i] * predict(X[i], tree) * alpha) / (float)(2 * np.sqrt(err * (1 - err)))
	return weight

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
alpha, weakLearners = adaBoost(10, train_X,train_y)#
for i in range(10):
	print("Training Accuracy")
	print(getAccuracy(alpha[:i + 1],weakLearners[:i + 1], train_X,train_y))
	print("Test Accuracy")
	print(getAccuracy(alpha[:i + 1],weakLearners[:i + 1], test_X,test_y))