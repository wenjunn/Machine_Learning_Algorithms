import numpy as np
from pprint import pprint
from collections import defaultdict
#calculate entropy H(Y)= -p1logp1 - p2logp2

def entropy(rows):
    res = occurance(rows)
    ent = 0.0
    for r in res.keys():
        p = float(res[r])/len(rows) 
        ent -= p * np.log2(p)
    return ent
def occurance(X):
    res = {}
    for key in X:
        res[key] = res.get(key, 0) + 1
    return res
def conditional(X,Y,x):
    sub_Y = []
    for i in range(len(X)):
        if X[i] == x:
            sub_Y.append(Y[i])
    return sub_Y
   
def infomatrion_gain(Y,X):
    conditional_entropy = entropy(Y)
    res = occurance(X)
    for x in res.keys():
        p = float(res[x])/len(X)
        sub_Y = conditional(X,Y,x)
        conditional_entropy -= p * entropy(sub_Y)
    return conditional_entropy
    
def best_attribute(X,y):
    gain = np.array([infomatrion_gain(y,x_attr) for x_attr in np.transpose(X)])
    gain = gain[::-1]
    return gain.size - np.argmax(gain) - 1
def majority_vote(y):
    count = occurance(y)
    return max(count,key=count.get)
def divide_Tree(X,y, idx):
    sub = {}
    for x, label in zip(X,y):
        attribute_val = x[idx]
        sub_X, sub_Y = sub.setdefault(attribute_val,[[],[]])
        sub_X.append(np.concatenate((x[:idx], x[idx:]), axis=None))
        sub_Y.append(label)
    return sub, sub_Y


def build_Tree(X,y,attribute):
    if len(set(y)) == 1:
        return y[0]
    #if there is no feature to split, return the label with majority vote
    if len(attribute) == 0:
        return majority_vote(y)
    best_attr = best_attribute(X,y)
    split_dict,sub_Label = divide_Tree(X,y,best_attr)
    tree={}
    tree[best_attr]={}
    for attr_val, (X_sub,y_sub) in split_dict.items():
        tree[best_attr][attr_val] = build_Tree(X_sub,y_sub,attribute)

    return tree

def traverse(X,tree):
    if isinstance(tree,str):
        return tree
    else:
        feature = tree.keys()[0]#Node
        value = X[feature]
        sub_tree = tree[feature][value]
        return traverse(X, sub_tree)

def predict(X,tree):
    label = []
    for x in X:
        label.append(traverse(x,tree))
    return label
def partition(a):
    return {c: (a == c).nonzero()[0] for c in np.unique(a)}


def cal_one_attr_accuracy(y, X, feature):
    sets = partition(X[:, feature])
    count = 0
    for k,v in sets.items():
        y_subset = y.take(v, axis=0)
        if len(set(y_subset)) == 1:
            count = count + len(y_subset)
    return count/len(y)

train_data = np.genfromtxt("mush_train.data",dtype='str',delimiter=',')
test_data = np.genfromtxt("mush_test.data",dtype='str',delimiter=',')
train_y = train_data[:,0]
train_X = train_data[:, 1:]
test_y = test_data[:, 0]
test_X = test_data[:, 1:]
attribute = [ 'cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type',
'veil-color','ring-number','ring-type','spore-print-color','population','habitat']

tree = build_Tree(train_X,train_y,attribute)
pprint(tree)
y_predict = predict(train_X, tree)
accuracy = np.sum(y_predict == train_y) * 100 / len(y_predict)
print('train set accuracy is ', accuracy)
y_predict = predict(test_X, tree)
accuracy = np.sum(y_predict == test_y) * 100 / len(y_predict)
print('test set accuracy is ', accuracy)
features = train_X.shape[1]
accuracy = np.zeros((features))
#for feature in range(features):
#    accuracy[feature] = one_node_accuracy(train_y, train_X, feature)