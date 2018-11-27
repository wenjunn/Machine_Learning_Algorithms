import numpy as np

def normalization_file(data):
	samples, features=data.shape
	newdata = np.zeros((samples, features))
	mean = np.mean(data,axis=0)
	for i in range(features):
		newdata[:,i] = (data[:,i] - mean[i]) 
	return newdata

def PCA(X):
	W = X.T
	U,eig,VT = np.linalg.svd(W)
	return U

def process(filename):
	data = np.genfromtxt(filename,dtype = "str",delimiter = ",")
	samples, features = data.shape
	Y = data[:,features-1]
	X = data[:,0:features-1]
	x = X.astype(np.float)
	label = []
	for y in Y:
		if y == '2':
			label.append(2)
		else:
			label.append(1)
	return x, label

def split_data(X, Y):
	pos = []
	neg = []
	for i in range(len(Y)):
		if Y[i] == 1:
			pos.append(X[i])
		else:
			neg.append(X[i])
	pos = np.array(pos)
	neg = np.array(neg)
	return pos, neg

def get_gaussian_parameter(data):
	return np.mean(data,axis = 0), np.var(data,axis = 0)

def train_NaiveBayes(X,Y):
	samples, features = X.shape
	pos, neg = split_data(X, Y)
	p0 = (float)(len(neg)) / samples
	p1 = (float)(len(pos)) / samples
	miu0, theta0 = get_gaussian_parameter(neg)
	miu1, theta1 = get_gaussian_parameter(pos)
	return p0, p1, miu0, miu1, theta0, theta1

def predict(p0, p1, miu0, miu1, theta0, theta1, X):
	label = []
	samples, features = X.shape
	for i in range(samples):
		p_neg = 1
		p_pos = 1
		for j in range(features):
			p_pos *= (1.0 / np.sqrt(2 * np.pi * theta1[j])) * np.exp(-0.5 * (X[i][j] - miu1[j])**2 / theta1[j])
			p_neg *= (1.0 / np.sqrt(2 * np.pi * theta0[j])) * np.exp(-0.5 * (X[i][j] - miu0[j])**2 / theta0[j])
		p_pos *= p1
		p_neg *= p0
		if p_pos >= p_neg:
			label.append(1)
		else:
			label.append(2)
	return label

def NaiveBayes(train_X, train_Y, test_X, test_Y):
	p0, p1, miu0, miu1, theta0, theta1 = train_NaiveBayes(train_X, train_Y)
	label = predict(p0, p1, miu0, miu1, theta0, theta1, test_X)
	err = 0.0
	for i in range(len(test_Y)):
		if(label[i] != test_Y[i]):
			err += 1.0
	return 1.0 - err/len(test_Y)
	
def deleteOneEle(array,idx):
	if(idx<array.size and array.size>1):
		
		if(idx==0):
			array_output=array[1:]
			return array_output
		elif(idx==array.size-1):
			array_output=array[:-1]
			return array_output
		else:
			array_output=np.zeros((array.size-1,))
			for i in range(array.size):
				if(i<idx):
					array_output[i]=array[i]
				elif(i==idx):
					continue
				elif(i>idx):
					array_output[i-1]=array[i]
			return array_output

def select_feature(Q,s,k):
	idx_sel=[]
	Qk=Q[:,:k]
	idx=np.array(range(Qk.shape[0]))
	pi=np.sum(Qk**2,axis=1)/k
	while(s>0):
		cdf_pi=np.cumsum(pi)
		cdf_pi[-1]=1.0
		rd=np.random.uniform(0.0,1.0)
		selected_idx=np.searchsorted(cdf_pi,rd,side='left')
		idx_sel.append(idx[selected_idx])
		pi=deleteOneEle(pi,selected_idx)
		pi=pi/sum(pi)
		idx=deleteOneEle(idx,selected_idx)
		s=s-1
	idx_sel=np.array(idx_sel)
	idx_sel = idx_sel.astype(np.int)
	return np.sort(idx_sel)


if __name__ == "__main__":
	train_X, train_Y = process("sonar_train.csv")
	test_X, test_Y = process("sonar_test.csv")
	print("NaiveBayes probability")
	print(NaiveBayes(train_X, train_Y, test_X, test_Y))
	norm_train = normalization_file(train_X)
	PCA_X = PCA(norm_train)
	for k in range(1, 11):
		for s in range(1, 21):
			acc = 0.0
			for i in range(100):
				feature = select_feature(PCA_X, s, k)	
				train_selected = train_X[:,feature]
				test_selected = test_X[:,feature]
				acc += NaiveBayes(train_selected, train_Y, test_selected, test_Y)
			acc /= 100
			print("k ", k, "s", s)
			print(acc)