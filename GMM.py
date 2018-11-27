import numpy as np
import random
from scipy.stats import multivariate_normal
def GMM(X,k,rounds,random_type):
	log = []
	distance = []
	for i in range(rounds):
		mean, cov, lamda = initialize_parameters(X,k,random_type)
		loglikelihood, dist = EM(mean, cov, lamda, X, k)
		log.append(loglikelihood)
		distance.append(dist)
	print(k,np.mean(log),np.var(log))
	if k == 36: 
		print(k,np.mean(distance),np.var(distance)) 

def evaluate_Kmeans(q,X,k):
	label = [np.argmax(p) for p in q]
	centroids = []
	cluster = {}
	for i in range(k):
		cluster[i] = []
	for x, l in zip(X, label):
		cluster[l].append(x)
	for c in cluster:
		if len(cluster[c]):
			centroids.append(np.mean(cluster[c],axis=0))
		else:
			centroids.append(np.random.uniform(-3,3,size = features))
	dist = 0.0
	for x,l in zip(X,label):
		dist += np.linalg.norm(x - centroids[l])**2
	return dist

def EM(mean, cov, lamda, X, k):
	log_likelihood = []
	samples, features = X.shape
	reg_cov = 1e-3*np.identity(features)
	pre = 0.0
	while True:
		#E step
		q = np.zeros((samples, k))
		divisor_sum = np.sum([pi_c * multivariate_normal(mean=mu_c,cov=cov_c).pdf(X) for pi_c,mu_c,cov_c in zip(lamda,mean,cov+reg_cov)],axis=0)
		for m, c, l, j in zip(mean, cov, lamda, range(k)):
			q[:,j] = l * multivariate_normal(mean=m,cov=c).pdf(X) / divisor_sum
		#M step
		mean = []
		cov = []
		lamda = []
		for c in range(k):
			m_c = np.sum(q[:,c],axis=0)
			lamda.append(m_c / np.sum(q))
			mu_c = np.sum(X * q[:,c].reshape(samples,1),axis=0) / m_c
			mean.append(mu_c)
			cov.append(((1/m_c)*np.dot((np.array(q[:,c]).reshape(samples,1)*(X-mu_c)).T,(X-mu_c)))+reg_cov)
		loss = get_log_likelihoods(X,mean,cov,lamda)
		if abs(loss - pre) < 0.1:
			break
		pre = loss
	if k == 36:
		dist = evaluate_Kmeans(q,X,k)
	else:
		dist = 0
	return loss, dist

def process_data(filename):
	data = np.genfromtxt(filename,dtype = "float",delimiter = ",")
	samples, features = data.shape
	X = data[:,1:]
	mean = np.mean(X,axis = 0)
	var = np.std(X, axis=0)
	X = (X - mean)/var
	return X

def get_log_likelihoods(X,mean,cov,lamda):
	reg_cov = 1e-6*np.identity(len(X[0]))
	pmatrix = [l * multivariate_normal(m,c + reg_cov).pdf(X) for l,m,c in zip(lamda,mean,cov)]
	pmatrix = np.array(pmatrix).T
	return np.sum(np.log(np.sum(p)) for p in pmatrix)

def initialize_parameters(X,k,random_type):
	samples, features = X.shape
	if random_type == 0:
		mu = np.random.random((k,features)) * 6 - 3
	else:
		mu = kmeans_plus_centroids(X,k)#kmeans plus random
	cov = []
	for i in range(k):
		cov.append(np.eye(features))#identity matrix
	cov = np.array(cov)
	pi = np.ones(k)/k #pi : fraction per cluster, equally at first
	return mu, cov, pi

def kmeans_plus_centroids(X,k):
	
	samples, features = X.shape
	centroids = np.zeros((k,features))
	center = X[random.randint(0,samples)]
	centroids[0] = center
	distance = [np.linalg.norm(x - center)**2 for x in X]
	newcenter = X[np.random.choice(samples,1,distance)]
	for i in range(k - 1):
		for j in range(samples):
			dist = np.linalg.norm(X[j] - newcenter)**2
			if dist < distance[j]:
				distance[j] = dist
		centroids[i + 1] = np.array(newcenter)
		newcenter = X[np.random.choice(samples,1,distance)]
	return centroids

if __name__ == "__main__":
	X = process_data("leaf.data")
	K = [36]
	#K = [12, 18, 24, 36, 42]
	for k in K:
		GMM(X, k, 20, 0)#random
		GMM(X, k, 20, 1)#kmeans plus random
		