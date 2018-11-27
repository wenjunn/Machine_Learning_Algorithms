import numpy as np
def process_data(filename):
	data = np.genfromtxt(filename,dtype = "float",delimiter = ",")
	samples, features = data.shape
	X = data[:,1:]
	mean = np.mean(X,axis = 0)
	var = np.std(X, axis=0)
	X = (X - mean)/var
	return X
	
def get_random_centroids(k, attributes):
	centroids = {}
	for i in range(k):
		centroids[i] = np.random.uniform(-3,3,size = attributes)
	return centroids

def get_cluster(k):
	cluster = {}
	for i in range(k):
		cluster[i] = []
	return cluster


def get_distance(X, centroids, k):
	samples, features = X.shape
	label = []
	for i in range(300):
		cluster = get_cluster(k)
		for x in X:
			distances = [np.linalg.norm(x - centroids[centroid]) for centroid in centroids]
			classification = distances.index(min(distances))
			cluster[classification].append(x)
		prev_centroids = dict(centroids)
		for classification in cluster:
			if len(cluster[classification]):
				centroids[classification] = np.mean(cluster[classification],axis=0)
			else:
				centroids[classification] = np.random.uniform(-3,3,size = features)
		optimized = True
		for c in centroids:
			prev_c = prev_centroids[c]
			current_c = centroids[c]
			if np.sum((current_c - prev_c)/prev_c * 100.0) > 0.1:
				for x in X:
					distances = [np.linalg.norm(x - centroids[centroid]) for centroid in centroids]
					label.append(distances.index(min(distances)))
				optimized = False
		if optimized:
			break
	dist = 0.0
	for classification in cluster:
		dist += np.sum([np.linalg.norm(x - centroids[classification])** 2 for x in cluster[classification]])
	return dist, label

def kmeans(X, K, rounds):
	samples, attributes = X.shape
	for k in K:
		print(k)
		dist = []
		for i in range(rounds):
			centroids = get_random_centroids(k, attributes)
			#centroids = kmeans_plus_centroids(k, X)
			d, label = get_distance(X, centroids, k)
			dist.append(d)
		print(np.mean(dist),np.var(dist))
		
def kmeans_plus_centroids(k, X):
	centroids = {}
	samples, features = X.shape
	centroids[0] = X[np.random.randint(samples, size=1)]
	distance = [np.linalg.norm(x - centroids[0])**2 for x in X]
	newcenter = X[np.random.choice(samples,1,distance)]
	for i in range(k - 1):
		for j in range(samples):
			dist = np.linalg.norm(X[j] - newcenter)**2
			if dist < distance[j]:
				distance[j] = dist
		centroids[i + 1] = newcenter
		newcenter = X[np.random.choice(samples,1,distance)]
	return centroids

if __name__ == "__main__":
	X = process_data("leaf.data")
	#K = [12, 18, 24, 36, 42]
	K = [42]
	kmeans(X, K, 20)


