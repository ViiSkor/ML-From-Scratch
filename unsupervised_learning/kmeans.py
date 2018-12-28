import numpy as np
from preprocessing import hypercube_normalization

class KMeans(object):    
    """ A clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to 
    the center of the new formed clusters.
    
    Attributes:
        epsilon: float
            The number that is added to the divisor to prevent division by zero.
        max_iterations: int, optional
            The number of iterations the algorithm will run for if it does
            not converge before that. 
        n_clusters: int, optional
            The number of clusters the algorithm will form.
        init_mode: str
            The mode of centroids initialization.
        centroids: numpy.array, None at class object initialization
            An array stores the coordinates of the centroids.
        clusters: numpy.array, None at class object initialization
            An array stores dataset samples labels.
    """
    
    def __init__(self, max_iter=300, n_clusters=2, init_mode="random_sample"):
        self.epsilon = 1e-8
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.init_mode = init_mode
        self.centroids = None
        self.clusters = None
       
    def __init_centroids(self, data, n, mode):
        """ Initialize the centroids.
        Has two mode: take n random samples of data as the centroids and 
        random sharing.
        
        Read about random sharing:
        https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
        
        Args:
            data: numpy array
                2d dataset array, where a horizontal axis is equal to the 
                number of features, a vertical axis is equal to the number of 
                dataset samples.
            n: int
                The number of clusters the algorithm will form.
                
        Returns:
            centroids: numpy 0d array
                An array stores the init coordinates of the centroids.
        """
        
        if mode == "random_sample":
            index = np.random.choice(data.shape[0], n, replace=False)
            return data[index]
        elif mode == "sharding_init":
            attr_sum = np.sort(np.sum(data, axis=1))
            clusters_idx = np.array_split([*range(len(attr_sum))], n)
            centroids = []
            for idx in clusters_idx:
                centroids.append(np.mean(data[idx], axis=0))      
            return np.array(centroids)
        else:
            raise Exception('No such init!')
            
    def __compute_distance(self, x, centroid):
        """ Finds the Euclidean distance between a sample and a centroid.
        
        Args:
            x: numpy.array
                An array of the coordinates of a dataset sample.
            centroid: numpy.array
                An array of the coordinates of one of k centroids.
                
        Returns:
            numpy.float: Euclidean distance.
        """
        
        diff = x - centroid
        return np.sqrt(np.dot(diff.T, diff))
        
    def __find_nearest_centroids(self, data):
        """ Finds the closest centroid to each dataset sample.
        
        Args:
            data: numpy.array
                2d array, where a horizontal axis is equal to the number of 
                features, a vertical axis is equal to the number of dataset 
                samples.
        """
        
        self.clusters = np.array([])                
        for i, d in enumerate(data):
            min_dist = np.inf
            self.clusters = np.concatenate((self.clusters, np.array([-1])))
            for j, c in enumerate(self.centroids):
                dist = self.__compute_distance(d, c)
                if min_dist > dist:
                    min_dist = dist
                    self.clusters[i] = j
        
    def __move_centroids(self, data):
        """ Moves the centroids to the center of the newly formed clusters.
        
        Args:
            data: numpy.array
                2d array, where a horizontal axis is equal to the number of 
                features, a vertical axis is equal to the number of dataset 
                samples.
        """
        
        for i in range(len(self.centroids)):
            members_cluster = data[self.clusters == i]
            self.centroids[i] = np.sum(members_cluster, axis=0) / (len(members_cluster) + self.epsilon)
                       
    def fit(self, data, stop=1e-8):
        """ Does K-Means clustering.
        
        Args:
            data: numpy.array
                2d dataset array, where a horizontal axis is equal to the 
                number of features, a vertical axis is equal to the number of 
                dataset samples.
            stop: float
                The maximum allowable difference between centroids and 
                prev_centroids which indicates to stop searching for the 
                cluster structure.
                
        Returns:
            centroids: numpy 0d array
                An array stores the coordinates of the centroids.
        """
        
        iteration = 0
        data = hypercube_normalization(data)
        self.centroids = self.__init_centroids(data, self.n_clusters, self.init_mode)
        prev_centroids =  np.empty_like(self.centroids)
    
        
        while np.any(np.power(self.centroids - prev_centroids, 2) > stop) and self.max_iter > iteration:
            iteration += 1
            self.__find_nearest_centroids(data)
            prev_centroids = self.centroids.copy()
            self.__move_centroids(data)
        
        return self.centroids
    
    def predict(self):
        """ Predicts clusters.
       
        Returns:
            clusters: numpy.array
                The cluster label corresponding to each sample.
        """
        
        return self.clusters