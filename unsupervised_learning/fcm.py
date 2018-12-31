import numpy as np
from utils.preprocessing import hypercube_normalization
from utils.clusterization import init_centroids

class FCM(object):   
    """ 
    Fuzzy c-means is a method of clustering which allows one piece of data 
    to belong to two or more clusters with some degree of membership.
    The implemented algorithm is a modification of the С-means - 
    the Gustafson-Kessel algorithm. The Gustafson-Kessel clustering algorithm 
    associates each cluster with both a point and a matrix, respectively 
    representing the cluster centre and its covariance. 
        
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
        membership: numpy.array, None at class object initialization
            An array stores the degree of belonging of each sample to each 
            cluster.
    """
    
    def __init__(self, max_iter=300, n_clusters=2, init_mode="random_sample"):
        self.epsilon = 1e-8
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.init_mode = init_mode
        self.centroids = None
        self.clusters = None
        self.membership = None
    
    def __mahalanobis(self, data, cov_matrix):
        """ Finds the Mahalanobis distance between a sample and a centroid.
        
        Args:
            data: numpy.array
                An array of the coordinates of a dataset sample.
            cov_matrix: numpy.array
                An array of the covariance matrices of the clusters.
                
        Returns:
            numpy.array: Mahalanobis distance.
        """
        
        distances = []
        for i in range(self.centroids.shape[0]):
            dist = []
            diff = data - self.centroids[i]
            for sample in diff:
                dist.append(np.dot(np.dot(sample, cov_matrix[i]), sample.T))
            distances.append(dist)
        return np.array(distances)
    
    def __compute_membership(self, distances):
        """ Calculates memberships - to what extent each sample belongs to each 
        cluster.
        
        Args:
            distances: numpy.array
                An array of distances between samples and centroids.
                
        Returns:
            numpy.array: the membership array of each sample to each cluster.
        """
        
        membership = []
        for i in range(self.centroids.shape[0]):
            partition = []
            for dist in distances.T:
                dist += self.epsilon
                partition.append(np.power(dist[i], -2) / np.sum(np.power(dist, -2)))
            membership.append(partition)
        return np.array(membership)  
    
    def __compute_centroids(self, data):
        """ Computes the coordinate of the centroids.
        
        Args:
            data: numpy.array
                2d array, where a horizontal axis is equal to the number of 
                features, a vertical axis is equal to the number of dataset 
                samples.
                
        Returns:
            centroids: numpy 0d array
                An array stores the coordinates of the centroids.
        """
        
        centroids = []
        for i in range(self.n_clusters):
            c = []
            for x in data.T:
                members = np.power(self.membership[i], 2) 
                c.append(np.dot(members, x) / np.sum(members))
            centroids.append(c)
        return np.array(centroids)
    
    def __compute_cov(self, data):
        """ Computes the covariance matrix of each cluster.
        
        Args:
            data: numpy.array
                2d array, where a horizontal axis is equal to the number of 
                features, a vertical axis is equal to the number of dataset 
                samples.
                
        Returns:
            cov_matrix: numpy.array
                An array of the covariance matrices that has 
                a dimension [n_clusters, n_features, n_features], where 
                n_features - the number of dataset features
        """
        
        cov_matrix = []
        members = np.power(self.membership, 2) + self.epsilon
        for i in range(self.centroids.shape[0]):
            cov = np.zeros((self.centroids.shape[1], self.centroids.shape[1]))
            for j, x in enumerate(data):
                diff = (x - self.centroids[i]).reshape(1, self.centroids.shape[1])
                cov += members[i, j] * np.dot(diff.T, diff)
            cov_matrix.append(cov / np.sum(members[i]))
        
        return np.array(cov_matrix)
    
    def __compute_scaling_matrix(self, data, cov_matrix):
        """ Calculate a new scaling matrix for each centroid.
       
        Returns:
            scaling_matrix: numpy.array
                An array of the scaling matrices that has 
                a dimension [n_clusters, n_features, n_features], where 
                n_features - the number of dataset features
        """
        
        scaling_matrix = []
        for i in range(cov_matrix.shape[0]):
            root_det = np.power(np.linalg.det(cov_matrix[i]), 1/data.shape[1])
            inv = np.linalg.inv(cov_matrix[i])
            scaling_matrix.append(np.dot(root_det, inv))
        return np.array(scaling_matrix)
    
    def __extract_clusters(self):
        """ Assigns cluster labels to each sample with respect to the highest 
        degree of membership.
       
        Returns:
            clusters: numpy.array
                The cluster label corresponding to each sample.
        """
        
        clusters = []
        for member in self.membership.T:
            clusters.append(member.argmax())
        return np.array(clusters)
    
    def fit(self, data, stop=1e-5):
        """ Does the Gustafson-Kessel clustering algorithm.
        
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
        
        self.centroids = init_centroids(data, self.n_clusters, self.init_mode)
        prev_centroids = np.empty_like(self.centroids)
        # Scaling matrix initialized by unit matrices
        scaling_matrix = np.array([np.eye(data.shape[1]) for _ in range(self.n_clusters)])
        distances = self.__mahalanobis(data, scaling_matrix)
        self.membership = self.__compute_membership(distances)
        self.clusters = self.__extract_clusters()
    
        while np.any(np.power(self.centroids - prev_centroids, 2) > stop) and self.max_iter > iteration:
            iteration += 1
            prev_centroids = self.centroids.copy()
            self.centroids = self.__compute_centroids(data)
            cov_matrix = self.__compute_cov(data)
            scaling_matrix = self.__compute_scaling_matrix(data, cov_matrix)
            distances = self.__mahalanobis(data, scaling_matrix)
            self.membership = self.__compute_membership(distances)
            self.clusters = self.__extract_clusters()
                
        return self.centroids
    
    def predict(self):
        """ Predicts clusters.
       
        Returns:
            clusters: numpy.array
                The cluster label corresponding to each sample.
            membership: numpy.array
                An array stores the degree of belonging of each sample to each 
                cluster.
        """
        
        return self.clusters, self.membership