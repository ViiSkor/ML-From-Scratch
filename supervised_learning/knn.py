import numpy as np
from utils.similarity import euclidean

class KNN(object):
    """ k-Nearest Neighbors algorithm. The principle behind KNN is to
    find a predefined number of training samples nearest to the new data point 
    and classified it based on this training samples.
    
    Attributes:
        k: int
            Number of neighbours.
        data: numpy.array (n_samples, n_features)
            Training vectors, where n_samples is the  number of samples and 
            n_features is the number.
            of features.
        Y: numpy.array (n_samples, 1)
            Target array of the training vectors, where n_samples is the 
            number of samples.
    """
    
    def __init__(self):
        self.k = 1
        self.data = None
        self.Y = None

    def __compute_weight(self, neighbours):
        """ Weighted points by the inverse of their distance.
       
        Args:
            neighbours: list (k_neighbours, (1, 2))
                List of the tuples of the neighbours, where k_neighbours is the number of the 
                nearest neighbours
        
        Returns:
            weights: numpy.array
                Weights of each neighbour.
        """
        
        weights = []
        for dist, i in neighbours:
            weights.append(np.power(dist, -2))
        return np.array(weights)
    
    def __uniform(self, X):
        """ Classification data points. Weight points by the inverse of their 
        distance. Closer neighbors of a query point will have a greater
        influence than neighbors which are further away.
       
        Args:
            X: numpy.array (n_samples, n_features)
                Test vectors, where n_samples is the number of samples and 
                n_features is the number of features.
        
        Returns:
            prediction: numpy.array
                Predicted class for each data point.
        """
        
        prediction = []
        for x in X:
            neighbours = self.__KNN(x)
            idx = [n[1] for n in neighbours]
            counts = np.bincount(self.Y[idx])
            prediction.append(np.argmax(counts))
        return prediction
    
    def __distance_weighted(self, X):
        """ Classification data points. All points in each neighborhood are 
        weighted equally.
       
        Args:
            X: numpy.array (n_samples, n_features)
                Test vectors, where n_samples is the number of samples and 
                n_features is the number of features.
        
        Returns:
            prediction: numpy.array
                Predicted class for each data point.
        """
        
        prediction = []
        for x in X:
            neighbours = self.__KNN(x)
            weights = self.__compute_weight(neighbours)
            distances = np.array([n[0] for n in neighbours])
            weights *= distances
            votes = {}
            for n, w in zip(neighbours, weights):
                votes[self.Y[n[1]]] = votes.get(self.Y[n[1]], 0) + n[0] * w
            prediction.append(max(votes, key=votes.get))
        return prediction
        
    def __KNN(self, x):
        """ Finds the K-neighbors of a datapoint.
       
        Args:
            x: numpy.array (1, n_features)
                Vector of the features of the sample, where n_samples is the 
                number of samples and n_features is the number of features.
                
        Returns:
            distances: numpy.array
                Distance from datapoint to K nearest neighbors.
        """
        
        distances = []
        for i, d in enumerate(self.data):
            distances.append((euclidean(x, d), i))
        
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def fit(self, X, Y, k):
        """ Fit the dataset for future classification.
       
        Args:
            X: numpy.array (n_samples, n_features)
                Test vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            Y: numpy.array (n_samples)
                Target array.
            k: int
                Number of neighbours.
        """
        
        self.data = X
        self.Y = Y
        self.k = k
    
    def predict(self, X, mode="uniform"):
        """ Perform classification.
       
        Args:
            X: numpy.array (n_samples, n_features)
                Test vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            mode: str
                Mode of the weight function used in prediction. Possible values:
                "uniform" : uniform weights. All points in each neighborhood 
                are weighted equally.
                "weighted" : weight points by the inverse of their distance. 
                Closer neighbors of a query point will have a greater influence 
                than neighbors which are further away.
        
        Returns:
            prediction: numpy.array
                Predicted class for each data point.
        """
        
        if mode == "uniform":
            prediction = self.__uniform(X)
        elif mode == "weighted":
            prediction = self.__distance_weighted(X)
        else:
            raise Exception('No such distance mode!')
        return prediction