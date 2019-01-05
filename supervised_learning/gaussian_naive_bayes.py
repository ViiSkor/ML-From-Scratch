import numpy as np
from utils.preprocessing import compute_average, compute_variance

class GaussianNaiveBayes(object):
    """ A Bayes classifier is a simple probabilistic classifier, which is based 
    on applying Bayes' theorem. The feature model used by a naive Bayes 
    classifier makes strong independence assumptions. This means that the 
    existence of a particular feature of a class is independent or unrelated to 
    the existence of every other feature.
    
    Attributes:
        epsilon: float
            The number that is added to the divisor to prevent division by zero.
        labels: numpy.array
            The array of the labels of the classes.
        priors: numpy.array (n_classes, 1)
            The array of the probability of each class, where n_classes is the 
            number of classes.
        mean: numpy.array (n_classes, n_features)
            The array of the mean of each feature for each class, where 
            n_classes is the  number of classes and n_features is the number 
            of features.
        variance: numpy.array (n_classes, n_features)
            The array of the variance of each feature for each class, where 
            n_classes is the number of classes and n_features is the number 
            of features.
    """
    
    def __init__(self):
        self.epsilon = 1e-8
        self.n_classes = None
        self.labels = None
        self.priors = None
        self.mean = None
        self.variance = None
    
    def __extract_labels(self, Y):
        """ Extracts the labels of the classes.
        
        Args:
            Y: numpy.array (n_samples)
                The target array, where n_samples is the number of samples.
        """
        self.labels = np.unique(Y)
        self.n_classes = self.labels.shape[0]
        
    def __prior(self, X, Y):
        """ Prior P(Y) is the probability of hypothesis Y being true 
        (regardless of the data).
        
        Args:
            X: numpy.array (n_samples, n_features)
                Training vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            Y: numpy.array (n_samples)
                Target array.
        """
        
        self.priors = np.zeros(self.n_classes).reshape(self.n_classes, 1)
        for l in self.labels:
            self.priors[l] = len(X[Y == l]) / X.shape[0]
    
    def __gaussian(self, X, Y):
        """ Learns Gaussian components of the likelihood P(X|Y).
        
        Args:
            X: numpy.array (n_samples, n_features)
                Training vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            Y: numpy.array (n_samples)
                Target array.
        """
        
        mean =  []
        variance = []
        for l in np.nditer(self.labels):
            mean.append(compute_average(X[Y == l]))
            variance.append(compute_variance(X[Y == l]))
            
        self.mean = np.array(mean)
        self.variance = np.array(variance)
    
    def __likelihood(self, X, label):
        """ Likelihood method for one class P(datapoints|Y=class)
        
        Args:
            X: numpy.array (n_samples, n_features)
                Training vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            label: int|str
                Label of a class.
        """
        
        pdf = np.zeros(X.size).reshape(X.shape[0], X.shape[1])
        for i in range(X.shape[0]):
            pdf[i] = np.power(np.e, -(np.power(X[i] - self.mean[label], 2)) 
                            / (2 * self.variance[label] +self.epsilon))
            pdf[i] /= np.sqrt(2*np.pi*self.variance[label])
            
        return np.prod(pdf, axis=1)
    
    def __posterior(self, X, label):
        """ Compute posterior P(Y|datapoints) for one class.
        
        Args:
            X: numpy.array (n_samples, n_features)
                Training vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            label: int|str
                Label of a class.
                
        Returns:
            posteriors: numpy.array
                The array of the posterior probability of the class.
        """
        
        posteriors = self.__likelihood(X, label) * self.priors[label]
        return posteriors
    
    def fit(self, X, Y):
        """ Fit Gaussian Naive Bayes according to X, y.
        
        Args:
            X: numpy.array (n_samples, n_features)
                Training vectors, where n_samples is the number of samples and 
                n_features is the number of features.
            Y: numpy.array (n_samples)
                Target array.
        """
        
        self.__extract_labels(Y)
        self.__prior(X, Y)
        self.__gaussian(X, Y)
    
    def predict(self, X):
        """ Perform classification.
       
        Args:
            X: numpy.array (n_samples, n_features)
                Test vectors, where n_samples is the number of samples and 
                n_features is the number of features.
        
        Returns:
            class: numpy.array
                Predicted class for each data point.
        """
        
        posteriors = np.zeros(X.shape[0] * self.n_classes).reshape(X.shape[0], self.n_classes)
        for l in self.labels:
            posteriors[:, l] = self.__posterior(X, l)
        return np.argmax(posteriors, axis=1)