import numpy as np

class LinearRegression(object):
    """ Linear model.
        
        Attributes:
            coefficients: numpy.array, None at class object initialization
                An array stores the coefficients of the linear function.
                Where element with index 0 is independent coefficient.
    """
    def __init__(self):
        self.coefficients = None  
    
    def fit(self, X, Y):
        """ Fitting a linear equation to observed data.
        
        Args:
            X: numpy.array
                2d dataset array, where a horizontal axis is equal to the 
                number of features plus column with 1 for the independent 
                coefficient, a vertical axis is equal to the number of 
                dataset samples.
            Y: numpy.array
                1d array of target values.
        """
        
        p1, p2 = 0, 0
        for x, y in zip(X, Y):
            x = x.reshape(X.shape[1], 1)
            y = y.reshape(1, 1)
            p1 += np.dot(x, x.T)
            p2 += x * y
        
        self.coefficients = np.dot(np.linalg.inv(p1), p2)
        
    def predict(self, X):
        """ Predict using the linear model
         
        Args:
            X: numpy.array
                2d dataset array, where a horizontal axis is equal to the 
                number of features plus column with 1 for the independent 
                coefficient, a vertical axis is equal to the number of 
                dataset samples.
        
        Returns:
            numpy.array, shape (n_samples, 1)
                Predicted values.
        """
        
        return np.dot(X, self.coefficients)