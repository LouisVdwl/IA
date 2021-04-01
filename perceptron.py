import numpy as np
class perceptron:
    # Initialisation du perceptron
    def __init__(self, poids, f):
        self.poids = np.append(poids,1)
        self.f = f
        self.alpha = 0.001
    
    def predict(self,x):
        x = np.append(x,1)
        y = np.dot(self.poids,x)
        y = self.f(y)
        return y

    def fit(self, X,Y):
        erreur = 0
        for i in range(10):
            for i, x in enumerate(X):
                y = self.predict(x)
                print (x,">",y, " <- prédiction | ", Y[i])
                erreur = Y[i] - y
                self.poids = self.poids - (erreur*self.alpha * np.append(x,1))
                print("Poids: ", self.poids)

def h(x):
    return 1 if x >= 0 else 0

    
    

