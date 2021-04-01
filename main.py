import perceptron as p
import numpy as np

ex = np.array([[0,0], [0,1], [1,0], [1,1]])
res = np.array([0,0,0,1])

p1 = p.perceptron(np.array([1,1]), p.h)
p1.fit(ex, res)
print(p1.poids)

p2 = p.perceptron(np.array([-0.8, -0.8]), p.h)
p2.poids = p1.poids

print(p2.predict(np.array([0,0])))
print(p2.predict(np.array([0,1])))
print(p2.predict(np.array([1,0])))
print(p2.predict(np.array([1,1])))