import numpy as np

a = np.array([[1, 2 ,3], [4, 5, 6]])
print(a.shape)
print(np.sum(a,axis=0,keepdims=True))

pre = np.array([[0.1,0.3,0.6],[0.2,0.6,0.2],[0.9,0.1,0],[0.3,0.4,0.3]])
lab = np.array([[0,1,0],[1,0,0],[1,0,0],[1,0,0]])
