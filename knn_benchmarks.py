import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors.ball_tree import BallTree
import matplotlib.pyplot as plt
import pandas as pd
import faiss
import time
import sys

n_points = int(sys.argv[1])
d = int(sys.argv[2])
k = 256
batch_size = 512
X = np.random.random(size=(n_points, d)).astype(np.float32)

res = faiss.StandardGpuResources()
flat_config = faiss.GpuIndexFlatConfig()
flat_config.device = 0
index = faiss.GpuIndexFlatL2(res, d, flat_config)
index.add(X)

for bi in range(3,10):
    for ki in range(3, 10):
        t = time.time()
        D, I = index.search(X[0:2**bi,:], 2**ki)
        print 2**bi, 2**ki, int((time.time()-t)*1000)

t = time.time()
cpu_index = BallTree(X)
print("BallTree build time (mins)", int((time.time()-t)/60))

#t = time.time()
#D, I = cpu_index.query(X[0:batch_size,:], k)
#print int((time.time()-t)*1000)

for bi in range(3,10):
    for ki in range(3, 10):
        t = time.time()
        D, I = cpu_index.query(X[0:2**bi,:], 2**ki)
        print 2**bi, 2**ki, int((time.time()-t)*1000)


