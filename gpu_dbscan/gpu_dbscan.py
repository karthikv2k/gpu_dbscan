import numpy as np
import faiss
import time
from sklearn.neighbors.ball_tree import BallTree
import utils
import sys
from sklearn.cluster import DBSCAN


class GpuDbscan():
    def __init__(self, X, index_type="CPU", gpu_device_num=0, gpu_index=None):
        self.X = X
        self.knn_max_k = 0
        self.knn_search_cnt = 0
        self.gpu_index = gpu_index#self.get_gpu_index(X)
        self.cpu_index = self.get_ball_tree_index(X)

    @staticmethod
    def get_ball_tree_index(X):
        return BallTree(X)

    @staticmethod
    def get_gpu_index(X, gpu_device_num=0):
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = gpu_device_num
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
        index.add(X)
        return index

    def knn_search_single_instance_gpu(self, point, eps, start_k=512, end_k=512):
        D, I = self.gpu_index.search(np.array([point]), start_k)
        D = D[0]
        I = I[0]
        I = I[D <= eps]
        if len(I) >= start_k:
            if start_k <= end_k:
                #return self.knn_search_single_instance_cpu(point, eps)
                1
            else:
                return self.knn_search_single_instance_gpu(point, eps, start_k=2 * start_k)
        self.knn_max_k = max(self.knn_max_k, start_k)
        self.knn_search_cnt += 1
        return I

    def knn_search_single_instance_cpu(self, point, eps):
        ind_list = self.cpu_index.query_radius(np.array([point]), eps)
        return ind_list[0]

    def gpu_dbscan(self, eps, min_samples):
        self.knn_search_cnt = 0
        self.knn_max_k = 0

        eps = eps ** 2
        UNCLASSIFIED = -2
        NOISE = -1
        n_points = self.X.shape[0]
        cluster_ids = np.zeros(n_points, dtype=np.int32)
        cluster_ids[:] = UNCLASSIFIED
        cluster_id = -1
        visited_flag = np.zeros(n_points, dtype=np.bool8)
        core_point_flag = np.zeros(n_points, dtype=np.bool8)
        loop_cnt = 0
        for i, xi in enumerate(self.X):
            if not visited_flag[i]:
                visited_flag[i] = True
                neighbor_pts = self.knn_search_single_instance_gpu(xi, eps=eps).tolist()
                if len(neighbor_pts) < min_samples:
                    cluster_ids[i] = NOISE
                else:
                    cluster_id += 1
                    cluster_ids[i] = cluster_id
                    core_point_flag[i] = True
                    neighbor_pts_tmp = []
                    for pt in neighbor_pts:
                        if visited_flag[pt]:
                            if cluster_ids[pt] < 0:
                                cluster_ids[point_idx] = cluster_id
                        else:
                            neighbor_pts_tmp.append(pt)
                    neighbor_pts = set(neighbor_pts_tmp)
                    while len(neighbor_pts) > 0:
                        loop_cnt += 1
                        point_idx = neighbor_pts.pop()
                        if not visited_flag[point_idx]:
                            visited_flag[point_idx] = True
                            neighbor_pts_p = self.knn_search_single_instance_gpu(self.X[point_idx, :], eps=eps)
                            if len(neighbor_pts_p) >= min_samples:
                                core_point_flag[point_idx] = True
                                cluster_ids[point_idx] = cluster_id
                                for n_point in neighbor_pts_p:
                                    if visited_flag[n_point]:
                                        if cluster_ids[point_idx] < 0:
                                            cluster_ids[n_point] = cluster_id
                                    else:
                                        neighbor_pts.add(n_point)
                        else:
                            print("already visited node", point_idx)
                        if loop_cnt % int(n_points / 10) == 0:
                            print("loop cnt", loop_cnt, i, self.knn_search_cnt, visited_flag.sum())
        return cluster_ids, core_point_flag, visited_flag


if __name__ == '__main__':
    n_points = int(sys.argv[1])
    d = int(sys.argv[2])
    mode = sys.argv[3]
    X = None
    if mode == 'random':
        X = np.random.random((n_points, d)).astype(np.float32)
    elif mode == 'blob':
        X = utils.get_test_blobs(n_points, d)
    else:
	print 'Unrecognized mode'
	sys.exit(1)

    t = time.time()
    DBSCAN(eps=1, min_samples=10, metric='l2').fit(X)
    print int((time.time() - t) * 1000)

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    index.add(X)
    gd = GpuDbscan(X, gpu_index=index)
    cluster_ids, core_point_flag, visited_flag = gd.gpu_dbscan(1, 10)
    print(np.unique(cluster_ids))
