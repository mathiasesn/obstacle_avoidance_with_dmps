"""Iterative Closests Point
"""


import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    """Calculates the least-squares best-fit transform that maps corresponding
        points A to B in m spatial dimensions.
    
    Arguments:
        A {np.array} -- (n x m) array of corresponding points
        B {np.array} -- (n x m) array of corresponding points
    
    Returns:
        np.array -- ((m+1) x (m+1)) homogeneous transformation matrix that maps
            A on to B
        np.array -- (m x m) rotation matrix
        np.array -- (m x 1) translation vector
    """
    assert A.shape == B.shape, f'inputs does not have same shape, got {A.shape} and {B.shape}'
    # get number of dimensions
    dims = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[dims-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    # homogeneours transformation
    T = np.identity(dims+1)
    T[:dims, :dims] = R
    T[:dims, dims] = t
    return T, R, t


def nearest_neighbor(src, dst):
    """Nearest neighbor: finds the nearest (euclidean) neighbor in dst for each
        point in src.
    
    Arguments:
        src {np.array} -- (n x m) array of points
        dst {np.array} -- (n x m) array of points
    
    Returns:
        np.array -- euclidean distances of the nearest neighbor
        np.array -- dst indices of the nearest neighbor
    """
    assert src.shape == dst.shape, f'inputs does not have same shape, got {src.shape} and {dst.shape}'
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    dists, inds = neigh.kneighbors(src, return_distance=True)
    return dists.ravel(), inds.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """Iteractive Closest point: finds best-fit transform that maps points A
        on to points B.
    
    Arguments:
        A {np.array} -- (n x m) array of source points
        B {np.array} -- (n x m) array of destination points
    
    Keyword Arguments:
        init_pose {np.array} -- ((m+1) x (m+1)) homogenous transformation (default: {None})
        max_iterations {int} -- terminate after max_iterations (default: {20})
        tolerance {float} -- convergence criteria (default: {0.001})
    
    Returns:
        np.array -- final transformation that maps A on to B
        np.array -- euclidean distances (error) of the nearest neighbor
        int -- number of iterations to converge
    """
    assert A.shape == B.shape, f'inputs does not have same shape, got {A.shape} and {B.shape}'
    # get number of dimensions
    dims = A.shape[1]
    # make points homogeneous and copy them to maintain the originals
    src = np.ones((dims+1, A.shape[0]))
    dst = np.ones((dims+1, B.shape[0]))
    src[:dims, :] = np.copy(A.T)
    dst[:dims, :] = np.copy(B.T)
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        dists, inds = nearest_neighbor(src[:dims, :].T, dst[:dims, :].T)
        # compute the transformation between the current source and destination points
        T, _, _ = best_fit_transform(src[:dims, :].T, dst[:dims, inds].T)
        # update the current source
        src = np.dot(T, src)
        # check error
        mean_error = np.mean(dists)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:dims, :].T)
    return T, dists, i