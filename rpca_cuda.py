from pycuda.elementwise import ElementwiseKernel
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import skcuda.linalg
import numpy as np


shrinker = ElementwiseKernel("float *x, float *z, float tau",
                             "z[i] = copysign(max(abs(x[i]) - tau, 0), x[i])") 

squared = ElementwiseKernel("float *x, float *z",
                            "z[i] = abs(pow(x[i], 2))") 

def robust_pca(D):
    """ 
    Parrallel RPCA using ALM, adapted from https://github.com/nwbirnie/rpca.
    Takes and returns numpy arrays
    """
    M = gpuarray.to_gpu(D)
    L = gpuarray.zeros_like(M)
    S = gpuarray.zeros_like(M)    
    Y = gpuarray.zeros_like(M)
    print M.shape

    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5

    while not converged(M, L, S):
        L = svd_shrink(M - S - (mu**-1) * Y, mu)
        S = shrink(M - L + (mu**-1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
    return L.get(), S.get()
    
def svd_shrink(X, tau):
    U, s, V = skcuda.linalg.svd(X)
    return skcuda.linalg.dot(U, skcuda.linalg.dot(skcuda.linalg.diag(shrink(s, tau)), V))
    
def shrink(X, tau):
    Z = gpuarray.empty_like(X)
    shrinker(X, Z, tau)
    return Z
 
def frobeniusNorm(X):
    Z = gpuarray.empty_like(X)
    Z = X**2
    accum = gpuarray.sum(Z).get()
    print accum
    return np.sqrt(accum)

def L1Norm(X):
    return gpuarray.max(X * (gpuarray.zeros((X.shape[1],), dtype=int) + 1)).get()

def converged(M, L, S):
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    print "error =", error
    return error <= 10e-6


if __name__ == "__main__":
    test = np.random.randint(0, 255, (100, 100))
    L, S = robust_pca(test)
    print L
    print S


