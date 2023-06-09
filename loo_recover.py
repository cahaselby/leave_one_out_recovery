
"""
12 MAY 2023

This is the module needed to run leave one out recovery

author: Cullen Haselby 
"""
#########################
# IMPORTS
#########################


import numpy as np
from scipy import fftpack
import tensorly as tl 
import time
from tensorly.decomposition import tucker, parafac
from scipy.linalg import subspace_angles,dft
import itertools
import timeit
from scipy.optimize import curve_fit
from scipy.linalg import khatri_rao,hilbert,null_space,subspace_angles, hadamard, dft
#import skvideo.io
from sklearn.cluster import KMeans


#Helper function for returning a gaussian JL measurement matrix
def gauss_meas(dim, k):
    #np.random.seed(0)
    return np.sqrt(1/(k))*np.random.normal(0.0, 1.0, [k, dim])

def SHRT_meas(dim, k):
    D = np.diag(np.random.choice([1,-1], dim))
    H = np.sqrt(1/k)*hadamard(dim)
    idx = np.zeros(dim,dtype=bool)
    idx[:k] = True
    np.random.shuffle(idx)
    S = np.eye(dim)[idx]
    
    return S@H@D

def RFD_meas(dim, k):
    D = np.diag(np.random.choice([1,-1], dim))
    F = np.sqrt(dim/k)*dft(dim)
    idx = np.zeros(dim,dtype=bool)
    idx[:k] = True
    np.random.shuffle(idx)
    S = np.eye(dim)[idx]
    
    return S@F@D

def generate_super_diagonal_tensor(diagonal_elems, dim):
    '''
    Generate super diagonal tensor of dimension = dim
    '''
    n = len(diagonal_elems)
    tensor = np.zeros(np.repeat(n, dim))
    for i in range(n):
        index = tuple([i for _ in range(dim)])
        tensor[index] = diagonal_elems[i]
    return tl.tensor(tensor)

def my_random_matrix_generator(n,m,  std=1, typ='g', random_seed=1, sparse_factor=1):
    '''
    Generate random matrix of size m x n
    :param m: length
    :param n: width
    :param Rinfo_bucket: parameters fo generating the random matrix: std (standard devidation for each entry); typ
    (u: uniform; g: Gaussian; sp: sparsity=sparse_factor; sp0: sparsity=2/3;sp1:sparsity=1-1/sqrt(n) )
    :return: random matrix
    '''

    #np.random.seed(random_seed)
    types = set(['r','g', 'u', 'sp', 'sp0', 'sp1','rfd','real_rfd', "shrt"])
    assert typ in types, "please aset your type of random variable correctly"

    if typ == 'g':
        return np.random.normal(0, 1, size=(m, n)) * std
        
    elif typ =='rfd':
        D = np.diag(np.random.choice([1,-1], n))
        F = np.sqrt(n/m)*dft(n)
        idx = np.zeros(n,dtype=bool)
        idx[:m] = True
        np.random.shuffle(idx)
        S = np.eye(n)[idx]
        return S@F@D
    
    elif typ =='real_rfd':
        D = np.diag(np.random.choice([1,-1], n))
        F = np.sqrt(n/m)*dft(n)
        idx = np.zeros(n,dtype=bool)
        idx[:m] = True
        np.random.shuffle(idx)
        S = np.eye(n)[idx]
        return np.real(S@F@D)
    
    elif typ == "shrt":
        D = np.diag(np.random.choice([1,-1], n))
        H = np.sqrt(1/m)*hadamard(n)
        idx = np.zeros(n,dtype=bool)
        idx[:m] = True
        np.random.shuffle(idx)
        S = np.eye(n)[idx]
        return S@H@D

    elif typ == 'u':
        return np.random.uniform(low=-1, high=1, size=(m, n)) * np.sqrt(3) * std
    elif typ == 'sp':
        return np.random.choice([-1, 0, 1], size=(m, n), p=[sparse_factor / 2, \
                                                            1 - sparse_factor, sparse_factor / 2]) * np.sqrt(
            1 / sparse_factor) * std
    elif typ == 'r':
        return std*np.diag(np.random.choice([-1,1], size=n))[:m,:]
    elif typ == 'sp0':
        return np.random.choice([-1, 0, 1], size=(m, n), p=[1 / 6, 2 / 3, 1 / 6]) * np.sqrt(3) * std
    elif typ == 'sp1':
        return np.random.choice([-1, 0, 1], size=(m, n), p= \
            [1 / (2 * np.sqrt(n)), 1 - 1 / np.sqrt(n), 1 / (2 * np.sqrt(n))]) * np.sqrt(np.sqrt(n)) * std
    
def eval_rerr(X, X_hat, X0=None):
    """
    :param X: tensor, X0 or X0+noise
    :param X_hat: output for apporoximation
    :param X0: true signal, tensor
    :return: the relative error = ||X- X_hat||_F/ ||X_0||_F
    """
    if X0 is not None:
        error = X0 - X_hat
        return tl.norm(error) / tl.norm(X0)
        #return np.linalg.norm(error.reshape(np.size(error), 1), 'fro') / \
           #np.linalg.norm(X0.reshape(np.size(X0), 1), 'fro')
    error = X - X_hat
    return tl.norm(error) / tl.norm(X)
            #np.linalg.norm(error.reshape(np.size(error), 1), 'fro') / \
           #np.linalg.norm(X0.reshape(np.size(X), 1), 'fro')

def add_noise(T,relnoise):

    Noise = np.random.normal(size=T.shape)
    #Normalize 
    Noise = Noise / tl.norm(Noise)
    #Scale relative to norm of T
    Noise = (relnoise *tl.norm(T) )* Noise

    return T +  Noise

def average_results(results,headers,groupfield,fieldy):


    df = pd.DataFrame(results,columns = headers)


    agg = df.groupby([groupfield])[fieldy].agg(['mean', 'sem', 'median'])
    agg['ci95_hi'] = agg['mean'] + 1.96* agg['sem']
    agg['ci95_lo'] = agg['mean'] - 1.96* agg['sem']

    return agg

def measurement_ensemble(N,d,S,R,meas_func,typ='g',std=1):

    A = {}
    A[N] = []  
    for m in range(N):
        A[m] = []
        for n in range(N):
            if n==m:
                #A[m].append(meas_func(d,d))
                A[m].append(np.eye(d[n]))
            else:
                A[m].append(meas_func(d[n],R[n],typ=typ))
            
  
        A[N].append(meas_func(d[m],S[m],typ=typ))
    
    return A


def measure_tensor(T,A,mode='kron'):
    N = len(A.keys()) - 1

    B = {}
    
    if mode=='kron':
        for m in range(N):
            B[m] = tl.tenalg.multi_mode_dot(T,A[m],skip=m)

        #B[N] = tl.tenalg.multi_mode_dot(T,A[N])
        
    elif mode=='khat':

        for m in range(N):
            measure_mat = tl.tenalg.khatri_rao([a.T for a in A[m]],skip_matrix=m)

            #reshaped = tuple(A[m][k].shape[0] for k  in range(N))
            B[m] = tl.unfold(T,m)@measure_mat

    B[N] = tl.tenalg.multi_mode_dot(T,A[N])
    return B

def square_tensor_gen(n, r, dim=3, typ='lk', noise_level=0.1, seed=None, sparse_factor=0.2):
    '''
    :param n: size of the tensor generated n*n*...*n
    :param r: rank of the tensor or equivalently, the size of core tensor
    :param dim: # of dimensions of the tensor, default set as 3
    :param typ: identity as core tensor or low rank as core tensor
    :param noise_level: sqrt(E||X||^2_F/E||error||^_F)
    :return: The tensor with noise, and The tensor without noise
    '''
    if seed:
        np.random.seed(seed)

    types = set(['id', 'lk', 'fpd', 'spd', 'sed', 'fed', 'slk'])
    assert typ in types, "please set your type of tensor correctly"
    total_num = np.power(n, dim)

    if typ == 'id':
        # identity
        elems = [1 for _ in range(r)]
        elems.extend([0 for _ in range(n - r)])
        noise = np.random.normal(0, 1, [n for _ in range(dim)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        return X0 + noise * np.sqrt((noise_level ** 2) * r / total_num), X0

    if typ == 'spd':
        # Slow polynomial decay
        elems = [1 for _ in range(r)]
        elems.extend([1.0 / i for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        true_signal_mag = np.linalg.norm(X0) ** 2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        return X0 + noise * np.sqrt((noise_level ** 2) * true_signal_mag / total_num), X0

    if typ == 'fpd':
        # Fast polynomial decay
        elems = [1 for _ in range(r)]
        elems.extend([1.0 / (i * i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        true_signal_mag = np.linalg.norm(X0) ** 2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        return X0 + noise * np.sqrt((noise_level ** 2) * true_signal_mag / total_num), X0

    if typ == 'sed':
        # Slow exponential decay
        elems = [1 for _ in range(r)]
        elems.extend([np.power(10, -0.25 * i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        true_signal_mag = np.linalg.norm(X0) ** 2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        return X0 + noise * np.sqrt((noise_level ** 2) * true_signal_mag / total_num), X0

    if typ == 'fed':
        # Fast Exponential decay
        elems = [1 for _ in range(r)]
        elems.extend([np.power(10, (-1.0) * i) for i in range(2, n - r + 2)])
        X0 = generate_super_diagonal_tensor(elems, dim)
        true_signal_mag = np.linalg.norm(X0) ** 2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        return X0 + noise * np.sqrt((noise_level ** 2) * true_signal_mag / total_num), X0

    if typ == "lk":
        # Low rank
        core_tensor = np.random.uniform(0, 1, [r for _ in range(dim)])
        arms = []
        tensor = core_tensor
        for i in np.arange(dim):
            arm = np.random.normal(0, 1, size=(n, r))
            arm, _ = np.linalg.qr(arm)
            arms.append(arm)
            tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
        true_signal_mag = np.linalg.norm(tensor) ** 2
        noise = np.random.normal(0, 1, np.repeat(n, dim))
        X = tensor + noise * np.sqrt((noise_level ** 2) * true_signal_mag / total_num)
        return X, tensor, core_tensor, arms

    if typ == "slk":
        # Sparse low rank
        core_tensor = np.random.normal(0, 1, [r for _ in range(dim)])
        arms = []
        tensor = core_tensor
        for i in np.arange(dim):
            arm = np.random.normal(0, 1, size=(n, r))
            arm = arm * np.random.binomial(1, sparse_factor, size=(n, r))
            arms.append(arm)
            tensor = tl.tenalg.mode_dot(tensor, arm, mode=i)
        true_signal_mag = np.linalg.norm(tensor) ** 2
        tensor0 = tensor
        tensor = tensor + np.random.normal(0, 1, size=[n for _ in range(dim)]) \
                 * np.sqrt((noise_level ** 2) * true_signal_mag / total_num)
        return tensor, tensor0
    
def lsmlsvd_brks(A,B,R,mode='kron'):
    N = len(A.keys()) - 1
    U = []
    
    for m in range(N):
        if mode=='kron': 
            rhs = tl.unfold(B[m],m)
        else:
            rhs = B[m]

            
        F = np.linalg.lstsq(A[m][m], rhs,rcond=None)
        #Q,_ = np.linalg.qr(F[0])
        Q,_, _ = np.linalg.svd(F[0])

        U.append(Q[:,:R[m]])


    S_tilde = B[N]
    mr = list(S_tilde.shape)
    S_tilde = tl.unfold(S_tilde,0)
    for n in range(N):
        S_tilde = tl.unfold(S_tilde,n)

        mr[n] = R[n]

        S_new = np.linalg.lstsq(A[N][n]@U[n], S_tilde,rcond=None)
        S_tilde = tl.fold(S_new[0],n,mr)
    

    return S_tilde,U