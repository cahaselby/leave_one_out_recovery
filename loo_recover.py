"""
8 MAY 2025

This is the module needed to run leave one out recovery. 

One can use it to generate a ensemble of measurement matrices A to measure a randomly generated tensor T producing measurement tensors B and then recover the factors of the tucker (potentially approximation)

T = [[S_tilde; U_0, ..., U_N-1]] 

Below we have an example for some choices of parameteers:
 
Suppose
    dim = 3, number of modes of the tensor
    n = 300, side length of the tensor
    r = 10, tucker rank (which will be repeated for each mode (10,10,10))
    tt = 'lk', tensor type, which could be a random low tucker rank tensor
    eps = 1e-3, relative magnitude of the additive noise to be added to the tensor
    M = (20,20,20), sketching dimension for the leave-one-out measurment tensors
    M_c = (40,40,40), sketching dimensions for the measurement tensor for the core (one-pass scenario)
    mt = 'kron', method to use the matrices to measure the tensor

Generate a tensor with known properties to test on Y is noisey, Y_true is noiseless, S_true is the (a) core factor, U_true are the factor matrices

    Y, Y_true, S_true, U_true = square_tensor_gen(n, r, dim=3, typ=tt, noise_level=eps, seed=None, sparse_factor=0.2)

Create the measurement ensemble A_kron where M and M_c store the sketching dimensions to be used for the factors and core measurements:

    A_kron = measurement_ensemble(dim,N,M_c,M,my_random_matrix_generator,typ=mt)
            
Now create the measurement tensors and put them in dictionary B_kron using the ensemble A_kron

    B_kron = loo.measure_tensor(Y,A_kron,mode=mode)

Solve the least square problems for finding the factors U and core S_tilde from the measurements. This requires a single pass
    
    S_tilde,U = lsmlsvd_brks(A_kron,B_kron,R,mode=mode)
    
If you want the full tensor, can construct now from the (estiamted) factors

    T_hat_one_pass = tl.tucker_tensor.tucker_to_tensor((S_tilde,U))

Another pass on the original tensor can be used to calculate a more accurate core factor, just apply the transpose of calculated factors U to the tensor 
    
    _hat_two_pass = tl.tucker_tensor.tucker_to_tensor((tl.tenalg.multi_mode_dot(Y,U,transpose=True),U))

Calculate the relative errors with helper functions

    rel_error_one_pass = loo.eval_rerr(Y,T_hat_one_pass,Y_true)
    rel_error_two_pass = loo.eval_rerr(Y,T_hat_two_pass,Y_true)

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
from sklearn.cluster import KMeans

#These helper functions are for various random matrices used to construct measurement ensembles. 
#Helper function for returning a gaussian JL measurement matrix
def gauss_meas(dim, k):
    #np.random.seed(0)
    return np.sqrt(1/(k))*np.random.normal(0.0, 1.0, [k, dim])
    
#Rademacher along diagonal (D),  Hadamard (H), randomly sample the row (S)
def SHRT_meas(dim, k):
    D = np.diag(np.random.choice([1,-1], dim))
    H = np.sqrt(1/k)*hadamard(dim)
    idx = np.zeros(dim,dtype=bool)
    idx[:k] = True
    np.random.shuffle(idx)
    S = np.eye(dim)[idx]
    
    return S@H@D

#Rademacher along the diagonal (D), Discrete Fourier Transform (F), randomly sample the rows (S)
def RFD_meas(dim, k):
    D = np.diag(np.random.choice([1,-1], dim))
    F = np.sqrt(dim/k)*dft(dim)
    idx = np.zeros(dim,dtype=bool)
    idx[:k] = True
    np.random.shuffle(idx)
    S = np.eye(dim)[idx]
    
    return S@F@D

#Super diagonal tensors only have non-zero entries along coordinates i,i,i for i in [dim], use this as a tensor to measure
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
    :param R: parameters fo generating the random matrix: std (standard devidation for each entry); typ
    (u: uniform; g: Gaussian; sp: sparsity=sparse_factor; sp0: sparsity=2/3;sp1:sparsity=1-1/sqrt(n) )
    :return: random matrix
    '''

    #np.random.seed(random_seed)
    types = set(['r','g', 'u', 'sp', 'sp0', 'sp1','rfd','real_rfd', "shrt"])
    assert typ in types, "please set your type of random variable correctly"

    #type of random matrix used,  g = gaussian, rfd = sampled rademacher x DFT, shrt = sampled rademacher x hadamard, u = uniform, sp = sparse, r = rademacher, sp0 = alternate sparse, sp1 = alternate sparse
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
    """
    :param T: tensor
    :param relnoise: the proportion of T's norm that the white noise will be scaled to and added to T
    :return: T + scaled Noise
    """
    Noise = np.random.normal(size=T.shape)
    #Normalize 
    Noise = Noise / tl.norm(Noise)
    #Scale relative to norm of T
    Noise = (relnoise *tl.norm(T) )* Noise

    return T +  Noise

def average_results(results,headers,groupfield,fieldy):
    """
    helper function that takes result rows from run_trial.py and calculates basic statistics 
    """
    df = pd.DataFrame(results,columns = headers)


    agg = df.groupby([groupfield])[fieldy].agg(['mean', 'sem', 'median'])
    agg['ci95_hi'] = agg['mean'] + 1.96* agg['sem']
    agg['ci95_lo'] = agg['mean'] - 1.96* agg['sem']

    return agg

def measurement_ensemble(N,d,S,R,meas_func,typ='g',std=1):
    """
    Returns a dictionary of random matrices that can be used to measure a tensor in a leave one out fasion. e.g. A[0][1] will store a random matric of type typ intended to measure mode 1 when creating a measurement tensor thatleaves out mode 0
    
    Stores the matrices for creating the core sketch A[N]
    
    :param N: integer number of modes of the tensor. e.g. n=3
    :param d: tuple that lists the side lengths of the tensor to be measured, e.g. (300,200,300)
    :param S: tuple of the sketching dimensions for the core sketch e.g. (20,20,20)
    :param R: tuple of the sketching dimensions for the leave-one-out measurements so R = (10,10,10) would result in A[0][1] storing a matrix of size 10 x 300
    :meas_func: function handle that will generate the random matrix, e.g. my_random_matrix_generator if the user wishes to use the types of random matrices defined in this module
    :typ: type of random matrix used,  g = gaussian, rfd = sampled rademacher x DFT, shrt = sampled rademacher x hadamard, u = uniform, sp = sparse, r = rademacher, sp0 = alternate sparse, sp1 = alternate sparse
    :std: deprecated - scaling factor
    :return A: a dictionary of (N+1)*d random matrices that can be used to measure a tensor in a leave-one-out fashion using the measure_tensor function in this module.
    """
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
    """
    Returns a dicttionary of leave-one-out measurement tensors which are calculated by applying the ensemble A to tensor T in either a Kronecker or Khatri-Rhao fashion. 
    
    The last entry in the returned dictionary is the measurement tensor which has all modes sketched and is suitable for estimating the core factor of T
    
    :param T: The tensor to be measured
    :param A: The dictionary of matrices to be used to measure T, e.g. the output of the measurement_ensemble function
    :param mode: 'khat' or 'kron' method for using the measurement ensemble
   
    :return B: a dictionary of measurement tensors, one leave-one-out tensor per mode of T plus the last entry is a tensor where all modes are sketched for use in calculating the core (in the one-pass scenario).
    """
    #number of modes to the tensor
    N = len(A.keys()) - 1

    #dictionary to store the measurement tensors
    B = {}

    #if its kronecker style measurements, for each mode m, generate a measurement tensor B[m] by sketching each mode i by A[m][i]. Where i=m there should be no compression since that matrix will be full rank in the leave-one-out ensemble
    if mode=='kron':
        for m in range(N):
            B[m] = tl.tenalg.multi_mode_dot(T,A[m],skip=m)

        #B[N] = tl.tenalg.multi_mode_dot(T,A[N])
        
    elif mode=='khat':
        #in the Khatri-Rhao instance, for each mode m flatten the tensor and right multiply by the khatri rhao products of A[m] for all but the left out mode
        for m in range(N):
            measure_mat = tl.tenalg.khatri_rao([a.T for a in A[m]],skip_matrix=m)

            #reshaped = tuple(A[m][k].shape[0] for k  in range(N))
            B[m] = tl.unfold(T,m)@measure_mat

    #the last entry in the dictionary is for core estimation in the one-pass scenario. This measurement tensor is sketched on all modes, with none left out using matrices in A[N]
    B[N] = tl.tenalg.multi_mode_dot(T,A[N])
    return B

def square_tensor_gen(n, r, dim=3, typ='lk', noise_level=0.1, seed=None, sparse_factor=0.2):
    '''
    This function is for generating different types of tensors to measure and recover

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
    '''
    After potentially unfolding tensors, this function solves a series of linear least squared problems (e.g. AU = B, where U is unknown) 
    
    This can be used to recover factors from leave-one-out style measurements (kronecker or khatri-rhao) of tensors which permit a Tucker or multi-mode Tensor Singular Value Decomposition

    Returns the factors S_tilde, U of the recovered tensor, S_tilde is the core factor (in the one pass scenario) and U is a list of factor matrices for each mode.

    :param A: Dictionary of measurement matrices such as the output of measurement_ensemble. For example A[0][1] is the matrix used to compress mode 1 in the measurement tensor which leaves out mode 0.
    :param B: Dictionary of measurement tensors, such as the output of measure_tensor. tensor B[m] leaves mode m uncompressed. The last entry, B[N] is assumed to be sketched in all dimensions and used for estimating the core factor in the one-pass scenario
    :param R: tuple that is the (truncated) tucker rank of the tensor to be recovered
    :param mode: 'khat' or 'kron' depending on if the method to produce B using A was kronecker or khatri-rhao
    '''

    #Number of modes
    N = len(A.keys()) - 1

    #this is where the factor matrices go
    U = []
    
    for m in range(N):
        #for each mode prepare the right hand side of the least square problem. If its Kronecker, B will be a tensor
        if mode=='kron': 
            rhs = tl.unfold(B[m],m)
        #if its not kronecker, B should already be unfolded so the right hand side doesn't need anything special 
        else:
            rhs = B[m]

        #solve the least square problem A[m][m] x F = B
        F = np.linalg.lstsq(A[m][m], rhs,rcond=None)
        
        #Truncate and normalize the solution by finding SVD of unknown, keeping the first R[m] columns. Normalization not strictly necessary but useful convention we can impose on the factors
        #This would be an alternative method to do the same Q,_ = np.linalg.qr(F[0])
        Q,_, _ = np.linalg.svd(F[0])

        U.append(Q[:,:R[m]])

    #this loop solves iteratively for the core factor S_tilde on each mode; this is the way to estimate the core in the one-pass scenario
    #this is equivalent (but more memory movement efficient) to solving (A[N][0] x U_0) kron (A[N][1] x U_1) kron ... kron (A[N-1][N-1] U_N-1) vec(S_tilde) = vec(B[N])
    S_tilde = B[N]
    mr = list(S_tilde.shape)
    S_tilde = tl.unfold(S_tilde,0)
    for n in range(N):
        S_tilde = tl.unfold(S_tilde,n)

        mr[n] = R[n]

        S_new = np.linalg.lstsq(A[N][n]@U[n], S_tilde,rcond=None)
        S_tilde = tl.fold(S_new[0],n,mr)
    

    return S_tilde,U
