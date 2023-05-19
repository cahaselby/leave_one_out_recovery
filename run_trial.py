
"""
12 MAY 2023

This is a script is for running a job and gathering and saving results into a dataframe and csv 

author: Cullen Haselby 
"""
#########################
# IMPORTS
#########################

import csv
from datetime import datetime
import sys
import timeit
import numpy as np
import tensorly as tl
import loo_recover as loo
from random import randint, randrange
from scipy.linalg import subspace_angles,dft

###########################################################
# PARSE INPUTS, RUN TRIALS, STORE RESULTS IN CSV
##########################################################


if __name__ == "__main__":
    
    #Take the command line arguments, get rid of the script name
    args = sys.argv[1:]

    #store the settings into a list, splitting and casting as ints

    ts=[int(t) for t in args[0].split(",")]
    ns=[int(n) for n in args[1].split(",")]
    rs=[int(r) for r in args[2].split(",")]
    ms=[int(m) for m in args[3].split(",")]
    m_cs=[int(m_c) for m_c in args[4].split(",")]
    noises=[float(eps) for eps in args[5].split(",")]
    measure_types=[m_t for m_t in args[6].split(",")]
    tensor_types=[t_t for t_t in args[7].split(",")]
    modes=[mode for mode in args[8].split(",")]

    #organize the parameters into a list of tuples
    params=[(t,n,r,eps,m,m_c,mt,tt,mode) for t in ts for n in ns for r in rs for eps in noises for m in ms for m_c in m_cs for mt in measure_types for tt in tensor_types for mode in modes]

    #open a file for writing the results in.
    now = datetime.now()
    dt_string = now.strftime("%m%d%H%M%S")
    
    name="results/results_"+dt_string+"_"+str(randint(100, 999))+".csv"
     
    print("Writing results to ", name)
    
    print("Trial \t meas \t t_typ \t loo \t n \t rank \t m \t m_c \t SNR \t 2-error \t 1-error \t compress \t sketch_t \t recover_t \t angle")
    
    #cols=["trial","meas", "tensor", "loo_mode", "n","r","m","m_c","SNR","2rel_error","1rel_error","compress", "angle", "time"]
    cols=["trial","meas", "tensor", "loo_mode", "n","r","m","m_c","SNR","2rel_error","1rel_error","compress", "sketch_t", "recover_t","angle"]
    with open(name, 'a', newline='') as f_object:  
        # Pass the CSV  file and write in the column headers
        write = csv.writer(f_object)
        write.writerow(cols)  
        f_object.close()
    
    results = []
    for p in params:
        T,n,r,eps,m,m_c,mt,tt,mode = p
        for t in range(T):
            dim=3

            #print("Running trial(s) for parameters ", p)


            gen_data = loo.square_tensor_gen(n, r, dim=3, typ=tt, noise_level=eps, seed=None, sparse_factor=0.2)
            if len(gen_data) != 4:
                Y = gen_data[0]
                Y_true = gen_data[1]
                U_true = [np.eye(n)[:,:r] for _ in range(dim)]
            else:
                Y, Y_true, S_true, U_true = gen_data
                
            N = [n for _ in range(dim)]
            R = np.array([r for _ in range(dim)])
            M = [m for _ in range(dim)]
            M_c = [m_c for _ in range(dim)]
            
            
            if mt=='mix':
                A_kron_g = loo.measurement_ensemble(dim,N,M_c,M,loo.my_random_matrix_generator,typ='g',std=(1/np.sqrt(m)))
                A_kron_sp0 = loo.measurement_ensemble(dim,N,M_c,M,loo.my_random_matrix_generator,typ='sp0',std=(1/np.sqrt(m)))
                A_kron_rfd = loo.measurement_ensemble(dim,N,M_c,M,loo.my_random_matrix_generator,typ='rfd',std=(1/np.sqrt(m)))
                A_kron = A_kron_g
                A_kron[0][1] = A_kron_sp0[0][1]
                A_kron[0][2] = A_kron_rfd[0][2]

                A_kron[1][1] = A_kron_sp0[1][1]
                A_kron[1][2] = A_kron_rfd[1][2]

                A_kron[2][1] = A_kron_sp0[2][1]
                A_kron[2][2] = A_kron_rfd[2][2]

                A_kron[3][1] = A_kron_sp0[3][1]
                A_kron[3][2] = A_kron_rfd[3][2]
            else:
                A_kron = loo.measurement_ensemble(dim,N,M_c,M,loo.my_random_matrix_generator,typ=mt)
                
            starttime = timeit.default_timer()
            B_kron = loo.measure_tensor(Y,A_kron,mode=mode)
            sketch_time = timeit.default_timer() - starttime
            
            starttime = timeit.default_timer()
            S_tilde,U = loo.lsmlsvd_brks(A_kron,B_kron,R,mode=mode)
            recover_time = timeit.default_timer() - starttime
            T_hat_one_pass = tl.tucker_tensor.tucker_to_tensor((S_tilde,U))
            T_hat_two_pass = tl.tucker_tensor.tucker_to_tensor((tl.tenalg.multi_mode_dot(Y,U,transpose=True),U))
            rel_error_one_pass = loo.eval_rerr(Y,T_hat_one_pass,Y_true)
            rel_error_two_pass = loo.eval_rerr(Y,T_hat_two_pass,Y_true)
            compress = 0

            for v in B_kron.values():
                compress += v.size / Y.size

            angles = []
            for i in range(dim):
                angles.append(np.rad2deg(subspace_angles(U[i],U_true[i])))
            angles = np.array(angles)     

            if eps == 0: 
                SNR = np.inf
            else: 
                SNR = round(10*np.log(tl.norm(Y_true)/tl.norm(Y-Y_true))/np.log(10))

            print("{} \t{} \t{} \t{} \t{} \t {} \t {} \t {} \t {} \t {:.2E} \t {:.2E} \t{:.2E} \t {:.2E} \t {:.2E} \t {:.2E} ".format(t,mt,tt,mode,n,r,M[0],M_c[0],SNR,rel_error_two_pass,rel_error_one_pass,compress,sketch_time,recover_time, np.max(angles[:,0])))

            results.append((t,mt,tt,mode,n,r,M[0],M_c[0],SNR,rel_error_two_pass,rel_error_one_pass,compress,sketch_time,recover_time, np.max(angles[:,0])))

        with open(name, 'a+', newline='') as f_object:
            write = csv.writer(f_object)
            write.writerows(results)
            f_object.close()
            
    
        
