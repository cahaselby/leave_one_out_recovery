# leave_one_out_recovery
Code to reproduce experiments and figures in the pre-print "Efficient and Generalized Low Rank Tensor Recovery"

after installing loo_recovery.py and run_trial.py, the following snippet will run trials and save results in a csv.

python run_trial 10 300 10 10 20 1e-3 g lk kron

arguments are number of trials, side length, rank, m, m_c, noise level, measurement type, tensor type, and leave one out type
