# leave_one_out_recovery
Code to reproduce experiments and figures in the pre-print "Efficient and Generalized Low Rank Tensor Recovery"

## Quick-start
The file `run_trial.py` parses command line arguments to run and produce results with the provided parameters. For example, in an environment with the requirements installed and `loo_recovery.py` the command

``python run_trial.py 10 300 10 10 20 1e-3 g lk kron``

will run trials and save results in a csv. The arguments are number of trials, side length, rank, m, m_c, noise level, measurement type, tensor type, and leave one out type, so in this case 10 independent trials of a tensor with sidelengths of 300 of Tucker rank (10,10,10). The measurements will sketch a side to size 10 for producing leave-one-out measurement tensors of size (10,10,300), (10,300,10) and (300,10,10) as well as a measurement tensor of size (20,20,20) for estimating the core (two-pass scenario). The noise level is additive gaussian noise at an SNR of 30 db (`1e-3`), the component measurement matrices are guassian (`g`) the unsketched tensor is a randomly generated low rank (`lk`) tensor, and the measurement ensemble is equivalent to the Kronecker product of the component measurement matrices (`kron`)

Writing results to  results/results_<mmddtttttt_serial>.csv
Trial    meas    t_typ   loo     n       rank    m       m_c     SNR     2-error         1-error         compress        sketch_t        recover_t       angle
0       g       lk      kron    300      10      10      20      30      6.15E-04        1.35E-03       3.63E-03         4.72E-02        4.89E-02        6.58E-02
1       g       lk      kron    300      10      10      20      30      6.51E-04        1.59E-03       3.63E-03         4.76E-02        5.58E-02        8.27E-02
2       g       lk      kron    300      10      10      20      30      7.15E-04        1.53E-03       3.63E-03         4.47E-02        5.40E-02        7.86E-02
3       g       lk      kron    300      10      10      20      30      6.83E-04        2.25E-03       3.63E-03         4.70E-02        5.63E-02        9.13E-02
4       g       lk      kron    300      10      10      20      30      6.62E-04        1.39E-03       3.63E-03         4.61E-02        5.18E-02        7.01E-02
5       g       lk      kron    300      10      10      20      30      6.88E-04        1.57E-03       3.63E-03         4.47E-02        5.52E-02        8.98E-02
6       g       lk      kron    300      10      10      20      30      6.30E-04        1.38E-03       3.63E-03         4.45E-02        5.31E-02        9.61E-02
7       g       lk      kron    300      10      10      20      30      7.19E-04        1.28E-03       3.63E-03         4.97E-02        5.34E-02        8.37E-02
8       g       lk      kron    300      10      10      20      30      6.74E-04        1.36E-03       3.63E-03         4.78E-02        5.39E-02        8.53E-02
9       g       lk      kron    300      10      10      20      30      6.32E-04        1.42E-03       3.63E-03         4.49E-02        5.34E-02        6.13E-02
