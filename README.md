# leave_one_out_recovery
Code to reproduce experiments and figures in the pre-print "Efficient and Generalized Low Rank Tensor Recovery"

## Quick-start
The file `run_trial.py` parses command line arguments to run and produce results with the provided parameters. For example, in an environment with the requirements installed and `loo_recovery.py` the command

``python run_trial 10 300 10 10 20 1e-3 g lk kron``

will run trials and save results in a csv. The arguments are number of trials, side length, rank, m, m_c, noise level, measurement type, tensor type, and leave one out type, so in this case 10 independent trials of a tensor with sidelengths of 300 of Tucker rank (10,10,10). The measurements will sketch a side to size 10 for producing leave-one-out measurement tensors of size (10,10,300), (10,300,10) and (300,10,10) as well as a measurement tensor of size (20,20,20) for estimating the core (two-pass scenario). The noise level is additive gaussian noise at an SNR of 30 db (`1e-3`), the component measurement matrices are guassian (`g`) the unsketched tensor is a randomly generated low rank (`lk`) tensor, and the measurement ensemble is equivalent to the Kronecker product of the component measurement matrices (`kron`)
