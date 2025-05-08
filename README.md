# Leave-One-Out Measurement and Recovery

Code to reproduce experiments and figures in the pre-print *[Fast and Low-Memory Compressive Sensing Algorithms for Low Tucker-Rank Tensor Approximation from Streamed Measurements.](https://arxiv.org/abs/2308.13709)*

## Quick-start
The file `run_trial.py` parses command line arguments to run and produce results with the provided parameters. For example, in an environment with the ``requirements.txt`` installed and `loo_recovery.py` the command

``python run_trial.py 10 300 10 10 20 1e-3 g lk kron``

will run trials and save results in a csv. The arguments are number of trials, side length, rank, m, m_c, noise level, measurement type, tensor type, and leave one out type, so in this case 10 independent trials of a tensor with sidelengths of 300 of Tucker rank (10,10,10). The measurements will sketch a side to size 10 for producing leave-one-out measurement tensors of size (10,10,300), (10,300,10) and (300,10,10) as well as a measurement tensor of size (20,20,20) for estimating the core (two-pass scenario). The noise level is additive gaussian noise at an SNR of 30 db (`1e-3`), the component measurement matrices are guassian (`g`) the unsketched tensor is a randomly generated low rank (`lk`) tensor, and the measurement ensemble is equivalent to the Kronecker product of the component measurement matrices (`kron`).

The output then writes to a timestamped csv. Sample output is below:

``Writing results to  results/results_<mmddtttttt_serial>.csv``
|Trial|meas|t_typ|loo| n|rank|m|m_c| SNR| 2-error|  1-error|  compress| sketch_t| recover_t|angle|
|-|-|--|----|---|--|--|--|--|--------|---------|--------|----------|---------|---------|
|0|g|lk|kron|300|10|10|20|30|6.15E-04| 1.35E-03|3.63E-03|  4.72E-02| 4.89E-02| 6.58E-02|
|1|g|lk|kron|300|10|10|20|30|6.51E-04| 1.59E-03|3.63E-03|  4.76E-02| 5.58E-02| 8.27E-02|
|2|g|lk|kron|300|10|10|20|30|7.15E-04| 1.53E-03|3.63E-03|  4.47E-02| 5.40E-02| 7.86E-02|
|3|g|lk|kron|300|10|10|20|30|6.83E-04| 2.25E-03|3.63E-03|  4.70E-02| 5.63E-02| 9.13E-02|
|4|g|lk|kron|300|10|10|20|30|6.62E-04| 1.39E-03|3.63E-03|  4.61E-02| 5.18E-02| 7.01E-02|
|5|g|lk|kron|300|10|10|20|30|6.88E-04| 1.57E-03|3.63E-03|  4.47E-02| 5.52E-02| 8.98E-02|
|6|g|lk|kron|300|10|10|20|30|6.30E-04| 1.38E-03|3.63E-03|  4.45E-02| 5.31E-02| 9.61E-02|
|7|g|lk|kron|300|10|10|20|30|7.19E-04| 1.28E-03|3.63E-03|  4.97E-02| 5.34E-02| 8.37E-02|
|8|g|lk|kron|300|10|10|20|30|6.74E-04| 1.36E-03|3.63E-03|  4.78E-02| 5.39E-02| 8.53E-02|
|9|g|lk|kron|300|10|10|20|30|6.32E-04| 1.42E-03|3.63E-03|  4.49E-02| 5.34E-02| 6.13E-02|

## Streaming Demo

In ``video-demo.ipynb`` we have a demonstration of the leave-one-out measurements and recovery applied to a streaming, one-pass of a data tensor. The data tensor in this case is the video found in <https://figshare.com/articles/media/Walking_Past_Camera/15135186?file=29084181> similar to what is done for this paper: <https://github.com/OsmanMalik/tucker-tensorsketch>.

## Example of using loo_recover.py

Below we have an example for some choices of parameteers of how to use the functions exposed in ``loo_recover.py``
 
Suppose

    ``dim = 3, number of modes of the tensor``
    
    ``n = 300, side length of the tensor``
    
    ``r = 10, tucker rank (which will be repeated for each mode (10,10,10))``
    
    ``tt = 'lk', tensor type, which could be a random low tucker rank tensor``
    
    ``eps = 1e-3, relative magnitude of the additive noise to be added to the tensor``
    
    ``M = (20,20,20), sketching dimension for the leave-one-out measurment tensors``
    
    ``M_c = (40,40,40), sketching dimensions for the measurement tensor for the core (one-pass scenario)``
    
    ``mt = 'g', measurement type of random variable e.g. guassian random matrices``
    
    ``mode = 'kron', method to use the matrices to measure the tensor``

Generate a tensor with known properties to test on Y is noisey, Y_true is noiseless, S_true is the (a) core factor, U_true are the factor matrices

   `` Y, Y_true, S_true, U_true = loo_recover.square_tensor_gen(n, r, dim=3, typ=tt, noise_level=eps, seed=None, sparse_factor=0.2)``

Create the measurement ensemble A_kron where M and M_c store the sketching dimensions to be used for the factors and core measurements:

   `` A_kron = loo_recover.measurement_ensemble(dim,N,M_c,M,my_random_matrix_generator,typ=mt)``
            
Now create the measurement tensors and put them in dictionary B_kron using the ensemble A_kron

    ``B_kron = loo_recover.measure_tensor(Y,A_kron,mode=mode)``

Solve the least square problems for finding the factors U and core S_tilde from the measurements. This requires a single pass
    
   `` S_tilde,U = loo_recover.lsmlsvd_brks(A_kron,B_kron,R,mode=mode)``
    
If you want the full tensor, can construct now from the (estiamted) factors

   `` T_hat_one_pass = tensorly.tucker_tensor.tucker_to_tensor((S_tilde,U))``

Another pass on the original tensor can be used to calculate a more accurate core factor, just apply the transpose of calculated factors U to the tensor 
    
   ``T_hat_two_pass = tensorly.tucker_tensor.tucker_to_tensor((tl.tenalg.multi_mode_dot(Y,U,transpose=True),U))``

Calculate the relative errors with helper functions

    ``rel_error_one_pass = loo_recover.eval_rerr(Y,T_hat_one_pass,Y_true)``
    ``rel_error_two_pass = loo_recover.eval_rerr(Y,T_hat_two_pass,Y_true)``
