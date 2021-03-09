# Development of a GPU-Accelerated Computational Platform for Bayesian Networks

This VRES project used GPUs to parallelize a Markov Chain Monte Carlo (MCMC) algorithm proposed by Neiswanger et al. (2014) to perform parameter estimation. The GPU implementation is written in CUDA and allows for both shared and global memory. It requires the NVCC pre-compiler which then calls the platform's C++ compiler. Full report and powerpoint presentation are also provided in the Documentation folder.

<h3>1. Generate Python samples using sample_generator.py (n_dim) (n_samples)</h3>

```python sample_generator.py 10 40000```

OR, using 2-dimensions and 10 000 samples: ```python sample_generator.py 2 10000```

<h3>2. Make project using Makefile provided</h3>



```make```

<h3>3. Use executable</h3>

Using global memory: ```mcmc_gpu.exe --n-blocks=30 --n-threads=1920 --n-dim=10 --sm=0```

OR, using shared memory: ```mcmc_gpu.exe --n-blocks=30 --n-threads=1920 --n-dim=2 --sm=1```

(Note: shared memory requires very small datasets to work and n_threads is the total number of threads spread across the blocks/grids.)

Usage: MCMC options are...

    Number of steps: --n_steps=10000
    Number of dimensions: --n_dim=10
    Evaluation frequency: --eval_freq=1000
    Store data in shared memory (requires small datasets): --sm=0
    Number of threads: --n_threads=256
    Number of blocks: --n_blocks=1
