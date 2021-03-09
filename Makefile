all: mcmc_gpu.cu
	nvcc mcmc_gpu.cu -lcurand -o mcmc_gpu
	
clean:
	rm mcmc_gpu.exp mcmc_gpu.lib *.txt