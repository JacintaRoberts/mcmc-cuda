#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846 // known value of pi


//------------------CUDA ERROR HANDLING------------------//
#define gpuErrChk(e) gpuAssert(e, __FILE__, __LINE__)

// Catch GPU errors in CUDA runtime calls
inline void gpuAssert(cudaError_t call, const char* file, int line) {
    if (call != cudaSuccess) {
        fprintf(stderr, "gpuAssert: %s %s %d\n", cudaGetErrorString(call), file, line);
        exit(-1);
    }
}


//------------------CALCULATIONS------------------//
// Calculate the covariance estimate on GPU
__device__ void calc_cov(float* cov, float* inv_cov, float* est_cov, int n, int n_threads) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread ID
    int offset = threadID * n;
    int i, j; 

    // Partition (1/covariance) array so that each thread is working on a different section
    for (i = offset; i < offset + n; i++) {
        inv_cov[i] = 1.f / cov[i];
    }

    // If the first thread, combine results
    if (threadID == 0) {
        for (i = 0; i < n; i++) {
            est_cov[i] = 0.f; // Initialise
            // Update using results from each thread
            for (j = 0; j < n_threads; j++) {
                est_cov[i] += inv_cov[i + j * n];
            }
            est_cov[i] = 1.f / est_cov[i]; // Inverse
        }
    }
    __syncthreads(); // Synchronise threads within the same block
}

// Calculate the population mean (mu) estimate on GPU
__device__ void calc_mu(float* mu, float* inv_cov, float* inv_mu, float* est_cov, float* est_mu, int n, int n_threads) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    int offset = threadID * n;
    int i, j;

    // Partition inv_mu array so that each thread is working on a different section
    for (i = offset; i < offset + n; i++) {
        inv_mu[i] = mu[i] * inv_cov[i];
    }

    // If the first thread, combine results
    if (threadID == 0) {
        for (i = 0; i < n; i++) {
            est_mu[i] = 0.f; // Initialise to 0
            // Update mu estimate using results from each thread
            for (j = 0; j < n_threads; j++) {
                est_mu[i] += inv_mu[i + j * n];
            }
            // Scale using the covariance estimate
            est_mu[i] = est_cov[i] * est_mu[i];
        }
    }
    __syncthreads(); // Synchronise threads within the same block
}

// Calculates determinant
__device__ float alt_calc_det(float* cov, int n) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    float det = 1.f;
    int initial_idx = n * threadID;

    for (int i = initial_idx; i < initial_idx + n; i++) {
        det *= cov[i];
    }

    return det;
}

// Calculates vector product
__device__ float calc_vec_mat_vec_prod(float* cov, float* data, float* mu, int data_idx, int n) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int j;
    int initial_idx = n * threadID;
    float diff;
    float cum_sum = 0.f;

    for (int i = initial_idx; i < initial_idx + n; i++) {
        j = i - initial_idx + data_idx;
        diff = data[j] - mu[i];
        cum_sum += diff * diff * 1. / cov[i];
    }
    return cum_sum;
}

// Calculates the log of the determinant
__device__ float get_log_det(float* A, int n) {
    float det = alt_calc_det(A, n);
    return log(det);
}

// Calculates the log likelihood (assuming Normal errors)
__device__ float get_log_likelihood(float* data, float* mu, float* cov, float cov_det, int data_idx, int n) {
    float t1, t2, t3;
    float L;
    float fl_inf = 10000000000000000000;

    t1 = -0.5 * cov_det;
    t2 = -0.5 * calc_vec_mat_vec_prod(cov, data, mu, data_idx, n);
    t3 = -0.5 * n * log(2 * PI);

    L = t1 + t2 + t3;

    if (isnan(L)) {
        return -1 * fl_inf;
    }
    else {
        return L;
    }
}

// Returns the log likelihood for all of the data, summing the result.
__device__ float get_total_log_likelihood(float* cov, float* mu, float* data, int n_samples_per_thread, int n_samples, int n) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int data_idx;
    float cov_det = get_log_det(cov, n);
    int data_offset = threadID * n * n_samples_per_thread;
    float cum_L = 0.f;

    for (data_idx = data_offset; data_idx < data_offset + n * n_samples_per_thread; data_idx += n) {
        cum_L += get_log_likelihood(data, mu, cov, cov_det, data_idx, n);
    }
    return cum_L;
}

// Calculates the square root of the cumulative sum
__device__ float calc_l2_norm(float* mu, float* true_mu, int n) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = n * threadID;
    float diff;
    float cum_sum = 0.;
    float result;

    for (int i = offset; i < offset + n; i++) {
        diff = true_mu[i] - mu[i];
        cum_sum += diff * diff;
    }

    result = sqrt(cum_sum);
    return result;
}


//------------------RANDOM GENERATORS------------------//
// Initialise random state for each CUDA thread
__device__ void cuda_rand_init(curandState* state) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int seed = 0; // Fixed seed to 0 for reproducible results, otherwise just use clock64()
    curand_init(seed + threadID, 0, 0, &state[threadID]); // Sets up initial cuda state for random generator
}

// Generates 2 psuedorandom floats from uniform distribution to update rand_num and rand_ints (for each thread)
__device__ void gen_uniform(curandState* state, float* rand_num, int* rand_ints, int max_int) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    // Returns a pseudorandom float uniformly distributed between 0 and 1
    float rand_int_num = (float)(curand_uniform(&state[threadID]));
    rand_num[threadID] = (float)(curand_uniform(&state[threadID]));
    rand_ints[threadID] = int(max_int * rand_int_num);
}

// Generates 2 pesudorandom floats from normal distribution to update rand_mu and rand_cov (for each thread)
__device__ void gen_normal(curandState* state, float sigma_1, float* norm_1, float sigma_2, float* norm_2) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    // Returns a pseudorandom float normally distributed with mean 0 and std dev of sigma_1 (or sigma_2)
    norm_1[threadID] = (float)(curand_normal(&state[threadID]) * sigma_1); // rand_mu
    norm_2[threadID] = (float)(curand_normal(&state[threadID]) * sigma_2); // rand_cov
}

// Updates the random variables for each thread
__device__ void generate_random_nums(curandState* state, float* rand_mu, float* rand_cov, float mu_step, float cov_step, float* rand_num, int* rand_ints, int n) {
    // Gen random numbers from the uniform distribution to update rand_num and rand_ints
    int n_params = 2 * n;
    gen_uniform(state, rand_num, rand_ints, n_params);
    // Gen random numbers from the normal distribution to update rand_mu and rand_cov
    gen_normal(state, mu_step, rand_mu, cov_step, rand_cov);
}

//------------------PERTURBATIONS------------------//
// Peturbates the covariance
__device__ void perturb_cov(float* old_cov, float* new_cov, int param_idx, float rand_cov_num, int n) {
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = threadID * n;
    int idx = param_idx - n;

    float new_val = old_cov[offset + idx] + rand_cov_num; // Add random noise from proposal distribution to old cov sample
    if (new_val > 0) { // If covariance is positive, set it as the new covariance
        new_cov[offset + idx] = new_val;
    }
}

// Peturbates a random parameter from the array parameters using a Normal dist with std_dev = step size in the array
__device__ void perturb_params(float* old_cov, float* old_mu, float* new_cov, float* new_mu, int* rand_ints, float* rand_mu, float* rand_cov, int n) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = threadID * n;
    int param_idx = rand_ints[threadID]; // Pick parameter to perturb

    if (param_idx < n) {
        new_mu[param_idx + offset] = old_mu[param_idx + offset] + rand_mu[threadID]; // Add random noise from proposal distribution to old mu sample
    }
    else {
        perturb_cov(old_cov, new_cov, param_idx, rand_cov[threadID], n); // Otherwise perturb thee covariance
    }
}


//------------------HELPERS------------------//
// Initialize the array with desired value (for mean and covariance)
__device__ void init_array(float initial_val, float* curr_val, int n) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = threadID * n;
    for (int i = offset; i < offset + n; i++) {
        curr_val[i] = initial_val;
    }
}

// Read data file (into temporary) and returns the number of samples
__host__ int count_n_samples(int n) {
    unsigned int i = 0;
    FILE* input;
    char input_file[50];
    float temp;
    sprintf(input_file, "samples_%d.txt", n);
    input = fopen(input_file, "r");
    // Count the number of floats
    while ((fscanf(input, "%f", &temp)) != EOF) {
        i++;
    }
    fclose(input);
    return i / n; // Return number of samples (total number of floats divided by the number of dimensions)
}

// Read data file into h_data for processing
__host__ void read_samples(int n_samples, int n, float* data) {
    unsigned int i = 0;
    FILE* input;
    char input_file[50];
    sprintf(input_file, "samples_%d.txt", n);
    input = fopen(input_file, "r");
    while ((fscanf(input, "%f", &data[i])) != EOF) { // Read each floating point number into new position within h_data
        i++;
    }
    fclose(input);
}

// Copies a vector
__device__ void vec_cpy(float* source_vec, float* destination_vec, int n) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = threadID * n;
    for (int i = offset; i < offset + n; i++) {
        destination_vec[i] = source_vec[i];
    }
}

// Print usage to command prompt
void print_usage(int default_n_steps, int default_n, int default_spacing, int default_shared, int default_n_threads, int default_n_blocks) {
    printf("Usage: MCMC options are...\n");
    printf("    Number of steps: --n_steps=%d\n", default_n_steps);
    printf("    Number of dimensions: --n_dim=%d\n", default_n);
    printf("    Evaluation frequency: --eval_freq=%d\n", default_spacing);
    printf("    Store data in shared memory (requires small datasets): --sm=%d\n", default_shared);
    printf("    Number of threads: --n_threads=%d\n", default_n_threads);
    printf("    Number of blocks: --n_blocks=%d\n", default_n_blocks);
}

// Matches the user input variable
int match_flag_id(char* arg, const char* keyword_array[]) {
    int id = 6;
    int i;
    // Search for flag match i
    for (i = 0; i < 6; i++) {
        if (arg[id] == keyword_array[i][id]) {
            return i;
        }
    }
    return -1;
}

// Argument parsing so that program parameters can be easily changed for experimental analysis
int* parse_args(int argc, char* argv[]) {
    // Initialise variables
    int i, id, init_id;
    int* args = (int*)malloc(6 * sizeof(int));
    for (i = 0; i < 6; i++) {
        args[i] = -1;
    }

    // Expected key words
    const char* keyword_array[6];
    keyword_array[0] = "--n_steps";
    keyword_array[1] = "--n_dim";
    keyword_array[2] = "--eval_freq";
    keyword_array[3] = "--sm";
    keyword_array[4] = "--n_threads";
    keyword_array[5] = "--n_blocks";

    // Default args
    int default_n_steps = 10000;
    int default_n = 10;
    int default_spacing = 1000;
    int default_shared = 0;
    int default_n_threads = 256; // ideally 128-256 threads per block
    int default_n_blocks = 1; // atleast as many as SMs available

    // Compare each argument flag
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { // Can ask for usage instructions with help
            print_usage(default_n_steps, default_n, default_spacing, default_shared, default_n_threads, default_n_blocks);
            exit(0);
        }
        else {
            id = match_flag_id(argv[i], keyword_array);
            if (id != -1) { // Flag matched
                init_id = strlen(keyword_array[id]) + 1;
                argv[i] += init_id;
                sscanf(argv[i], "%d", &args[id]); // Read and store value into appropriate id
            }
            else { // Flag mismatch occurred
                print_usage(default_n_steps, default_n, default_spacing, default_shared, default_n_threads, default_n_blocks);
                exit(0);
            }
        }
    }

    // Set other parameters to default values
    if (args[0] == -1) {
        printf("Number of steps not provided. Defaulting to %d steps.\n", default_n_steps);
        args[0] = default_n_steps;
    }
    if (args[1] == -1) {
        printf("Number of dimensions not provided. Defaulting to %d dimensions.\n", default_n);
        args[1] = default_n;
    }
    if (args[2] == -1) {
        printf("Evaluation frequency not provided. Defaulting to evaluating every %d steps.\n", default_spacing);
        args[2] = default_spacing;
    }
    if (args[3] == -1) {
        printf("Shared memory not specified. Defaulting to using global memory.\n");
        args[3] = default_shared;
    }
    if (args[4] == -1) {
        printf("Number of threads not specified. Defaulting to using %d.\n", default_n_threads);
        args[4] = default_n_threads;
    }
    if (args[5] == -1) {
        printf("Number of blocks not specified. Defaulting to using %d.\n", default_n_blocks);
        args[5] = default_n_blocks;
    }
    return args;
}

// Prints the parameter configuration to the command window
void print_params(int n_steps, int n, int spacing, int shared, int n_threads, int n_blocks, int n_samples, int n_samples_per_thread) {
    printf("N steps: %d\n", n_steps);
    printf("Number of dimensions: %d\n", n);
    printf("Evaluation frequency spacing: %d\n", spacing);
    printf("Using shared mem: %d\n", shared);
    printf("Using # threads: %d\n", n_threads);
    printf("Using # blocks: %d\n", n_blocks);
    printf("Number of samples (per dim): %d, number of total threads: %d, number of samples per thread: %d.\n", n_samples, n_threads, n_samples_per_thread);
}

// Writes the output of the MCMC to files for further analysis
void output_results(float* h_mu, float* h_cov, float* h_est_mu, float* h_rand, int n, int n_threads, int n_sample_points, float millis) {
    int i;
    // Write h_mu out to mu_data.txt
    FILE* f = fopen("mu_data.txt", "w");
    for (i = 0; i < n * n_threads; i++) {
        fprintf(f, "%f ", h_mu[i]);
    }
    fclose(f);

    // Write h_est_mu out to mu_ev_data.txt (evolution of mu)
    f = fopen("mu_ev_data.txt", "w");
    for (i = 0; i < n_sample_points * n; i++) {
        fprintf(f, "%f ", h_est_mu[i]);
    }
    fclose(f);

    // Write h_cov out to cov_data.txt
    f = fopen("cov_data.txt", "w");
    for (i = 0; i < n * n_threads; i++) {
        fprintf(f, "%f ", h_cov[i]);
    }
    fclose(f);

    // Write h_rand out to rand_data.txt
    f = fopen("rand_data.txt", "w");
    for (i = 0; i < n_threads; i++) {
        fprintf(f, "%f ", h_rand[i]);
    }
    fclose(f);

    // Write millis out to timing.txt
    f = fopen("timing.txt", "w");
    fprintf(f, "%f ", millis);
    fclose(f);
}

//------------------MCMC STEP------------------//
// Computes a single step of the MCMC algorithm
__device__ void mcmc_step(float* curr_L, float* new_cov, float* new_mu, float* old_cov, float* old_mu, int* rand_ints, float* rand_mu, float* rand_cov, float* rand_num, int n, int n_samples_per_thread, int n_samples, int* take_step, float* data) {
    int threadID = threadIdx.x + blockDim.x * blockIdx.x;
    float old_L = curr_L[threadID];
    float new_L = get_total_log_likelihood(new_cov, new_mu, data, n_samples_per_thread, n_samples, n);
    float threshold;
    if (new_L > old_L) { // If the fit (total log likelihood) has improved, take a step and update likelihood
        take_step[threadID] = 1;
        curr_L[threadID] = new_L;
    }
    else { // Otherwise accept or reject if less than threshold (exponential raised to the difference between the new and old likelihood)
        threshold = exp(new_L - old_L); 
        if (rand_num[threadID] < threshold) { // Accept or reject
            take_step[threadID] = 1; // Step was taken and update likelihood
            curr_L[threadID] = new_L;
        }
        else {
            take_step[threadID] = 0; // Did not take a step
        }
    }
}


//------------------FULL MCMC------------------//
__global__ void mcmc(int shared, int n, int n_samples, int n_samples_per_thread, int n_threads, int n_steps, int spacing, float mu_step, float cov_step, curandState* state, float* curr_cov, float* new_cov, float* curr_mu, float* new_mu, float* rand_num, int* rand_ints, float* rand_mu, float* rand_cov, float* curr_L, int* take_step, float* data, float* inv_mu, float* inv_cov, float* est_mu, float* est_cov, float* all_est_mu) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    float initial_mean = 2.0f; // Farily arbitrary selection here - should prob check this an okay value
    float initial_cov = 0.5f;
    int step_count = 0;
    int local_take_step, estimation_offset;
    // Initialise shared memory
    extern __shared__ float s_data[];

    // Initialise covariance and mean arrays
    init_array(initial_cov, curr_cov, n);
    init_array(initial_mean, curr_mu, n);
    cuda_rand_init(state);

    // If using shared memory use s_data
    // Note: Requires very small sample size for this to work without exceeding shared memory capacity
    if (shared == 1) {
        // Partition dataset according to threads
        int copy_offset = threadID * n * n_samples_per_thread; // Starting index
        int copy_end = copy_offset + n_samples_per_thread * n;
        for (int i = copy_offset; i < copy_end; i++) {
            s_data[i] = data[i];
        }
        // Get the current log likelihood using shared data
        curr_L[threadID] = get_total_log_likelihood(curr_cov, curr_mu, s_data, n_samples_per_thread, n_samples, n);
    }
    else {
        // Get the current log likelihood using global data
        curr_L[threadID] = get_total_log_likelihood(curr_cov, curr_mu, data, n_samples_per_thread, n_samples, n);
    }

    // Copy current vectors (covariance and mu) to new vectors
    vec_cpy(curr_cov, new_cov, n);
    vec_cpy(curr_mu, new_mu, n);

    while (step_count < n_steps) {
        // Generate random numbers for rand_num, rand_ints (both from uniform dist) and rand_mu and rand_cov (both from normal dist)
        generate_random_nums(state, rand_mu, rand_cov, mu_step, cov_step, rand_num, rand_ints, n);
        // Propose new sample for new mu and cov by perturbating (add small random noise) the most recent sample
        perturb_params(curr_cov, curr_mu, new_cov, new_mu, rand_ints, rand_mu, rand_cov, n);
        // Accept or reject new proposal as the new sample (if rejected, old sample is retained).
        if (shared == 1) { // If using shared, compute the step using shared data, otherwise use the global data
            mcmc_step(curr_L, new_cov, new_mu, curr_cov, curr_mu, rand_ints, rand_mu, rand_cov, rand_num, n, n_samples_per_thread, n_samples, take_step, s_data);
        }
        else {
            mcmc_step(curr_L, new_cov, new_mu, curr_cov, curr_mu, rand_ints, rand_mu, rand_cov, rand_num, n, n_samples_per_thread, n_samples, take_step, data);
        }
        local_take_step = take_step[threadID];
        // If individual thread took a step, copy the new vectors (covariance and mu) to the current vectors
        if (local_take_step == 1) {
            vec_cpy(new_cov, curr_cov, n);
            vec_cpy(new_mu, curr_mu, n);
        }
        // If at evaluation step, calculate the estimated covariance and mu
        if (step_count % spacing == 0) {
            calc_cov(curr_cov, inv_cov, est_cov, n, n_threads);
            calc_mu(curr_mu, inv_cov, inv_mu, est_cov, est_mu, n, n_threads);
            // If first thread, combine the results
            if (threadID == 0) {
                estimation_offset = step_count / spacing * n; // Error term
                for (int i = estimation_offset; i < estimation_offset + n; i++) {
                    all_est_mu[i] = est_mu[i - estimation_offset];
                }
            }
            __syncthreads(); // Synchronise threads within the same block
        }
        step_count += 1;
    }
}

//-------------MAIN---------------//
int main(int argc, char* argv[]) {
    // Process user inputs and parse arguments (could add in some logic here to prevent invalid inputs that would cause program to crash)
    int* args = parse_args(argc, argv);
    int n_steps = args[0];
    int n = args[1];
    int spacing = args[2];
    int shared = args[3];
    int n_threads = args[4];
    int n_blocks = args[5];    
    int n_samples = count_n_samples(n); // Note: n_samples for each of the n dimensions
    int n_samples_per_thread = n_samples / n_threads;
    int n_sample_points = n_steps / spacing;
    float mu_step = 0.2;
    float cov_step = 0.2;
    print_params(n_steps, n, spacing, shared, n_threads, n_blocks, n_samples, n_samples_per_thread);

    // Declare variables
    cudaEvent_t start, stop;
    curandState* state;
    float *rand_num, *rand_mu, *curr_mu, *new_mu, *inv_mu, *est_mu, *all_est_mu, *rand_cov, *curr_cov, *new_cov, *inv_cov, *est_cov, *curr_L, *data;
    int *rand_ints, *take_step;

    // Allocate memory for GPU (device) variables
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    gpuErrChk(cudaMalloc(&state, n_blocks * n_threads * sizeof(curandState)));
    gpuErrChk(cudaMalloc((void**)&rand_num, n_blocks * n_threads * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&rand_mu, n_blocks * n_threads * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&curr_mu, n * n_threads * n_blocks * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&new_mu, n * n_threads * n_blocks * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&inv_mu, n * n_blocks * n_threads * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&est_mu, n * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&all_est_mu, n * n_sample_points * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&rand_cov, n_blocks * n_threads * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&curr_cov, n * n_threads * n_blocks * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&new_cov, n * n_threads * n_blocks * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&inv_cov, n * n_threads * n_blocks * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&est_cov, n * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&curr_L, n_threads * n_blocks * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&data, n * n_samples * sizeof(float)));
    gpuErrChk(cudaMalloc((void**)&rand_ints, n_blocks * n_threads * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&take_step, n_threads * n_blocks * sizeof(int)));

    // Allocate memory for CPU (host) variables
    float* h_data = (float*)malloc(n * n_samples * sizeof(float));
    float* h_cov = (float*)malloc(n * n_threads * n_blocks * sizeof(float));
    float* h_mu = (float*)malloc(n * n_threads * n_blocks * sizeof(float));
    float* h_rand = (float*)malloc(n_threads * n_blocks * sizeof(float));
    float* h_est_mu = (float*)malloc(n * n_sample_points * sizeof(float));

    // Read the samples into h_data
    read_samples(n_samples, n, h_data);

    // Copy all of the samples from the CPU to the GPU
    gpuErrChk(cudaMemcpy(data, h_data, n * n_samples * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid and block dimension
    int grid_dim = n_blocks; // Default to 1D structures - could provide an option to use more complicated grid dimensions which could boost performance
    int block_dim = n_threads / n_blocks;
    printf("Grid dim: %d, Block dim: %d\n", grid_dim, block_dim);

    if (shared == 1) {
        cudaEventRecord(start);
        // Usage: mykernel<<<grid_dim, block_dim, shared_mem>>>(args);
        // NOTE: shared_mcmc only works for very small data sets
        mcmc<<<grid_dim, block_dim, n * n_samples * sizeof(float) >>>(shared, n, n_samples, n_samples_per_thread, n_threads, n_steps, spacing, mu_step, cov_step, state, curr_cov, new_cov, curr_mu, new_mu, rand_num, rand_ints, rand_mu, rand_cov, curr_L, take_step, data, inv_mu, inv_cov, est_mu, est_cov, all_est_mu);
        cudaEventRecord(stop);
        gpuErrChk(cudaDeviceSynchronize()); // Error check for shared memory sync
    }
    else {
        cudaEventRecord(start);
        mcmc<<<grid_dim, block_dim>>>(shared, n, n_samples, n_samples_per_thread, n_threads, n_steps, spacing, mu_step, cov_step, state, curr_cov, new_cov, curr_mu, new_mu, rand_num, rand_ints, rand_mu, rand_cov, curr_L, take_step, data, inv_mu, inv_cov, est_mu, est_cov, all_est_mu);
        cudaEventRecord(stop);
        gpuErrChk(cudaDeviceSynchronize()); // Error check for shared memory sync
    }

    // Timing results
    float millis = 0.f;
    cudaEventElapsedTime(&millis, start, stop);
    printf("Code executed in %f ms.\n", millis);

    // Copy from GPU to CPU
    gpuErrChk(cudaMemcpy(h_cov, curr_cov, n * n_threads * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_mu, curr_mu, n * n_threads * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_est_mu, all_est_mu, n * n_sample_points * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrChk(cudaMemcpy(h_rand, rand_num, n_threads * sizeof(float), cudaMemcpyDeviceToHost));

    // Output results to file
    output_results(h_mu, h_cov, h_est_mu, h_rand, n, n_threads, n_sample_points, millis);
    
    // Deallocate CUDA memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(state);
    cudaFree(rand_num);
    cudaFree(rand_mu);
    cudaFree(curr_mu);
    cudaFree(new_mu);
    cudaFree(inv_mu);
    cudaFree(est_mu);
    cudaFree(all_est_mu);
    cudaFree(rand_cov);
    cudaFree(curr_cov);
    cudaFree(new_cov);
    cudaFree(inv_cov);
    cudaFree(est_cov);
    cudaFree(curr_L);
    cudaFree(data);
    cudaFree(rand_ints);
    cudaFree(take_step);

    // Dellocate CPU memory
    free(h_data);
    free(h_cov);
    free(h_mu);
    free(h_rand);
    free(h_est_mu);

    return 0;
}
