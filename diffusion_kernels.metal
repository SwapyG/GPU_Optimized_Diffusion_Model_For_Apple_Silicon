// Filename: diffusion_kernels.metal

#include <metal_stdlib>
using namespace metal;

// The 'kernel' keyword is the MSL equivalent of CUDA's '__global__'.
// It defines a function that can be launched from the CPU to run on the GPU.
kernel void reverse_diffusion_step_kernel(
    // 'device' indicates memory accessible by the GPU. 'const' means read-only.
    // '[[buffer(n)]]' is how we bind arguments from the CPU to the GPU kernel. Each has a unique index.
    device const float* x_t [[buffer(0)]],             // Input: Current noisy image (x_t)
    device const float* noise_pred [[buffer(1)]],      // Input: The model's predicted noise
    device float* output [[buffer(2)]],                // Output: The denoised image (x_{t-1})
    device const float* noise [[buffer(3)]],           // Input: Random noise for this step

    // 'constant' is for small, read-only data that's broadcast efficiently to all threads.
    constant const float& alpha_t [[buffer(4)]],       // Input: alpha_t parameter
    constant const float& beta_t [[buffer(5)]],        // Input: beta_t parameter
    constant const bool& add_noise [[buffer(6)]],      // Input: Flag to control noise addition
    constant const float& one_minus_alpha_cumprod_t [[buffer(7)]],

    // '[[thread_position_in_grid]]' is Metal's simple way to get a unique ID for each thread.
    // It's equivalent to the 'blockIdx.x * blockDim.x + threadIdx.x' calculation in CUDA.
    uint tid [[thread_position_in_grid]]
) {
    // The core mathematical logic remains identical to the CUDA version.
    // This is the "fused operation" that minimizes memory access and maximizes speed.

    // Pre-calculates the scaling factor for the predicted noise.
    float noise_scale = beta_t / sqrt(one_minus_alpha_cumprod_t);

    // Pre-calculates the scaling factor for the main term.
    float alpha_scale = 1.0f / sqrt(alpha_t);

    // Reads the values for the current element (tid), performs the math,
    // and stores the result in a local variable.
    float result = alpha_scale * (x_t[tid] - noise_scale * noise_pred[tid]);

    // If 'add_noise' is true (i.e., it's not the final denoising step, t > 0),
    // this adds the scaled random noise. 'sigma_t' is sqrt(beta_t).
    if (add_noise) {
        result += sqrt(beta_t) * noise[tid];
    }

    // The final computed value is written to the output tensor's memory location.
    output[tid] = result;
}