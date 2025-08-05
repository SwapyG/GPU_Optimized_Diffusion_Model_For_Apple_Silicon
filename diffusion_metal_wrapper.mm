// Filename: diffusion_metal_wrapper.mm

#include <torch/extension.h>
#include <Metal/Metal.h>
#include <vector>

// Embed the Metal kernel source directly into the C++ file for robust distribution.
const char* metal_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void reverse_diffusion_step_kernel(
    device const float* x_t [[buffer(0)]],
    device const float* noise_pred [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* noise [[buffer(3)]],
    constant const float& alpha_t [[buffer(4)]],
    constant const float& beta_t [[buffer(5)]],
    constant const bool& add_noise [[buffer(6)]],
    constant const float& one_minus_alpha_cumprod_t [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    float noise_scale = beta_t / sqrt(one_minus_alpha_cumprod_t);
    float alpha_scale = 1.0f / sqrt(alpha_t);
    float result = alpha_scale * (x_t[tid] - noise_scale * noise_pred[tid]);
    if (add_noise) {
        result += sqrt(beta_t) * noise[tid];
    }
    output[tid] = result;
}
)";

torch::Tensor reverse_diffusion_step_metal(
    torch::Tensor x, torch::Tensor noise_pred, float alpha_t, float beta_t,
    float alpha_cumprod_t, torch::Tensor noise, bool add_noise
) {
    // Wrap the entire function in an @autoreleasepool to ensure correct
    // memory management of temporary Metal objects.
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSError *error = nil;
        NSString *librarySource = [NSString stringWithUTF8String:metal_kernel_source];
        id<MTLLibrary> library = [device newLibraryWithSource:librarySource options:nil error:&error];
        if (!library) { throw std::runtime_error("Failed to compile Metal library."); }
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"reverse_diffusion_step_kernel"];
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) { throw std::runtime_error("Failed to create pipeline state."); }

        // Create INPUT buffers by copying data from PyTorch tensors.
        id<MTLBuffer> x_buffer = [device newBufferWithBytes:x.data_ptr() length:x.nbytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> noise_pred_buffer = [device newBufferWithBytes:noise_pred.data_ptr() length:noise_pred.nbytes() options:MTLResourceStorageModeShared];
        id<MTLBuffer> noise_buffer = [device newBufferWithBytes:noise.data_ptr() length:noise.nbytes() options:MTLResourceStorageModeShared];

        // Create a new, empty OUTPUT buffer managed entirely by Metal.
        id<MTLBuffer> output_buffer = [device newBufferWithLength:x.nbytes() options:MTLResourceStorageModeShared];

        float one_minus_alpha_cumprod_t = 1.0f - alpha_cumprod_t;
        id<MTLBuffer> alpha_t_buffer = [device newBufferWithBytes:&alpha_t length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> beta_t_buffer = [device newBufferWithBytes:&beta_t length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> add_noise_buffer = [device newBufferWithBytes:&add_noise length:sizeof(bool) options:MTLResourceStorageModeShared];
        id<MTLBuffer> one_minus_alpha_cumprod_t_buffer = [device newBufferWithBytes:&one_minus_alpha_cumprod_t length:sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
        [commandEncoder setComputePipelineState:pipelineState];
        [commandEncoder setBuffer:x_buffer offset:0 atIndex:0];
        [commandEncoder setBuffer:noise_pred_buffer offset:0 atIndex:1];
        [commandEncoder setBuffer:output_buffer offset:0 atIndex:2];
        [commandEncoder setBuffer:noise_buffer offset:0 atIndex:3];
        [commandEncoder setBuffer:alpha_t_buffer offset:0 atIndex:4];
        [commandEncoder setBuffer:beta_t_buffer offset:0 atIndex:5];
        [commandEncoder setBuffer:add_noise_buffer offset:0 atIndex:6];
        [commandEncoder setBuffer:one_minus_alpha_cumprod_t_buffer offset:0 atIndex:7];

        MTLSize gridSize = MTLSizeMake(x.numel(), 1, 1);
        NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
        if (threadGroupSize > x.numel()) { threadGroupSize = x.numel(); }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
        [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [commandEncoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Create a "blob" view of the Metal buffer's data.
        auto output_blob = torch::from_blob([output_buffer contents], x.sizes(), x.options());
        // .clone() performs a clean copy into a new, safely managed PyTorch tensor.
        return output_blob.clone();
    }
}

// Bind the C++ function to the Python module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reverse_diffusion_step", &reverse_diffusion_step_metal, "Reverse diffusion step (Metal/MPS)");
}