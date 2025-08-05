# Filename: metal_diffusion.py (Final Version with CPU Bridge Fix)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Import the custom Metal extension we built.
import diffusion_metal


class MetalDiffusionModel:
    """Diffusion model implementation using our custom Metal kernel for sampling."""

    def __init__(self, model, n_steps=1000, beta_min=1e-4, beta_max=0.02, device="mps"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample(self, n_samples, img_size, channels=1):
        """Generate new images using the trained diffusion model with our Metal optimization."""
        self.model.eval()
        x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)
        start_time = time.time()

        for t in range(self.n_steps - 1, -1, -1):
            if t % 100 == 0:
                print(f"Sampling step {self.n_steps - t}/{self.n_steps}, time elapsed: {time.time() - start_time:.2f}s")

            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                noise_pred = self.model(x, t_batch)

            alpha_t = self.alphas[t].item()
            alpha_cumprod_t = self.alphas_cumprod[t].item()
            beta_t = self.betas[t].item()
            add_noise_flag = t > 0

            if add_noise_flag:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # --- THE CPU BRIDGE FIX ---
            # 1. Move all input tensors to the CPU before calling the C++ function.
            x_cpu = x.cpu()
            noise_pred_cpu = noise_pred.cpu()
            noise_cpu = noise.cpu()

            # 2. Call our custom Metal function with the CPU tensors.
            output_cpu = diffusion_metal.reverse_diffusion_step(
                x_cpu, noise_pred_cpu, alpha_t, beta_t, alpha_cumprod_t, noise_cpu, add_noise_flag
            )

            # 3. Move the result tensor back to the MPS device for the next loop iteration.
            x = output_cpu.to(self.device)
            # --- END OF FIX ---

        total_time = time.time() - start_time
        print(f"Custom Metal kernel sampling completed in {total_time:.2f}s, {total_time / n_samples:.2f}s per image")

        self.model.train()
        return x