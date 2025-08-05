# Filename: coreml_integration.py (Final Corrected Version)

import torch
import coremltools as ct
import time
import numpy as np  # Import numpy for the fix
from basic_diffusion import SimpleUNet


class CoreMLDiffusionModel:
    def __init__(self, model_path, n_steps=1000, beta_min=1e-4, beta_max=0.02, device="mps",
                 img_size=28, channels=1, batch_size=4):
        self.device = torch.device(device)
        self.n_steps = n_steps

        # Correct model loading from previous fix
        orig_model = SimpleUNet(in_channels=channels, out_channels=channels, time_dim=256)
        state_dict = torch.load(model_path, map_location='cpu')
        orig_model.load_state_dict(state_dict)
        orig_model.to(self.device)
        orig_model.eval()
        self.orig_model = orig_model

        # Define noise schedule
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(self.device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Core ML Conversion
        print("Converting PyTorch model to Core ML... (This may take a moment)")
        example_x = torch.rand(batch_size, channels, img_size, img_size).to(self.device)
        example_t = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        traced_model = torch.jit.trace(self.orig_model, (example_x, example_t))
        inputs = [
            ct.TensorType(name="x", shape=example_x.shape),
            ct.TensorType(name="t", shape=example_t.shape)
        ]
        self.coreml_model = ct.convert(
            traced_model,
            inputs=inputs,
            compute_units=ct.ComputeUnit.ALL
        )
        print("Core ML conversion complete!")

    def sample(self, n_samples, img_size, channels=1):
        """Generate images using the Core ML model."""
        x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)
        start_time = time.time()

        for t in range(self.n_steps - 1, -1, -1):
            if t % 100 == 0:
                print(f"Sampling step {self.n_steps - t}/{self.n_steps}, time elapsed: {time.time() - start_time:.2f}s")

            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)

            x_numpy = x.cpu().numpy()
            t_batch_numpy = t_batch.cpu().numpy()

            # --- THE FINAL FIX ---
            # Explicitly cast the integer timestep array to float32, as expected by Core ML.
            t_batch_numpy = t_batch_numpy.astype(np.float32)
            # --- END OF FIX ---

            coreml_inputs = {"x": x_numpy, "t": t_batch_numpy}
            coreml_output_dict = self.coreml_model.predict(coreml_inputs)
            output_key = list(coreml_output_dict.keys())[0]
            noise_pred = torch.from_numpy(coreml_output_dict[output_key]).to(self.device)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / alpha_t.sqrt()) * (
                        x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * noise_pred) + beta_t.sqrt() * noise

        total_time = time.time() - start_time
        print(f"Core ML sampling completed in {total_time:.2f}s, {total_time / n_samples:.2f}s per image")

        return x