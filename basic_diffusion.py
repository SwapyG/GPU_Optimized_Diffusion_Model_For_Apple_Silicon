import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class SimpleUNet(nn.Module):
    """A simplified U-Net architecture for the diffusion model."""

    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)  # Downsample
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)  # Downsample

        # Bottleneck
        self.bottleneck1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bottleneck2 = nn.Conv2d(256, 256, 3, padding=1)

        # Time embedding projection for bottleneck
        self.time_proj = nn.Linear(time_dim, 256)

        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # Upsample
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # Upsample
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)

        # Encoder
        x1 = F.silu(self.conv1(x))
        x2 = F.silu(self.conv2(x1))
        x3 = F.silu(self.conv3(x2))

        # Bottleneck with time embedding added
        time_emb = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        x3 = x3 + time_emb
        x3 = F.silu(self.bottleneck1(x3))
        x3 = F.silu(self.bottleneck2(x3))

        # Decoder
        x = F.silu(self.up1(x3))
        x = F.silu(self.conv4(x))
        x = F.silu(self.up2(x))
        x = F.silu(self.conv5(x))
        x = self.conv6(x)

        return x


class DiffusionModel:
    """Implementation of the diffusion process for training and sampling."""

    def __init__(self, model, n_steps=1000, beta_min=1e-4, beta_max=0.02, device="cuda"):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device

        # Define noise schedule (linear beta schedule)
        self.betas = torch.linspace(beta_min, beta_max, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0, t):
        """Add noise to the input image according to the diffusion schedule at time t."""
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().reshape(-1, 1, 1, 1)

        # q(x_t | x_0) = sqrt(alphas_cumprod) * x_0 + sqrt(1 - alphas_cumprod) * noise
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def training_loss(self, x_0, t):
        """Compute the training loss for the model."""
        x_noisy, noise_added = self.add_noise(x_0, t)

        # Predict the noise that was added
        noise_pred = self.model(x_noisy, t)

        # Loss is the mean squared error between the added noise and predicted noise
        return F.mse_loss(noise_pred, noise_added)

    def sample(self, n_samples, img_size, channels=1):
        """Generate new images using the trained diffusion model."""
        self.model.eval()

        # Start with random noise
        x = torch.randn(n_samples, channels, img_size, img_size).to(self.device)

        # Gradually denoise the image
        for t in range(self.n_steps - 1, -1, -1):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(x, t_batch)

            # Compute parameters for denoising step
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * noise_pred) + sigma_t * noise
            x = (1 / alpha_t.sqrt()) * (
                    x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * noise_pred
            ) + beta_t.sqrt() * noise

        self.model.train()
        return x


def train_diffusion_model(diffusion, dataloader, optimizer, n_epochs, device="cuda"):
    """Train the diffusion model."""
    for epoch in range(n_epochs):
        total_loss = 0

        for batch in dataloader:
            x, _ = batch
            x = x.to(device)

            # Sample random timesteps
            t = torch.randint(0, diffusion.n_steps, (x.shape[0],), device=device)

            optimizer.zero_grad()
            loss = diffusion.training_loss(x, t)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


# Example usage with MNIST
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 64
    img_size = 28
    channels = 1
    n_epochs = 5

    # Data loading
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    model = SimpleUNet(in_channels=channels, out_channels=channels, time_dim=256)
    diffusion = DiffusionModel(model, n_steps=1000, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_diffusion_model(diffusion, dataloader, optimizer, n_epochs, device)

    # Generate samples
    samples = diffusion.sample(10, img_size, channels)

    # Display samples
    samples = samples.cpu().detach()
    samples = (samples + 1) / 2  # Denormalize
    grid = torch.cat([sample for sample in samples], dim=2)
    plt.imshow(grid[0].numpy(), cmap='gray')
    plt.savefig('diffusion_samples.png')
    plt.show()

    # Save model
    torch.save(model.state_dict(), 'diffusion_model.pt')
    print("Model saved!")


if __name__ == "__main__":
    main()