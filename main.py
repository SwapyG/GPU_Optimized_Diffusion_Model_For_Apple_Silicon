# Filename: main.py

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import our Apple-specific modules
from basic_diffusion import SimpleUNet, DiffusionModel, train_diffusion_model
from metal_diffusion import MetalDiffusionModel
from coreml_integration import CoreMLDiffusionModel
# The benchmark function would need a new home or rewrite, so we'll comment it out for now
# from tensorrt_integration import benchmark_all_methods (This file is deleted)


def parse_args():
    parser = argparse.ArgumentParser(description="Metal-Optimized Diffusion Model for Apple Silicon")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], # Removed benchmark for simplicity
                        help="Mode to run: train or sample")
    parser.add_argument("--model_path", type=str, default="diffusion_model.pt",
                        help="Path to load/save the model")
    # ... (rest of the arguments are the same)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training or sampling")
    parser.add_argument("--img_size", type=int, default=28, help="Image size (assumed square)")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples to generate")
    # Updated choices for Metal-based implementations
    parser.add_argument("--implementation", type=str, default="original",
                        choices=["original", "metal", "coreml"],
                        help="Implementation to use for sampling")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10"],
                        help="Dataset to use for training")
    return parser.parse_args()


def main():
    args = parse_args()
    # --- Key Change: Set device to MPS (Metal Performance Shaders) ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) device.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Using CPU.")


    # Set up dataset (this logic remains the same)
    if args.dataset == "mnist":
        dataset_class = datasets.MNIST
        mean, std = (0.5,), (0.5,)
    elif args.dataset == "fashion_mnist":
        dataset_class = datasets.FashionMNIST
        mean, std = (0.5,), (0.5,)
    elif args.dataset == "cifar10":
        dataset_class = datasets.CIFAR10
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        args.channels = 3
    else:
        raise ValueError("Unsupported dataset")

    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    os.makedirs("samples", exist_ok=True)

    if args.mode == "train":
        dataset = dataset_class(root='./data', train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model = SimpleUNet(in_channels=args.channels, out_channels=args.channels, time_dim=256).to(device)
        diffusion = DiffusionModel(model, n_steps=args.n_steps, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        print(f"Training diffusion model on '{args.dataset}' for {args.n_epochs} epochs...")
        train_diffusion_model(diffusion, dataloader, optimizer, args.n_epochs, device)

        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

    elif args.mode == "sample":
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}. Please train a model first with --mode train.")

        # Load the base model structure
        model = SimpleUNet(in_channels=args.channels, out_channels=args.channels, time_dim=256)
        # Load the trained weights onto the CPU first, then move the model to the device.
        # This is safer than loading directly onto the MPS device.
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.to(device)

        # Select the correct implementation based on user choice
        if args.implementation == "original":
            diffusion = DiffusionModel(model, n_steps=args.n_steps, device=device)
            print("Using original PyTorch on MPS implementation.")
        elif args.implementation == "metal":
            diffusion = MetalDiffusionModel(model, n_steps=args.n_steps, device=device)
            print("Using custom Metal kernel implementation.")
        elif args.implementation == "coreml":
            diffusion = CoreMLDiffusionModel(args.model_path, n_steps=args.n_steps, device=device,
                                           img_size=args.img_size, channels=args.channels,
                                           batch_size=args.n_samples)
            print("Using Core ML implementation.")

        print(f"Generating {args.n_samples} samples with '{args.implementation}' implementation...")
        samples = diffusion.sample(args.n_samples, args.img_size, args.channels)

        output_file = f"samples/{args.dataset}_{args.implementation}_samples.png"
        save_samples(samples, args.channels, output_file)
        print(f"Samples saved to {output_file}")


def save_samples(samples, channels, filename):
    """Save generated samples as an image grid (no changes needed here)."""
    samples = (samples.cpu().detach() + 1) / 2
    samples = np.clip(samples, 0, 1)

    grid_size_h = int(np.sqrt(samples.shape[0]))
    grid_size_w = (samples.shape[0] + grid_size_h - 1) // grid_size_h

    fig, axes = plt.subplots(grid_size_h, grid_size_w, figsize=(grid_size_w * 2, grid_size_h * 2))
    axes = axes.flatten()

    for i in range(samples.shape[0]):
        if channels == 1:
            axes[i].imshow(samples[i, 0], cmap='gray')
        else:
            axes[i].imshow(np.transpose(samples[i], (1, 2, 0)))
        axes[i].axis('off')

    for i in range(samples.shape[0], len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()