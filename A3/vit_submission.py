from typing import Tuple
import datetime

import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class Args:
    """Command-line arguments to store model configuration."""

    num_classes = 10

    # Hyperparameters
    epochs = 10  # Should easily reach above 65% test acc after 20 epochs with an hidden_size of 64
    batch_size = 64
    lr = 0.0003
    weight_decay = 0.00001

    # Hyperparameters for ViT
    input_resolution = 32
    in_channels = 3
    patch_size = 4
    hidden_size = 64
    layers = 6
    heads = 8

    YOUR_CCID = "amaralpe"
    name = f"vit-cifar10-{YOUR_CCID}"


class PatchEmbeddings(nn.Module):
    """(0.5 out of 10) Compute patch embedding
    of shape `(batch_size, seq_length, hidden_size)`.
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        hidden_size: int,
        in_channels: int = 3,  # 3 for RGB, 1 for Grayscale
    ):
        super().__init__()

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x is of shape (batch_size, in_channels, input_resolution, input_resolution)

        # Apply the convolution to get patch embeddings
        x = self.projection(x)  # Now shape is (batch_size, hidden_size, num_patches, num_patches)

        # Flatten the last two dimensions (num_patches, num_patches) to get (batch_size, hidden_size, seq_length)
        embeddings = x.flatten(2)  # Shape becomes (batch_size, hidden_size, seq_length)

        # Transpose to (batch_size, seq_length, hidden_size) as expected
        embeddings = embeddings.transpose(1, 2)  # Shape becomes (batch_size, seq_length, hidden_size)

        return embeddings


class PositionEmbedding(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
    ):
        """TODO: (0.5 out of 10) Given patch embeddings,
        calculate position embeddings with [CLS] and [POS].
        """
        super().__init__()
        # Create a learnable CLS token (shape: 1, 1, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Create learnable position embeddings (shape: 1, num_patches + 1, hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        
        # Initialize parameters
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings shape: (batch_size, num_patches, hidden_size)
        batch_size = embeddings.shape[0]

        # Expand the cls_token to match the batch size (shape: batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Concatenate CLS token at the beginning of the patch embeddings (along sequence dimension)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # Shape: (batch_size, num_patches + 1, hidden_size)
        
        # Add position embeddings (shape: batch_size, num_patches + 1, hidden_size)
        embeddings = embeddings + self.position_embeddings
        
        return embeddings


class TransformerEncoderBlock(nn.Module):
    """TODO: (0.5 out of 10) A residual Transformer encoder block."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=True)
        
        # Layer Normalization before attention
        self.ln_1 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # Expand the hidden size
            nn.GELU(),                        # GELU activation
            nn.Linear(d_model * 4, d_model),  # Project back to d_model
        )
        
        # Layer Normalization before MLP
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x is of shape (batch_size, seq_length, d_model)

        # Self-Attention with residual connection
        attn_out, _ = self.attn(x, x, x)  # Self-attention (Q=K=V=x)
        x = x + attn_out  # Residual connection
        x = self.ln_1(x)  # Apply layer normalization

        # MLP with residual connection
        mlp_out = self.mlp(x)
        x = x + mlp_out  # Residual connection
        x = self.ln_2(x)  # Apply layer normalization

        return x


class ViT(nn.Module):
    """TODO: (0.5 out of 10) Vision Transformer."""

    def __init__(
        self,
        num_classes: int,
        input_resolution: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # Patch Embedding layer
        num_patches = (input_resolution // patch_size) ** 2  # Total number of patches
        self.patch_embed = PatchEmbeddings(
            input_resolution=input_resolution,
            patch_size=patch_size,
            hidden_size=hidden_size,
            in_channels=in_channels,
        )
        
        # Position Embedding layer
        self.pos_embed = PositionEmbedding(
            num_patches=num_patches,
            hidden_size=hidden_size
        )

        # Pre-transformer layer normalization
        self.ln_pre = nn.LayerNorm(hidden_size)
        
        # Transformer Encoder layers
        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(d_model=hidden_size, n_head=heads)
            for _ in range(layers)
        ])

        # Post-transformer layer normalization
        self.ln_post = nn.LayerNorm(hidden_size)

        # Classifier head to map from hidden size to number of classes
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        # Compute patch embeddings
        x = self.patch_embed(x)  
        
        # Add position embeddings (with the CLS token)
        x = self.pos_embed(x)  
        
        # Layer normalization before transformer
        x = self.ln_pre(x) 
        
        # Pass through transformer encoder blocks
        for layer in self.transformer:
            x = layer(x) 
        
        # Layer normalization after transformer
        x = self.ln_post(x)

        # Extract the CLS token (it's the first token in the sequence)
        cls_token = x[:, 0] 
        
        # Classifier head
        logits = self.classifier(cls_token) 

        return logits


def transform(
    input_resolution: int,
    mode: str = "train",
    mean: Tuple[float] = (0.5, 0.5, 0.5),  
    std: Tuple[float] = (0.5, 0.5, 0.5), 
):
    """TODO: (0.25 out of 10) Preprocess the image inputs
    with at least 3 data augmentation for training.
    """
    if mode == "train":
        # Training data augmentation and normalization
        tfm = transforms.Compose([
            transforms.RandomResizedCrop(input_resolution, scale=(0.8, 1.0)),  # Random resizing and cropping
            transforms.RandomHorizontalFlip(),  # Randomly flip image horizontally
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter as an augmentation
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize(mean=mean, std=std),  # Normalize with mean and std
        ])
    
    else:
        # Validation/Testing normalization (no augmentation)
        tfm = transforms.Compose([
            transforms.Resize(input_resolution),  # Resize to the input resolution
            transforms.CenterCrop(input_resolution),  # Center crop
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=mean, std=std),  # Normalize with mean and std
        ])

    return tfm



def inverse_transform(
    img_tensor: torch.Tensor,
    mean: Tuple[float] = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5,),  
    std: Tuple[float] = (1 / 0.5, 1 / 0.5, 1 / 0.5),  
) -> np.ndarray:
    """Given a preprocessed image tensor, revert the normalization process and
    convert the tensor back to a numpy image.
    """
    # Inverse normalization
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)], 
        std=[1.0 / s for s in std]
    )
    
    # Apply inverse normalization to the image tensor
    img_tensor = inv_normalize(img_tensor)
    
    # Rearrange the tensor to (H, W, C) format for visualization
    img_tensor = img_tensor.permute(1, 2, 0).clamp(0, 1)  
    
    # Convert to numpy array and scale to [0, 255]
    img = np.uint8(255 * img_tensor.numpy())

    return img


def train_vit_model(args):
    """ (0.25 out of 10) Train loop for ViT model."""

    tfm_train = transform(
        input_resolution=args.input_resolution,
        mode="train",
    )

    tfm_test = transform(
        input_resolution=args.input_resolution,
        mode="test",
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=tfm_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=tfm_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = ViT(
        num_classes=args.num_classes,
        input_resolution=args.input_resolution,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads
    )
    print(model)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} / {args.epochs}")

        for i, (x, labels) in enumerate(pbar):
            model.train()
            # Move data to GPU if available
            if torch.cuda.is_available():
                x, labels = x.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Show train loss at the end of epoch
            pbar.set_postfix({"loss": "{:.4f}".format(loss.item())})

        # Step the scheduler
        scheduler.step()

        # Evaluate at the end
        test_acc = test_classification_model(model, test_loader)

        # NOTE: DO NOT CHANGE
        # Save the model
        if test_acc > best_acc:
            best_acc = test_acc
            state_dict = {
                "model": model.state_dict(),
                "acc": best_acc,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(state_dict, "{}.pt".format(args.name))
            print("Best test acc:", best_acc)
        else:
            print("Test acc:", test_acc)
        print()


def test_classification_model(
    model: nn.Module,
    test_loader,
):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
