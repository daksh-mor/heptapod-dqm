"""
GSoC 2025 ContinuousML4DQM

This script can be used to train an instance of a DepthViT model
"""

import argparse
import os, sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import concurrent.futures
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURE_DATASET_DIR = os.path.join(PROJECT_ROOT, "tools", "dqm", "test_files", "dataset")
LEGACY_RESULTS_DIR = os.path.join(PROJECT_ROOT, "sandbox", "legacy_model_training")

from model_datasets import * 
from models_spatial import * 

# HCAL mask
subdetector_mask = np.load(os.path.join(FIXTURE_DATASET_DIR, 'he_segmentation_config_mask.npy')) 

#####################
### Load Datasets ###
#####################

# class MinMaxScalerCustom:
    
#     def __init__(self, feature_range=(0, 1), min_value=None, max_value=None):
#         self.feature_range = feature_range
#         self.min = min_value  # Set manually if provided
#         self.max = max_value  # Set manually if provided

#     def _get_numpy(self, data):
#         if isinstance(data, pd.DataFrame):
#             data = data.values
#         return data

#     def fit(self, data):
#         if self.min is None or self.max is None:
#             raise ValueError("Min and Max values must be set manually for this scaler.")

#     def transform(self, data):
#         data = self._get_numpy(data)
#         scale_min, scale_max = self.feature_range
#         # Apply the custom min and max values for scaling
#         scaled_data = (data - self.min) / (self.max - self.min)
#         scaled_data = scaled_data * (scale_max - scale_min) + scale_min
#         return scaled_data

#     def fit_transform(self, data):
#         self.fit(data)
#         return self.transform(data)

#     def inverse_transform(self, data):
#         data = self._get_numpy(data)
#         scale_min, scale_max = self.feature_range
#         # Reverse scaling to get back to original values using manual min and max
#         original_data = (data - scale_min) / (scale_max - scale_min)
#         original_data = original_data * (self.max - self.min) + self.min
#         return original_data

class QuantileScalerCustom:
    
    def __init__(self):
        pass

    def _get_numpy(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        return data

    def transform(self, data, mask):
        data = self._get_numpy(data)
        data = np.squeeze(data, axis=-1) # (N, 64, 72, 7, 1) → (N, 64, 72, 7)
        masked_data = np.where(mask, data, np.nan) #Need to set zero values to nan so that they are not considered in the following operations
        masked_q1 = np.nanpercentile(masked_data, 25, axis=(1, 2, 3))
        masked_q3 = np.nanpercentile(masked_data, 75, axis=(1, 2, 3))
        scaled_data = (masked_data - masked_q1[:, None, None, None]) / (masked_q3[:, None, None, None] - masked_q1[:, None, None, None] + 1e-10)
        non_nan_data = np.nan_to_num(scaled_data, nan=0.0) #Go from nan back to zeros in the numpy arrays
        return non_nan_data

    def inverse_transform(self, data):
        #No inverse_transform defined at this time, as it isnt necessary for anomaly
        #detection and would be difficult to implement
        pass

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, file_path, max_value=2567.0):
        data = np.load(file_path)  # Assuming key is 'data'

        # For MinMaxScaler
        scaler = MinMaxScalerCustom(feature_range=(0, 1), min_value=0.0, max_value=max_value)
        data = scaler.transform(data)

        # # For QuantileScaler
        # scaler = QuantileScalerCustom()
        # data = scaler.transform(data, mask=subdetector_mask)

        # Add an extra channel dimensions to agree with DepthViT architecture
        data = np.expand_dims(data, axis=1) # (N, 64, 72, 7) → (N, 1, 64, 72, 7)
        # data = np.expand_dims(data, axis=-1) # (N, 1, 64, 72, 7) → (N, 1, 64, 72, 7, 1)

        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def shape(self):
        return self.data.shape  # Allows accessing shape directly

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

# RunIDs
runid = [323940, 323997, 324021, 324022, 325117, 325170] #2018 Runs
# runid = [355456, 355680, 355769, 356381, 357081, 357112, 357442, 357479, 357612, 357815, 357899, 359694, 359764, 360019, 360459, 360820, 361240, 361957, 362091, 362760] #2022 Runs

for run in runid:

    print("********************")
    print(f"    run: {run}     ")
    print("********************")

    train_data_path = os.path.join(FIXTURE_DATASET_DIR, "train_data.npy")
    test_data_path = os.path.join(FIXTURE_DATASET_DIR, "test_data.npy")

    if not os.path.exists(train_data_path):
        print(f"Train data not found for {run}, skipping.")
        continue
    if not os.path.exists(test_data_path):
        print(f"Train data not found for {run}, skipping.")
        continue

    # Load datasets
    train_dataset = CustomDataset(train_data_path)
    test_dataset = CustomDataset(test_data_path)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #########################
    ### Define Parameters ###
    #########################

    early_stop_epoch = 10
    learning_rate = 1e-3
    vae_reg_beta = 1e-6
    weight_decay = 1e-7
    valid_size = 0.20  # 20% validation split
    shuffle_data = False

    feature_dim = 1  # Matches last dimension of input
    latent_dim = 56  # Matches VAE latent size
    target_dim = 1  # Should match input feature size
    spatial_dims = (64, 72, 7)  # Matches spatial size from input

    # Convolutional Hyperparameters
    e_num_conv_layers = 4  # Number of convolutional layers
    kernel_size = (3, 3, 2)  # Matches encoder CNN settings
    pool_size = (2, 2, 2)  # Pooling layers in encoder
    activation = "leakyrelu"
    norm_layer = "bn"  # Batch normalization
    use_res = False  # No residual connections

    memorysize = 1
    image_size=[64,72]
    k_factor=8
    num_layers=5
    mask_ratio=0.5
    patch_size=12

    # Variational Autoencoder settings
    isvariational = True  # Enable VAE

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthwiseCrossViTAE_MultiDim_SPATIAL(
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        target_dim=target_dim,
        spatial_dims=spatial_dims,
        memorysize=memorysize,
        image_size=image_size,
        k_factor=k_factor,
        num_layers=num_layers,
        mask_ratio=mask_ratio,
        patch_size=patch_size,
        isvariational=isvariational  # Ensures VAE is used
    ).to(device)
    print("Model hyperparameters...")
    print(model)

    #################
    ### Criterion ###
    #################

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay, amsgrad=True)

    ###################
    ### Train Model ###
    ###################

    # Initialize lists to store loss values
    train_losses = []
    test_losses = []
    num_epochs = 100

    # Early stopping parameters. This way ther needs to be no improvement for {patience} epochs before early stopping.
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training and Evaluation in the Same Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Forward pass
            if isvariational:
                output, mu, logvar = model(batch)
                recon_loss = criterion(output, batch)
                kl_loss = kl_divergence(mu, logvar).mean()
                loss = recon_loss + vae_reg_beta * kl_loss
            else:
                output = model(batch)
                loss = criterion(output, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluation
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                if isvariational:
                    output, mu, logvar = model(batch)
                    recon_loss = criterion(output, batch)
                    kl_loss = kl_divergence(mu, logvar).mean()
                    loss = recon_loss + vae_reg_beta * kl_loss
                else:
                    output = model(batch)
                    loss = criterion(output, batch)

                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Early stopping logic
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in test loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        ##################
        ### Save Model ###
        ##################

        save_folder = os.path.join(LEGACY_RESULTS_DIR, f"he_train_dataset_{run}")

        if not os.path.exists(save_folder):
                    os.makedirs(save_folder)

        # Save Model
        torch.save(model.state_dict(), os.path.join(save_folder, f"{run}_MODEL.pth"))

        #########################
        ### Visualize Results ###
        #########################

        # Plot Train and Test Loss
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss", marker="o")
        plt.plot(test_losses, label="Test Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Train vs Test Loss Over Epochs Run {run}")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(save_folder, f"{run}_results.pdf"))


