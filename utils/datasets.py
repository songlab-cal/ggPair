from torch.utils.data import Dataset
import torch

# updated to accept and store the gene names.
class SparseDataset(Dataset):
    def __init__(self, sparse_matrix, gene_names, norm_size):
        self.sparse_matrix = sparse_matrix.tocsr()  # Convert to CSR format for efficient row slicing
        self.gene_names = gene_names
        self.norm_size = norm_size

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        # Retrieve a single row (gene) and reshape it
        dense_matrix = self.sparse_matrix.getrow(idx).toarray()
        gene_name = self.gene_names[idx] # Get the gene name for this row
        tensor_matrix = torch.tensor(
            dense_matrix.reshape(self.norm_size, self.norm_size), dtype=torch.float32)
        return tensor_matrix, gene_name

    
class NoisyDataset(Dataset):
    def __init__(self, original_dataset, noise_factor=1.0, mode='add'):
        self.original_dataset = original_dataset
        self.noise_factor = noise_factor
        self.mode = mode  # 'add' or 'multiply'

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data, gene_name = self.original_dataset[idx]
        if self.mode == 'add':
            noisy_data = self.add_noise_to_data(data, self.noise_factor)
        elif self.mode == 'multiply':
            noisy_data = self.multiply_noise_to_data(data, self.noise_factor)
        else:
            raise ValueError("Invalid mode. Choose 'add' or 'multiply'.")
        return noisy_data, gene_name

    def add_noise_to_data(self, data, noise_factor):
        if noise_factor != 0:
            noise = torch.randn_like(data) * noise_factor
            max_value = data.max()  # Ensure noise is capped by the max value of the original image 
            noisy_data = data + noise
            noisy_data = torch.clip(noisy_data, 0, max_value)
            return noisy_data
        else:
            return data

    def multiply_noise_to_data(self, data, noise_factor):
        if noise_factor != 0:
            noise = torch.randn_like(data) * noise_factor
            noisy_data = data * (1 + noise)
            # Optional: Clip to ensure non-negativity or other bounds as necessary
            noisy_data = torch.clip(noisy_data, 0, data.max())
            return noisy_data
        else:
            return data
