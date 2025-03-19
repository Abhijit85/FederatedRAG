import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import RelatEModel  # Updated to match the FediRAG document
import numpy as np
import random
from cryptography.fernet import Fernet
from dataloader import load_data  # Handles data loading
from utils import secure_aggregation  # Secure aggregation function

# Load LLama3 model for dense text encoding
llama_model_name = "meta-llama/Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

# Generate a key for encryption/decryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

class FediRAGClient:
    def __init__(self, model, data, client_id, num_clients):
        self.model = model
        self.data = data
        self.client_id = client_id
        self.num_clients = num_clients
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_local_model(self, epochs=5):
        """Local training using RelatE model"""
        self.model.train()
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            inputs, targets = self.data  # Assume data is preprocessed
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        
        return self.get_model_updates()

    def get_model_updates(self):
        """Extract model updates for federated learning"""
        updates = {name: param.data.clone() for name, param in self.model.named_parameters()}
        return updates

    def apply_secure_masking(self, updates):
        """Apply pairwise random masking for secure aggregation"""
        masked_updates = {}
        random_masks = {}
        for name, param in updates.items():
            random_mask = torch.randn_like(param)
            masked_updates[name] = param + random_mask
            random_masks[name] = random_mask
        
        return masked_updates, random_masks
    
    def add_differential_privacy(self, updates, noise_std=0.01):
        """Inject differential privacy noise into updates"""
        dp_updates = {}
        for name, param in updates.items():
            dp_updates[name] = param + torch.randn_like(param) * noise_std
        
        return dp_updates

    def encrypt_data(self, updates):
        """Encrypt model updates before sharing with the server"""
        encrypted_updates = {}
        for name, param in updates.items():
            encrypted_updates[name] = cipher_suite.encrypt(param.numpy().tobytes())
        
        return encrypted_updates

class FediRAGServer:
    def __init__(self, num_clients, global_model):
        self.num_clients = num_clients
        self.global_model = global_model
        self.aggregated_updates = {}
    
    def aggregate_updates(self, client_updates):
        """Aggregate updates securely from multiple clients"""
        return secure_aggregation(client_updates, self.num_clients)

    def decrypt_updates(self, encrypted_updates):
        """Decrypt received updates before aggregation"""
        decrypted_updates = {}
        for name, param_bytes in encrypted_updates.items():
            decrypted_updates[name] = torch.tensor(np.frombuffer(cipher_suite.decrypt(param_bytes), dtype=np.float32))
        
        return decrypted_updates
    
    def update_global_model(self, aggregated_updates):
        """Update the global model using aggregated updates"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_updates:
                    param.copy_(aggregated_updates[name])

# Initialize clients and server
num_clients = 5
data_splits = load_data(num_clients)  # Load and distribute dataset
clients = [FediRAGClient(RelatEModel(), data=data_splits[i], client_id=i, num_clients=num_clients) for i in range(num_clients)]
server = FediRAGServer(num_clients=num_clients, global_model=RelatEModel())

# Federated training process
client_updates = []
for client in clients:
    local_updates = client.train_local_model()
    masked_updates, _ = client.apply_secure_masking(local_updates)
    dp_updates = client.add_differential_privacy(masked_updates)
    encrypted_updates = client.encrypt_data(dp_updates)
    decrypted_updates = server.decrypt_updates(encrypted_updates)
    client_updates.append(decrypted_updates)

aggregated_updates = server.aggregate_updates(client_updates)
server.update_global_model(aggregated_updates)

print("Federated training completed, global model updated.")
