from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_data(data: List[List[int]]=[]):
    # Create your own dataset (e.g., random NumPy array)
    #your_data = np.random.rand(1000, 28 * 28).astype(np.float32)  # Example data
    
    # Initialize the model, loss function, and optimizer
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    if data == []:
    
        tensor_data = torch.tensor(data, dtype=torch.float32)

        dataset = TensorDataset(tensor_data, tensor_data)  # Inputs are the same as targets in autoencoder
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    else:
        # Later, you can load the model and continue training
        model = Autoencoder()
        model.load_state_dict(torch.load('autoencoder_model.pth'))
        print("Model loaded successfully!")

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for data, _ in train_loader:
            output = model(data)
            loss = criterion(output, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")


    # Save the trained model
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Model saved successfully!")