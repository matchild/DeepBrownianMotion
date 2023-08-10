import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from stochastic.processes import FractionalBrownianMotion
from torch.utils.data import TensorDataset, DataLoader, Dataset




def generate_dataset(N_SIMS:int, trajectory_length:int):
  '''
  This function generates a dataset of fractional brownian motion trajectories.

  N_SIMS: number of simulations to be generated
  trajectory_length: number of steps in each trajectory
  '''
  # Create empty X and Y vectors
  X=np.zeros((N_SIMS,trajectory_length ,1)) 
  Y=np.zeros(N_SIMS)

  # Fill X and Y vectors with simulated fractional brownian motion trajectories, each with a different value for alpha.
  for i in range(N_SIMS):
    alpha=np.around(np.random.uniform(0.01, 1.8), 4)
    f = FractionalBrownianMotion(hurst=0.5*alpha, t=1)
    X[i]= f.sample(trajectory_length-1).reshape(trajectory_length, 1)
    Y[i]=alpha

    if i%10000 == 0:
      print('Simulations Generated:',i)

  Y=Y.reshape(-1,1)

  # Split train and test datasets (70/30)
  X_train=X[:int(N_SIMS*0.7)]
  Y_train=Y[:int(N_SIMS*0.7)]

  X_test=X[int(N_SIMS*0.7):]
  Y_test=Y[int(N_SIMS*0.7):]

  return X_train, Y_train, X_test, Y_test


def DatasetMAE(loader, model, device):
  '''
  This functions calculates the average Mean Absloute Error (MAE) on a dataset.

  loader: data loader corresponding to the dataset we want to evaluate the model on
  model: model to be evaluated
  device: device to run the model on (e.g. 'cuda')
  
  '''
  loss=0
  num_samples = 0
  model.eval()  # Set model to evaluation mode
  with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device) # move to device, e.g. GPU
          y = y.to(device=device)
          scores = model(x)

          loss+=(torch.abs(scores-y)).sum()
          num_samples += scores.size(0)

      print('Got an overall MAE of ', loss.item()/num_samples)
  return loss.item()/num_samples    #Return average loss


class SimulationDatset(Dataset):
  '''
  Fine tuned version of pytorch Dataset class. Trajectories are normalized and are converted into torch tensors.

  X: input trajectories of shape (N, trajectory length, 1)
  Y: vector of hurst parameters of shape (N, 1)
  '''

  def __init__(self, X, Y):

    # Scale features
    X=(X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True) #Normalize X
    Y=Y # No need to normalize y

    # Convert to torch tensors
    self.X=torch.tensor(X, dtype=torch.float32)
    self.Y=torch.tensor(Y, dtype=torch.float32)

  def __len__(self):
      return len(self.Y)

  def __getitem__(self, idx):
      return self.X[idx], self.Y[idx]
      
  


def train(dataloader, model, loss_fn, optimizer, device):
    '''
    This functions trains the model for an epoch.

    dataloader: dataloader object
    model: pytorch model
    loss_fn: loss function for the model
    optimizer: optimizer function
    device: device to run the model on (e.g. 'cuda')
    '''
    losses=[]
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    return losses


######################################


class DeepBrownianEncoder(torch.nn.Module):
    
    '''
    This is the deep learning architecture that used to predict the alpha parameter.
    It is based on a Transformer Encoder and it is made of 5 layers: embedding layer, positional encoding, transformer encoder, linear layer.
    It takes a batch of trajectories as input and outputs a vector of predicted alpha parameters.
    '''

    def __init__(self, embedding_dim=1, num_layers=1, num_heads=1, trajectory_length=100):
        super(DeepBrownianEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.trajectory_length = trajectory_length
        
        # Positional embedding module
        self.positional_encoder = nn.Embedding(trajectory_length, embedding_dim)

        self.linear1 = torch.nn.Linear(1, embedding_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.linear2 = torch.nn.Linear(self.trajectory_length * self.embedding_dim, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # Generate positional embeddings based on the position of each element in the input sequence
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        positional_embeddings = self.positional_encoder(positions)

        x = self.linear1(x) # Embed each step in the trajectory
        x = x + positional_embeddings  # Add positional embeddings to the input embeddings
        x = self.transformer_encoder(x)
        x = x.view(batch_size, -1) # Flatten the output tensor
        x = self.linear2(x) # Convert this tensor into a scalar
        return x
