#Main library

import utils
import numpy as np
import torch



class DeepBrownianMotion:

    def __init__(self, device) -> None:
        

        self.device=device

        #Load model and weights
        self.model=utils.DeepBrownianEncoder(embedding_dim=30, num_layers=1, num_heads=2).to(self.device)
        self.model.load_state_dict(torch.load('model/model_weights.pth'))
        self.model.eval()


    def inference(self, X:np.ndarray) -> np.ndarray:

        #If only one sample is provided, fix its shape
        if(len(X.shape)==1):
            X=np.expand_dims(X, axis=0)
        X=np.expand_dims(X, axis=2)
        #Normalize X
        X=(X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
        #Convert X to a tensor    
        X=torch.tensor(X, dtype=torch.float32).to(self.device)
        result=self.model(X)

        return result.detach().cpu().numpy()