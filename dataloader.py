import torch
import numpy as np
from typing import List, Tuple

class EmbeddingVectorsDataset(torch.utils.data.Dataset):
    # https://zenn.dev/a5chin/articles/original_data
    def __init__(self, X:np.ndarray, Y:np.ndarray, expand_dim: bool = False) -> None:
        super().__init__()
        self.X = X
        self.Y = X
        self.expand_dim = expand_dim
        
    def __getitem__(self,index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[index]
        y = self.Y[index]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        # Reshape for a Convolutional Model
        # if self.expand_dim:
        #     assert x.ndim==2,"Expected (N,D) shape"
        #     x = x.view(x.shape[0],-1,x.shape[1])
        #     y = y.view(y.shape[0],-1,y.shape[1])
        
        return x,y

    def __len__(self) -> int:
        return len(self.X)