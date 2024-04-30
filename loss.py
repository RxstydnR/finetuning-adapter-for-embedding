import torch
import torch.nn as nn

class Weighted_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(Weighted_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0)
        self.alpha = alpha

    def forward(self, output, target):
        mse = self.mse_loss(output, target)
        cos = self.cos_loss(output.squeeze(1), target.squeeze(1), torch.ones(output.shape[0]))
        # print(mse,cos)
        return self.alpha * mse + (1 - self.alpha) * cos