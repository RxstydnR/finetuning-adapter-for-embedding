import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

class VectorTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
# Example
# adapter = VectorTransformer(input_dim=1536,output_dim=1536).to(device)


class LlamaAdapter(nn.Module):
    """
    Reference:
    - https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding_adapter/
    """
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            bias: bool = False,
            add_residual: bool = False,
        ) -> None:
            super(LlamaAdapter, self).__init__()
            self.in_features = in_features
            self.hidden_features = hidden_features
            self.out_features = out_features
            self.bias = bias
    
            self.linear1 = nn.Linear(in_features, hidden_features, bias=self.bias)
            self.linear2 = nn.Linear(hidden_features, out_features, bias=self.bias)
            self._add_residual = add_residual
            self.residual_weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv).

        Args:
            embed (Tensor): Input tensor.

        """
        output1 = self.linear1(embed)
        output1 = F.relu(output1)
        output2 = self.linear2(output1)

        if self._add_residual:
            output2 = self.residual_weight * output2 + embed

        return output2

# adapter = LlamaAdapter(
#     in_features     = 1536,
#     hidden_features = int(1536//4),
#     out_features    = 1536,
#     bias=True,
#     add_residual=False,
# ).to(device)


# 1D-Convolutinal AutoEncoder
class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1dAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=64, stride=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=64, stride=3, padding=1, dilation=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=64, stride=3, padding=1, output_padding=1, dilation=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=64, stride=3, padding=1, output_padding=1, dilation=1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


from typing import List
from langchain_core.embeddings import Embeddings

class OpenAIEmbeddingsWithAdapter(Embeddings):
    def __init__(self, model, adapter):
        self.model = model
        self.adapter = adapter

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        assert isinstance(texts,List)
        # 埋め込みベクトル
        vectors = self.model.embed_documents(texts)
        # 再変換
        vectors = self.adapter(Tensor(vectors).to("cpu"))
        vectors = vectors.to("cpu").detach().numpy()
        return vectors
    
    def embed_query(self, text: str) -> List[float]:
        assert isinstance(text,str)
        return self.embed_documents([text])[0]
