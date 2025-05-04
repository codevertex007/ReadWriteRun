import torch.nn as nn

class SkipGramModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, )
        """

        X = self.embedding(inputs) # batch_size X embedding_dimension
        X = self.linear(X) # batch_size X vocab_size
        return X
