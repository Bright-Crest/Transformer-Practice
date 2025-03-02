from torch import nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fn1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.PReLU(),
            nn.Linear(32, 64),
            nn.PReLU(),
            nn.Linear(64, 16),
            nn.PReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, X):
        X = self.fn1(X)
        return X