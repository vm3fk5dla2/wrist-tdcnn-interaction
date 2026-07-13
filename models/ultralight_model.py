import torch.nn as nn

class UltraLightCNN1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(16, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace = True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
