import torch
import torch.nn as nn
import torch.nn.functional as F

class Modele(nn.Module):
    def __init__(self, in_channels=8):
        super(Modele,self).__init__()
        self.conv1 = self.conv_layer(in_channels, 16, 5)
        self.conv2 = self.conv_layer(16, 32, 3)
        self.pool = self.pool_layer()

        self.fc1 = nn.Linear(32,32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)





    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.mean(dim = 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


    def conv_layer(self, inp, out, kernel_size):
        return nn.Sequential(
            nn.Conv1d(inp, out, kernel_size = kernel_size, padding = (kernel_size - 1) // 2 ),
            nn.BatchNorm1d(out),
            nn.ReLU(),
        )

    def pool_layer(self):
        return nn.Sequential(
            nn.MaxPool1d(kernel_size = 2),
            nn.Dropout(0.5),
        )



