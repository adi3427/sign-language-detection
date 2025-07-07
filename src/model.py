import torch.nn as nn
from torchvision import models

class ResNetLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=100):  # default num_classes; set later
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = False  # freeze CNN

        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        with torch.no_grad():
            cnn_out = self.cnn(x).view(B, T, -1)
        lstm_out, _ = self.lstm(cnn_out)
        return self.fc(lstm_out[:, -1, :])
