import torch
import torch.nn as nn
import torchvision
from torchvision.models.video import R3D_18_Weights

class VideoVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=53):
        super().__init__()
        base = torchvision.models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.encoder_out_dim = 512

        self.fc_mu = nn.Linear(self.encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.encoder_out_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_out_dim, self.encoder_out_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(mu)
        return recon, mu, logvar, logits


class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.net(x)


class SimpleDNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
