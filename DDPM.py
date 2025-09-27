import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

T = 300  # number of diffusion steps
betas = linear_beta_schedule(timesteps=T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

# precompute values for speed
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
def forward_diffusion_sample(x_0, t, device=device):
    """Add noise to image x_0 at timestep t"""
    noise = torch.randn_like(x_0).to(device)
    sqrt_alpha_hat = sqrt_alphas_cumprod[t][:, None, None, None].to(device)
    sqrt_one_minus_alpha_hat = sqrt_one_minus_alphas_cumprod[t][:, None, None, None].to(device)
    return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc_time = nn.Linear(1, 128)  # embed time step
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        # Embed time
        t = t.float().unsqueeze(-1) / T
        time_emb = self.fc_time(t).unsqueeze(-1).unsqueeze(-1)
        # Down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Add time conditioning
        x = x + time_emb
        # Up
        x = F.relu(self.deconv1(x))
        return self.deconv2(x)
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - 0.5) * 2)  # normalize to [-1,1]
    ])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=128, shuffle=True)

dataloader = get_data()
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 5
for epoch in range(epochs):
    for step, (x, _) in enumerate(dataloader):
        x = x.to(device)
        t = torch.randint(0, T, (x.shape[0],), device=device).long()
        x_noisy, noise = forward_diffusion_sample(x, t)
        noise_pred = model(x_noisy, t)
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")
@torch.no_grad()
def sample(model, n=16):
    img = torch.randn((n, 1, 28, 28)).to(device)
    for i in reversed(range(1, T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        pred_noise = model(img, t)
        alpha = alphas[i]
        alpha_hat = alphas_cumprod[i]
        beta = betas[i]

        if i > 1:
            noise = torch.randn_like(img)
        else:
            noise = torch.zeros_like(img)

        img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise) + torch.sqrt(beta) * noise
    return img

samples = sample(model)
grid = torchvision.utils.make_grid(samples, nrow=4, normalize=True)
plt.imshow(grid.permute(1,2,0).cpu().numpy())
plt.show()
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x/timesteps) + s) / (1+s) * torch.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
