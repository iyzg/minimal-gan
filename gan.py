# General comments with one hashtag
### Any pedagogical comments have 3
### Took some tricks from https://github.com/soumith/ganhacks
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from torchvision.datasets import MNIST


# ----- hyperparameters -----

h_size = 128
noise_dim = 10

epochs = 3_000
batch_size = 64
d_steps = 40  # Number of steps to train discrim for per epoch
stats_every_n = 50
pic_every_n = 500

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------


class Generator(nn.Module):
    """Generator which takes in noise and tries to generate something from the training set"""

    def __init__(self, z_dim, h_dim, o_dim):
        super().__init__()

        # The tanh is since we normalize the data to [-1, 1]
        ### We use LeakyReLU here to try and promote good gradients while training
        ### You could probably use GeLU or some other ELU varieant here and be fine
        self.layers = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.LeakyReLU(), nn.Linear(h_dim, o_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    """Discriminator which tells whether sample is real or fake"""

    ### Sigmoid at the end is to clamp to [0, 1] so output can be interpreted
    ### as a probability.
    def __init__(self, input_dim, h_dim, dropout_rate=0.2):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def visualize_data(x, epoch=None):
    plt.figure()
    plt.grid(visible=False)
    plt.axis("off")
    if epoch:
        plt.title(f"Epoch {epoch}")
    plt.imshow(x, cmap="gray")
    plt.show()
    time.sleep(0.1)


def load_data():
    """Load MNIST and normalize data range to [-1, 1]"""
    dataset = MNIST(root="data", train=True, download=True)
    data = dataset.train_data.to(torch.float32)
    data = (data / data.max() - 0.5) * 2
    return data


if __name__ == "__main__":
    data = load_data().to(device)
    n_imgs = data.shape[0]

    data = data.reshape(n_imgs, -1).to(torch.float32)
    input_dim = data.shape[-1]

    g = Generator(noise_dim, h_size, input_dim).to(device)
    d = Discriminator(input_dim, h_size).to(device)

    d_opt = torch.optim.AdamW(d.parameters(), lr=0.001)
    g_opt = torch.optim.AdamW(g.parameters(), lr=0.001)

    for e in range(epochs):
        # Train the discriminator for a certain number of steps per epoch
        ### This is so the generator will have some sort of signal to
        ### train on
        for d_step in range(d_steps):
            z = torch.randn((batch_size, noise_dim)).to(device)  # B, Z
            real_idxs = torch.randint(0, n_imgs, (batch_size,))

            fake_imgs = g(z)
            real_imgs = data[real_idxs]  # B, ID

            d_reals = d(real_imgs)
            d_fakes = d(fake_imgs.detach())
            assert d_reals.shape == d_fakes.shape

            loss = -(torch.log(d_reals).mean() + torch.log(1 - d_fakes).mean())

            d_opt.zero_grad()
            loss.backward()
            d_opt.step()

        d.eval()

        z = torch.randn((batch_size, noise_dim)).to(device)  # B, Z
        fake_imgs = g(z)
        loss = -torch.log(d(fake_imgs)).mean()

        g_opt.zero_grad()
        loss.backward()
        g_opt.step()

        d.train()

        if e % pic_every_n == 0:
            visualize_data(fake_imgs[0].reshape(28, 28).detach().numpy(), e)

        if e % stats_every_n == 0:
            grad_norm = (
                sum(p.grad.norm() ** 2 for p in g.parameters() if p.grad is not None)
                ** 0.5
            ) / sum(1 for p in g.parameters() if p.grad is not None)
            print(
                f"epoch {e:4d} | g avg grad norm {grad_norm:4.2f} | g loss {loss.detach().item():4.2f}"
            )
