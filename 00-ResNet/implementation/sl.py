import io
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as img

to_tensor = transforms.ToTensor()

HR_PATCH_SIZE = 96
SCALE = 4
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE

DATA_DIR = Path(__file__).parent / "data" / "div2k"

random_crop = transforms.RandomCrop(HR_PATCH_SIZE)


class DIV2KDataset(Dataset):
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        with zipfile.ZipFile(zip_path) as zf:
            self.names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
        self._zf = None
        self._cache = {}

    def __len__(self):
        return len(self.names)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zf"] = None
        return state

    def _load_full(self, idx):
        if idx in self._cache:
            return self._cache[idx]

        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        with self._zf.open(self.names[idx]) as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")

        self._cache[idx] = image
        return image

    def __getitem__(self, idx):
        hr = random_crop(self._load_full(idx))
        lr = hr.resize((LR_PATCH_SIZE, LR_PATCH_SIZE), Image.BICUBIC)
        return {"hr": to_tensor(hr), "lr": to_tensor(lr)}


class ResLayer(nn.Module):

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


    def __init__(self, channels_in: int, channels_out: int):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, channels_in, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels_out)

        self.relu = nn.ReLU()

        if channels_in != channels_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels_out),
            )
        else:
            self.downsample = None



class ResNet(nn.Module):

    def __init__(self, lr, hr, channels=64, n=8):
        super().__init__()

        assert hr % lr == 0

        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)
        self.reslayers = nn.ModuleList([ResLayer(channels, channels) for _ in range(n)])
        self.upsample = nn.Upsample(scale_factor=hr // lr)
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor):
        out = self.conv_in(x)
        identity = out

        for reslayer in self.reslayers:
            out = reslayer(out)

        out = out + identity
        out = self.upsample(out)
        out = self.conv_out(out)

        return out


class PlainCNN(nn.Module):

    def __init__(self, lr, hr, channels=64, n=8):
        super().__init__()

        assert hr % lr == 0

        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)

        layers = []
        for _ in range(n):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            ]
        self.body = nn.Sequential(*layers)

        self.upsample = nn.Upsample(scale_factor=hr // lr)
        self.conv_out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x: torch.Tensor):
        out = self.conv_in(x)
        out = self.body(out)
        out = self.upsample(out)
        out = self.conv_out(out)

        return out


    
def train(train: DataLoader, test: DataLoader, model: nn.Module, epochs: int, device: str, learning_rate: int=.0001):
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train:
            lr, hr = batch["lr"].to(device), batch["hr"].to(device)

            optimizer.zero_grad()
            prediction = model(lr)
            loss = criterion(prediction, hr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test:
                lr, hr = batch["lr"].to(device), batch["hr"].to(device)
                test_loss += criterion(model(lr), hr).item()

        print(f"epoch {epoch}: train {train_loss/len(train):.4f}  test {test_loss/len(test):.4f}")


CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def train_or_load(name: str, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int, device: str):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        print(f"loaded {name} from {checkpoint_path}")
    else:
        train(train_loader, test_loader, model, epochs, device)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved {name} to {checkpoint_path}")

    return model


def compare_visually(models: dict, dataset: DIV2KDataset, device: str, idx: int = 0):
    full_hr = dataset._load_full(idx)
    full_lr = full_hr.resize((full_hr.width // SCALE, full_hr.height // SCALE), Image.BICUBIC)
    lr_tensor = to_tensor(full_lr).unsqueeze(0).to(device)

    outputs = {"LR (input)": full_lr, "HR (ground truth)": full_hr}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            prediction = model(lr_tensor).clamp(0, 1)
        outputs[name] = transforms.ToPILImage()(prediction.squeeze(0).cpu())

    fig, axes = plt.subplots(1, len(outputs), figsize=(4 * len(outputs), 4))
    for ax, (title, image) in zip(axes, outputs.items()):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    div2k_train = DIV2KDataset(DATA_DIR / "DIV2K_train_HR.zip")
    div2k_test = DIV2KDataset(DATA_DIR / "DIV2K_valid_HR.zip")

    train_dataloader = DataLoader(div2k_train, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(div2k_test, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)

    print("hr", div2k_train[0]["hr"].shape, "lr", div2k_train[0]["lr"].shape)
    full_hr = div2k_train._load_full(0)
    full_lr = full_hr.resize((full_hr.width // SCALE, full_hr.height // SCALE), Image.BICUBIC)
    print("full_hr size", full_hr.size, "full_lr size", full_lr.size)

    fig, (ax_hr, ax_lr) = plt.subplots(1, 2)
    ax_hr.imshow(full_hr)
    ax_hr.set_title(f"Full HR {full_hr.size}")
    ax_hr.axis("off")
    ax_lr.imshow(full_lr, interpolation="nearest")
    ax_lr.set_title(f"Full LR {full_lr.size}")
    ax_lr.axis("off")
    plt.show()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    resnet_model = ResNet(lr=LR_PATCH_SIZE, hr=HR_PATCH_SIZE)
    plain_model = PlainCNN(lr=LR_PATCH_SIZE, hr=HR_PATCH_SIZE)

    train_or_load("resnet", resnet_model, train_dataloader, test_dataloader, epochs=30, device=device)
    train_or_load("plain_cnn", plain_model, train_dataloader, test_dataloader, epochs=30, device=device)

    compare_visually({"ResNet": resnet_model, "PlainCNN": plain_model}, div2k_test, device, idx=0)
