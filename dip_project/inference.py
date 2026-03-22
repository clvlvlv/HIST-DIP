import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hist_dip_model.pth")
INPUT_DIR = os.path.join(BASE_DIR, "input_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_images")

INPUT_DEPTH = 32
SKIP_N33D = 128
SKIP_N33U = 128
SKIP_N11 = 4
NUM_SCALES = 5
PAD = 'reflection'
UPSAMPLE_MODE = 'bilinear'

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 网络定义 (必须与 train.py 完全一致) ====================
def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
    padder = None
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(kernel_size // 2)
    elif pad == 'replication':
        padder = nn.ReplicationPad2d(kernel_size // 2)
    layers = [padder] if padder else []
    layers.append(nn.Conv2d(in_f, out_f, kernel_size, stride=stride, bias=bias))
    return nn.Sequential(*layers)


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def act():
    return nn.LeakyReLU(inplace=True)


class SkipBlock(nn.Module):
    def __init__(self, depth, in_channels, is_last_block=False):
        super().__init__()
        self.depth = depth
        self.is_last_block = is_last_block

        self.down_conv = nn.Sequential(
            conv(in_channels, SKIP_N33D, 3, pad=PAD),
            bn(SKIP_N33D),
            act()
        )

        self.skip_proj = conv(in_channels, SKIP_N33U, 1, pad=PAD)

        if is_last_block:
            self.bottleneck = nn.Sequential(
                conv(SKIP_N33D, SKIP_N11, 3, pad=PAD),
                bn(SKIP_N11),
                act(),
                conv(SKIP_N11, SKIP_N33D, 3, pad=PAD),
                bn(SKIP_N33D),
                act()
            )
            self.pool = None
            self.recursive = None
            self.up_conv = None
        else:
            self.pool = nn.MaxPool2d(2, 2)
            self.recursive = SkipBlock(depth - 1, SKIP_N33D, is_last_block=(depth - 1 == 0))
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=UPSAMPLE_MODE),
                conv(SKIP_N33D, SKIP_N33U, 3, pad=PAD),
                bn(SKIP_N33U),
                act()
            )

    def forward(self, x):
        down_out = self.down_conv(x)

        if self.is_last_block:
            return self.bottleneck(down_out)
        else:
            pooled = self.pool(down_out)
            rec_out = self.recursive(pooled)
            up_out = self.up_conv(rec_out)
            skip_val = self.skip_proj(x)
            return up_out + skip_val


def get_net(input_depth):
    model_body = SkipBlock(depth=NUM_SCALES - 1, in_channels=input_depth, is_last_block=(NUM_SCALES - 1 == 0))
    final_conv = conv(SKIP_N33U, 1, 1, pad=PAD)
    return nn.Sequential(model_body, final_conv)


def get_noise(input_depth, spatial_size):
    torch.manual_seed(123)
    nz = torch.zeros(1, input_depth, spatial_size[0], spatial_size[1]).to(DEVICE)
    nz.uniform_()
    return nz


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return None
    net = get_net(INPUT_DEPTH)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.to(DEVICE)
    net.eval()
    print("✅ Model loaded.")
    return net


def process_image(image_path, model):
    try:
        img = Image.open(image_path).convert('L')
        w, h = img.size
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        if new_w != w or new_h != h:
            img = img.crop((0, 0, new_w, new_h))

        input_noise = get_noise(INPUT_DEPTH, (new_h, new_w))

        with torch.no_grad():
            output_tensor = model(input_noise)

        output_np = np.clip(output_tensor.squeeze().cpu().numpy(), 0, 1)
        output_img = Image.fromarray((output_np * 255).astype(np.uint8))

        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        save_path = os.path.join(OUTPUT_DIR, f"{name}_restored.png")
        output_img.save(save_path)
        return True, save_path
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    model = load_model()
    if not model: exit(1)

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]

    if not files:
        print(f"⚠️ No images in {INPUT_DIR}")
        exit(0)

    print(f"Processing {len(files)} images...")
    success = 0
    for f in tqdm(files):
        ok, res = process_image(os.path.join(INPUT_DIR, f), model)
        if ok:
            success += 1
        else:
            print(f"Failed {f}: {res}")

    print(f"Done. Success: {success}/{len(files)}")