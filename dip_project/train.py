import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==================== 配置部分 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_train")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 超参数
INPUT_DEPTH = 32
SKIP_N33D = 128
SKIP_N33U = 128
SKIP_N11 = 4
NUM_SCALES = 5  # 意味着有 5 层下采样，最底层是第 0 层
PAD = 'reflection'
UPSAMPLE_MODE = 'bilinear'
ENFORSE_DIV32 = True
LR = 0.01
NUM_ITER = 2000

INPUT_IMAGE_PATH = os.path.join(DATA_DIR, "shaky.tif")
if not os.path.exists(INPUT_IMAGE_PATH):
    print(f"❌ Error: Image not found at {INPUT_IMAGE_PATH}")
    sys.exit(1)


# ==================== 【核心修复】网络定义 ====================
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

        # 1. Down Conv (映射到 skip_n33d)
        self.down_conv = nn.Sequential(
            conv(in_channels, SKIP_N33D, 3, pad=PAD),
            bn(SKIP_N33D),
            act()
        )

        # 2. Skip Projection (将输入映射到 skip_n33u 以便后续相加)
        # 注意：这里先不相加，等上采样回来再加
        self.skip_proj = conv(in_channels, SKIP_N33U, 1, pad=PAD)

        if is_last_block:
            # --- Bottleneck (最底层) ---
            # 结构：Conv -> Act -> Conv -> Act (保持尺寸不变)
            self.bottleneck = nn.Sequential(
                conv(SKIP_N33D, SKIP_N11, 3, pad=PAD),
                bn(SKIP_N11),
                act(),
                conv(SKIP_N11, SKIP_N33D, 3, pad=PAD),
                bn(SKIP_N33D),
                act()
            )
            # 最底层没有 Pool，也没有 Recursive，也没有 Upsample
            self.pool = None
            self.recursive = None
            self.up_conv = None
        else:
            # --- 中间层 ---
            self.pool = nn.MaxPool2d(2, 2)
            # 递归调用下一层，输入通道为 SKIP_N33D
            self.recursive = SkipBlock(depth - 1, SKIP_N33D, is_last_block=(depth - 1 == 0))

            # 上采样 + 卷积 (将深层输出映射回 skip_n33u)
            # 这里的输入通道是 SKIP_N33D (来自递归的输出)
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=UPSAMPLE_MODE),
                conv(SKIP_N33D, SKIP_N33U, 3, pad=PAD),
                bn(SKIP_N33U),
                act()
            )

    def forward(self, x):
        # 1. 下采样分支
        down_out = self.down_conv(x)

        if self.is_last_block:
            # 最底层：直接过 bottleneck
            bottle_out = self.bottleneck(down_out)
            # 注意：最底层的输出需要被上一层上采样。
            # 但为了统一接口，我们在这里不做上采样，直接返回 bottleneck 结果 (通道 SKIP_N33D)
            # 上一层会负责 Upsample。
            # 等等，为了配合上一层的 up_conv (期望输入 SKIP_N33D)，这样是对的。
            # 但是 Skip 连接怎么加？
            # 最底层没有 Skip 连接加法操作，因为它是最深处。
            # 它的输出将被上一层上采样后，与上一层的 skip_proj 相加。
            return bottle_out
        else:
            # 中间层
            pooled = self.pool(down_out)
            rec_out = self.recursive(pooled)  # 输出通道 SKIP_N33D

            # 上采样
            up_out = self.up_conv(rec_out)  # 输出通道 SKIP_N33U, 尺寸恢复到 x 的尺寸

            # Skip 连接：上采样结果 + 投影后的原始输入
            skip_val = self.skip_proj(x)  # 输出通道 SKIP_N33U, 尺寸同 x

            return up_out + skip_val


def get_net(input_depth):
    # 创建最外层 block
    # depth 从 NUM_SCALES-1 开始递减到 0
    # is_last_block 当 depth==0 时为 True
    model_body = SkipBlock(depth=NUM_SCALES - 1, in_channels=input_depth, is_last_block=(NUM_SCALES - 1 == 0))

    # 最终输出层：将 SKIP_N33U 映射到 1 (灰度)
    final_conv = conv(SKIP_N33U, 1, 1, pad=PAD)

    return nn.Sequential(model_body, final_conv)


# ==================== 数据处理 ====================
def load_image(path, enforse_div32=False):
    img = Image.open(path).convert('L')
    w, h = img.size
    if enforse_div32:
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        if new_w != w or new_h != h:
            img = img.crop((0, 0, new_w, new_h))
            print(f"Cropped image from {w}x{h} to {new_w}x{new_h}")

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    return img_tensor, img.size


def get_noise(input_depth, spatial_size):
    nz = torch.zeros(1, input_depth, spatial_size[0], spatial_size[1]).to(DEVICE)
    nz.uniform_()
    return nz


# ==================== 主程序 ====================
if __name__ == "__main__":
    print(f"Loading image: {INPUT_IMAGE_PATH}")
    img_tensor, original_size = load_image(INPUT_IMAGE_PATH, enforse_div32=ENFORSE_DIV32)
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    print(f"Input shape: {img_tensor.shape}")

    input_noise = get_noise(INPUT_DEPTH, (H, W))

    net = get_net(INPUT_DEPTH)
    net.to(DEVICE)

    # 架构测试
    try:
        with torch.no_grad():
            test_out = net(input_noise)
        print(f"✅ Network Architecture OK! Output shape: {test_out.shape}")
    except Exception as e:
        print(f"❌ Network Architecture Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training for {NUM_ITER} iterations...")
    history_loss = []
    best_psnr = 0
    best_state = None

    pbar = tqdm(range(NUM_ITER))
    for i in pbar:
        def closure():
            optimizer.zero_grad()
            out = net(input_noise)
            loss = criterion(out, img_tensor)
            loss.backward()
            return loss


        loss = closure()
        optimizer.step()

        if i % 10 == 0 or i == NUM_ITER - 1:
            with torch.no_grad():
                out = net(input_noise)
                loss_val = criterion(out, img_tensor).item()
                out_np = np.clip(out.squeeze().cpu().numpy(), 0, 1)
                target_np = img_tensor.squeeze().cpu().numpy()
                current_psnr = psnr(target_np, out_np, data_range=1.0)

                history_loss.append(loss_val)
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

                pbar.set_description(f"Loss: {loss_val:.6f}, PSNR: {current_psnr:.2f}")

    print("\nTraining finished.")

    if best_state:
        net.load_state_dict(best_state)
        model_path = os.path.join(MODEL_DIR, "hist_dip_model.pth")
        torch.save(net.state_dict(), model_path)
        print(f"✅ Model saved: {model_path}")

    with torch.no_grad():
        final_out = net(input_noise)
        final_np = np.clip(final_out.squeeze().cpu().numpy(), 0, 1)
        final_img = Image.fromarray((final_np * 255).astype(np.uint8))
        save_path = os.path.join(OUTPUT_DIR, "restored_result.png")
        final_img.save(save_path)
        print(f"✅ Result saved: {save_path}")

    plt.figure(figsize=(10, 4))
    plt.plot(history_loss)
    plt.title("Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()