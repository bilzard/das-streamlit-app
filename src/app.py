import gc
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as TF
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2
from transformers import AutoModel, AutoProcessor

MODEL_PATH = "/ml-docker/input/hf/google/siglip-so400m-patch14-384"


def is_deterministic_algorithm_enabled():
    return (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":16:8"
    )


def seed_everything(seed: int = 42, use_deterministic_algorithm: bool = False):
    """Set the same seed for reproducibility across random, numpy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithm)


def interpolate(image, target_size, mode="bicubic"):
    return TF.interpolate(
        image,
        size=(target_size, target_size),
        mode=mode,
        **({"align_corners": False} if mode == "bicubic" or mode == "bilinear" else {}),
    )


def add_positional_rolling(image, max_shift=10):
    """
    `torch.roll()` を使って勾配を維持したまま Positional Jitter を適用
    :param image: (N, C, H, W) の PyTorch Tensor
    :param max_shift: 最大移動ピクセル数
    :return: ジッターを加えた画像
    """
    N, C, H, W = image.shape
    jittered_images = torch.zeros_like(image)

    # 各画像にランダムなシフトを適用
    for i in range(N):
        dx = torch.randint(-max_shift, max_shift + 1, (1,)).item()  # X 方向のシフト
        dy = torch.randint(-max_shift, max_shift + 1, (1,)).item()  # Y 方向のシフト

        # **torch.roll() を使って画像をシフト（勾配を維持）**
        jittered_images[i] = torch.roll(image[i], shifts=(dy, dx), dims=(1, 2))  # type: ignore

    return jittered_images


def apply_cutout(image, max_size=50):
    """
    `cutout` を適用してランダムな領域をマスク
    :param image: (N, C, H, W) の PyTorch Tensor
    :param max_size: 切り取る最大サイズ
    :return: `cutout` を適用した画像
    """
    N, C, H, W = image.shape
    cutout_image = image.clone()

    for i in range(N):
        # **ランダムな位置を選択**
        x = torch.randint(0, W - max_size, (1,)).item()
        y = torch.randint(0, H - max_size, (1,)).item()
        w = torch.randint(10, max_size, (1,)).item()
        h = torch.randint(10, max_size, (1,)).item()

        # **選択した領域をゼロ（黒）にする**
        cutout_image[i, :, y : y + h, x : x + w] = 0

    return cutout_image


def color_shift(image, shift: float = 1.0):
    """
    画像に Color Shift を適用
    :param image: (N, C, H, W) の PyTorch Tensor
    :return: Color Shift された画像
    """
    N, C, H, W = image.shape
    mu = torch.zeros((N, C, 1, 1), device=image.device).uniform_(
        -shift, shift
    )  # U[-1,1]
    sigma = torch.exp(
        torch.zeros((N, C, 1, 1), device=image.device).uniform_(-shift, shift)
    )  # exp(U[-1,1])

    return sigma * image + mu  # カラースケール & シフト


def gaussian_noise(image, sigma=1.0):
    """
    画像に Gaussian Smoothing を適用
    :param image: (N, C, H, W) の PyTorch Tensor
    :return: ノイズを加えた画像
    """
    noise = torch.randn_like(image)  # N(0,1) のノイズを生成
    return image + sigma * noise  # 画像にノイズを加える


def gaussian_blur(image, kernel_size=(3, 3), sigma=(0.1, 2.0)):
    return F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)


def random_posterize(image, bits=3):
    return F2.posterize(image, bits)


def total_variation_loss(image):
    """
    Total Variation (TV) Loss を計算
    :param image: (1, 3, H, W) の PyTorch Tensor
    :return: TV Loss のスカラー値
    """
    dx = torch.diff(image, dim=2).abs().mean()  # 横方向の変化
    dy = torch.diff(image, dim=3).abs().mean()  # 縦方向の変化
    return dx + dy


def l1_regularization(image):
    """
    L1 正則化 (スパース性を強調)
    :param image: (1, 3, H, W) の PyTorch Tensor
    :return: L1 Loss のスカラー値
    """
    return image.abs().mean()


def linear_schedule(step, max_steps, start=0.5, end=0.0, **kwargs):
    """
    線形スケジューリング (0.5 → 0)
    :param step: 現在のステップ
    :param max_steps: 総ステップ数
    :param start: 初期ノイズ強度
    :param end: 最終ノイズ強度
    :return: スケジュールされたノイズの標準偏差
    """
    return start + (end - start) * (step / max_steps)


def exponential_schedule(step, max_steps, start=0.5, end=0.0, rate=1.0, **kwargs):
    """
    指数減衰スケジューリング (0.5 → 0) に反応速度パラメータを追加

    :param step: 現在のステップ
    :param max_steps: 総ステップ数
    :param start: 初期ノイズ強度（正の値）
    :param end: 最終ノイズ強度（通常は 0.0 を指定するが、計算上は正の非常に小さな値を用いる）
    :param rate: 減衰速度の調整パラメータ（1.0 で通常の速度、>1 で速く、0<rate<1 で遅くなる）
    :return: スケジュールされたノイズの標準偏差
    """
    if step >= max_steps:
        return end
    effective_end = end if end > 0 else 1e-8
    decay_rate = (math.log(effective_end / start) / max_steps) * rate  # 反応速度を調整
    return start * math.exp(decay_rate * step)


def cosine_schedule(step, max_steps, start=0.5, end=0.0, **kwargs):
    """
    コサイン減衰スケジューリング (0.5 → 0)

    :param step: 現在のステップ
    :param max_steps: 総ステップ数
    :param start: 初期ノイズ強度
    :param end: 最終ノイズ強度
    :return: スケジュールされたノイズの標準偏差
    """
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    return end + (start - end) * cosine_decay


def calc_sim(model, processor, image, texts, device="cuda"):
    inputs = processor(
        text=texts,
        images=image,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model.text_model(input_ids=inputs["input_ids"].to(device))
        text_features = outputs.pooler_output
        text_features /= text_features.norm(dim=-1, keepdim=True)

        outputs = model.vision_model(pixel_values=inputs["pixel_values"].to(device))
        image_features = outputs.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze()

        return similarities.mean().item()


def inv_process(image):
    return (
        ((image / 2 + 0.5) * 255)
        .to(torch.uint8)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )


def plot_histogram(image):
    import matplotlib.pyplot as plt

    plt.hist(image[..., 0].flatten(), color="r", label="R", alpha=0.3)
    plt.hist(image[..., 1].flatten(), color="g", label="G", alpha=0.3)
    plt.hist(image[..., 2].flatten(), color="b", label="B", alpha=0.3)
    plt.legend()
    plt.show()


@dataclass
class Config:
    num_steps: int = 100
    batch_size: int = 8
    lambda_tv: float = 1e-2
    lambda_l1: float = 0.05
    lr: float = 0.1
    betas: tuple[float, float] = (0.5, 0.99)
    eta_min_ratio: float = 0.01
    eval_steps: int = 1
    max_shift: int = 64
    noise_schedule: str = "exponential"
    noise_schedule_params: dict[str, Any] = field(
        default_factory=lambda: {"decay_rate": 1.0}
    )
    noise_std_range: tuple[float, float] = (0.5, 0.05)
    color_shift_range: tuple[float, float] = (1.0, 0.1)
    image_resolutions: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    checkpoint_interval: int = 100
    mode: str = "bilinear"
    seed: int = 42
    use_deterministic_algorithm: bool = False


noise_schedule_map = {
    "linear": linear_schedule,
    "exponential": exponential_schedule,
    "cosine": cosine_schedule,
}


class DasAttacker:
    def __init__(self, model_path, device="cuda", config: Config = Config()):
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device
        self.text_features = None
        self.images = None
        self.config = config
        self.mode = config.mode
        self.checkpoints = []
        self.scores = []
        self.noise_schedule = noise_schedule_map[config.noise_schedule]
        self.max_size = self.config.image_resolutions[-1]
        self._freeze()

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def cache_positive_text_embeddings(self, texts):
        inputs = self.processor.tokenizer(
            texts, return_tensors="pt", padding="max_length", truncation=True
        )
        with torch.no_grad():
            outputs = self.model.text_model(
                input_ids=inputs["input_ids"].to(self.device)
            )
            text_features = outputs.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.texts = texts
        self.text_features = text_features

    def attack(self, progress_callback=None):
        cfg = self.config
        assert self.text_features is not None
        image_stack = [
            (nn.Parameter(torch.randn(1, 3, s, s, device=self.device) / s))
            for s in cfg.image_resolutions
        ]
        optimizer = torch.optim.Adam(image_stack, lr=cfg.lr, betas=cfg.betas)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_steps, eta_min=cfg.eta_min_ratio * cfg.lr
        )
        for step in range(cfg.num_steps):
            optimizer.zero_grad()
            noise_std = self.noise_schedule(
                step,
                cfg.num_steps,
                start=cfg.noise_std_range[0],
                end=cfg.noise_std_range[1],
                **cfg.noise_schedule_params,
            )
            c_shift = exponential_schedule(
                step,
                cfg.num_steps,
                start=cfg.color_shift_range[0],
                end=cfg.color_shift_range[1],
            )
            image = torch.stack(
                [interpolate(i, self.max_size, mode=self.mode) for i in image_stack]
            ).mean(0)
            image = image.tanh()
            images = image.repeat(cfg.batch_size, 1, 1, 1)
            images = add_positional_rolling(images, max_shift=cfg.max_shift)
            images = color_shift(images, c_shift)
            images = gaussian_noise(images, sigma=noise_std)
            self.pad = pad = cfg.max_shift // 2
            images = images[..., pad:-pad, pad:-pad]
            if images.shape[-1] != 384:
                images = interpolate(images, 384, mode=self.mode)
            outputs = self.model.vision_model(pixel_values=images)
            image_features = outputs.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ self.text_features.T).squeeze()
            loss_similarity = -similarities.mean()
            loss_tv = total_variation_loss(images)
            loss_l1 = l1_regularization(images)
            loss = loss_similarity + cfg.lambda_tv * loss_tv + cfg.lambda_l1 * loss_l1
            loss.backward()
            optimizer.step()
            scheduler.step()

            if progress_callback:
                progress_callback((step + 1) / cfg.num_steps)

            if (step + 1) % cfg.checkpoint_interval == 0:
                images_list = []
                image_tensor = torch.cat(
                    [
                        interpolate(i.detach(), self.max_size, mode=self.mode).cpu()
                        for i in image_stack
                    ]
                )
                for i in range(len(image_tensor)):
                    images_list.append(image_tensor[i])
                self.checkpoints.append(torch.stack(images_list))
                self.scores.append(similarities.mean().item())
        self.image_stack = image_stack
        torch.cuda.empty_cache()
        gc.collect()

    def evaluate(self):
        pad = self.pad
        with torch.no_grad():
            image = torch.stack(
                [
                    interpolate(i, self.max_size, mode=self.mode)[
                        ..., pad:-pad, pad:-pad
                    ]
                    for i in self.image_stack
                ]
            ).mean(0)
            image = image.tanh()
            image_np = inv_process(image)
            score = calc_sim(
                self.model, self.processor, image_np, self.texts, device=self.device
            )
            return image_np, score

    def get_checkpoint_images(self):
        pad = self.pad
        images = []
        with torch.no_grad():
            for idx, (checkpoint, score) in enumerate(
                zip(self.checkpoints, self.scores)
            ):
                img = torch.cat(
                    [
                        interpolate(i.unsqueeze(0), 384, mode=self.mode)[
                            ..., pad:-pad, pad:-pad
                        ]
                        for i in checkpoint
                    ]
                ).mean(0)
                img = img.tanh()
                img_np = inv_process(img)
                images.append(
                    (img_np, (idx + 1) * self.config.checkpoint_interval, score)
                )
        return images


def main():
    st.title("Demo: Direct Ascent Synthesis (DAS)")

    st.sidebar.header("Parameters")
    num_steps = st.sidebar.slider(
        "#steps", min_value=0, max_value=100, value=100, step=10
    )
    batch_size = st.sidebar.slider(
        "batch size", min_value=8, max_value=32, value=8, step=8
    )
    lr = st.sidebar.slider("lr", min_value=0.00, max_value=0.20, value=0.10, step=0.05)

    st.sidebar.header("Generation")
    prefix = st.sidebar.selectbox(
        "Prefix",
        [
            "A beautifully detailed illustration of",
            "A photorealistic illustration of",
            "",
        ],
    )
    suffix = st.sidebar.selectbox(
        "Suffix",
        [
            "in a cyberpunk style",
            "in a vibrant comic book style",
            "in a whimsical Disney animation style",
            "in a soft and painterly Ghibli-inspired style",
            "in an impressionist painting style",
            "",
        ],
    )
    use_deterministic_algorithm = (
        st.sidebar.checkbox(
            "Use Deterministic Algorithm",
            value=False,
            help="**Note**: it does not ensure reproducibility and it becomes slower",
        )
        if is_deterministic_algorithm_enabled()
        else False
    )
    interpolation_mode = st.sidebar.selectbox(
        "Interpolation Mode",
        ["bilinear", "bicubic"] if not use_deterministic_algorithm else ["bilinear"],
    )
    set_seed = st.sidebar.checkbox("Set Seed", value=False)
    seed = st.sidebar.number_input("seed", value=42, step=1, disabled=not set_seed)

    st.sidebar.header("Regularization")
    lambda_tv_exp = st.sidebar.slider(
        "Texture Suppression (TV)",
        min_value=-5,
        max_value=0,
        value=-3,
        help="higher value suppresses texture and increases smoothness",
    )
    lambda_tv = 10**lambda_tv_exp
    lambda_l1 = st.sidebar.slider(
        "Color Suppression (L1)",
        min_value=0.0,
        max_value=0.5,
        value=0.10,
        step=0.05,
        help="higher value suppresses color and increases grayness",
    )

    st.sidebar.header("Augmentation")
    st.sidebar.subheader("Gaussian Noise")
    noise_schedule = st.sidebar.selectbox(
        "Noise Schedule", ["linear", "exponential", "cosine"], index=1
    )
    noise_stds = st.sidebar.slider(
        "Noise Intensity Range",
        min_value=0.0,
        max_value=1.0,
        value=(0.1, 0.5),
        step=0.05,
    )
    decay_rate = st.sidebar.slider(
        "Noise Decay Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.1
    )
    st.sidebar.subheader("Positional Jitter")
    st.sidebar.subheader("Color Shift")
    color_shift_range = st.sidebar.slider(
        "Color Shift Range", min_value=0.0, max_value=1.0, value=(0.05, 0.10), step=0.05
    )

    cfg = Config(
        num_steps=num_steps,
        batch_size=batch_size,
        lambda_tv=lambda_tv,
        lambda_l1=lambda_l1,
        lr=lr,
        checkpoint_interval=num_steps,
        noise_schedule=noise_schedule,
        noise_std_range=noise_stds,
        noise_schedule_params={"decay_rate": decay_rate},
        color_shift_range=color_shift_range,
        seed=seed,
        mode=interpolation_mode,
        use_deterministic_algorithm=use_deterministic_algorithm,
    )

    sample_prompts = [
        "a gigantic metallic Mona Lisa in red pajama attacking the city",
        "a massive gamma-ray burst engulfs the towering Tokyo Tower on Earth's final day",
        "a stunning Mount Fuji, a majestic hawk, and a symbolic eggplant, set against the neon-lit skyline of cyberpunk Tokyo",
        "a cute Japanese Geisha girl wearing a rainbow-colored kimono, crying with huge drops of tears",
    ]
    selected_sample = st.radio("Choose a sample prompt:", sample_prompts, index=None)
    prompt = st.text_area(
        "Enter your prompt:",
        value=selected_sample if selected_sample else sample_prompts[0],
    )
    final_prompt = f"{prefix} {prompt}, {suffix}"

    if st.button("Generate Image"):
        st.write(f"**Prompt**: `{final_prompt}`")

        st.write("Generating Image...")
        if set_seed:
            seed_everything(
                cfg.seed,
                use_deterministic_algorithm=cfg.use_deterministic_algorithm,
            )
        attacker = DasAttacker(model_path=MODEL_PATH, config=cfg)
        attacker.cache_positive_text_embeddings([final_prompt])

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Progress: {int(progress * 100)}%")

        attacker.attack(progress_callback=update_progress)
        st.success("Image Generation Completed!")

        st.subheader("Result")
        result_img, score = attacker.evaluate()
        st.image(
            result_img,
            caption=f"CLIP: {score:.4f}",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
