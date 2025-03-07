# A Third-Party Streamlit Demo App of Direct Ascent Synthesis(DAS)

This project is a third-party implementation of Direct Ascent Synthesis (DAS) [1], a method that transforms CLIP models into text-to-image generators without fine-tuning.

It provides an interactive UI for granular parameter tuning, allowing users to experiment with different settings easily.

**Note**: Before I discovered the author's official implementation [2], I developed this version based on their paper [1]. As a result, some differences may exist between this and the original implementation.

## Screenshot

![Image](https://github.com/user-attachments/assets/6798c262-c89c-4a11-b047-a1f4128cfef6)

## About DAS

DAS turns CLIP models into text-to-image generation model without fine-tuning. For more detail, please refer to [1,2].

## Tested Configuration

- OS: Ubuntu 24.04.2 LTS
- GPU: NVIDIA GeForce RTX 4090
- CLIP Model: [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)

## Prerequisites

- [uv](https://docs.astral.sh/uv/concepts/tools/)
- NVIDIA Ampere or newer GPUs (for FlashAttention v2)

## Install

### Install Required Packages

This project requires the `flash-attn` package, which has build dependencies.
To ensure proper installation, we need to run `uv sync` twice:

1. Install standard dependencies:
    ```bash
    uv sync --extra build
    ```
2. Compile and install flash-attn:
    ```bash
    uv sync --extra build --extra compile
    ```

For more details, refer to [official documentation of uv](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation).

### Download CLIP Model

```bash
huggingface-cli download google/siglip-so400m-patch14-384 \
    --local-dir /path/to/your/model/folder \
    --local-dir-use-symlinks False
```

In this project, we assumed the model is stored in `/ml-docker/input/hf`. Make sure to update `src/app.py` with the correct path:

```python
MODEL_PATH = "/ml-docker/input/hf"
```

## Run Streamlit App

```bash
uv run streamlit run src/app.py
```

**Experimental**: If set `CUBLAS_WORKSPACE_CONFIG` environment variable, it uses deterministic algorithm. Note that it **does not ensure reproducibility**, and it **may limit overall performance**.
Please refer to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility for more detail.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run streamlit run src/app.py
# or
CUBLAS_WORKSPACE_CONFIG=:16:8 uv run streamlit run src/app.py
```

## How To Generate

1. adjust parameter (however, initial parameter should be fine).
2. choose one of the sample prompts or type your own prompt
3. push Generate Image

It took around **18 sec/image** for my environment.

## Art Gallery

prompt: `An illustration of Gigantic metallic Mona Lisa attacking the city`

![Image](https://github.com/user-attachments/assets/f248c514-315c-4509-9d4c-fa928a9f9b1a)

prompt: `A photo of Gamma-ray burst into gigantic Tokyo tower in the last day of Earth`

![Image](https://github.com/user-attachments/assets/ca9ac520-2a74-479f-a100-97b0e595cb38)

prompt `A photo of A beautiful Mt. Fuji, a hawk, and an eggplant, in a cyber-punk Tokyo`

![Image](https://github.com/user-attachments/assets/d634a7bc-0417-4d98-9ba9-e4043db3245c)

## Reference

- [1] Direct Ascent Synthesis: Revealing Hidden Generative Capabilities in Discriminative Models, 11 Feb 2025, https://arxiv.org/abs/2502.07753
- [2] https://github.com/stanislavfort/Direct_Ascent_Synthesis