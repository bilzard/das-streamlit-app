#!/bin/env python
import argparse
import getpass
import os

from huggingface_hub import login, snapshot_download

HF_LOCAL_CACHE_DIR = "/ml-docker/input/hf"


def parse_args():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face")
    parser.add_argument("repo_id", type=str, help="Repository ID on Hugging Face")
    parser.add_argument(
        "--login", action="store_true", help="Login to Hugging Face with access token"
    )
    return parser.parse_args()


def main():
    args: object = parse_args()

    token = os.getenv("HF_ACCESS_TOKEN")
    if token is None and args.login:
        token = getpass.getpass(prompt="Please enter your Hugging Face access token: ")

    if args.login:
        login(token=token)

    print(f"Downloading repository '{args.repo_id}' from Hugging Face...")
    local_dir = snapshot_download(
        repo_id=args.repo_id,
        local_dir=f"{HF_LOCAL_CACHE_DIR}/{args.repo_id}",
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
