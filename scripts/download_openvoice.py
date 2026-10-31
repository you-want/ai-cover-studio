"""下载 OpenVoice V2 converter 权重（自动尝试镜像与官方源）。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def download(endpoint: str) -> bool:
    os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import snapshot_download
        print(f"尝试从 {endpoint} 下载 OpenVoiceV2 converter 权重…")
        snapshot_download(repo_id="myshell-ai/OpenVoiceV2",
                          allow_patterns=["converter/*"],
                          local_dir=str(config.OPENVOICE_DIR / "OpenVoiceV2"),
                          max_workers=4)
        d = config.OPENVOICE_DIR / "OpenVoiceV2" / "converter"
        ok = (d / "checkpoint.pth").exists() and (d / "config.json").exists()
        if ok:
            print(f"✓ 已下载到 {d}")
        return ok
    except Exception as e:
        print(f"  [{endpoint}] 下载失败: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    primary = os.environ.get("HF_ENDPOINT", config.HF_ENDPOINT)
    ok = download(primary)
    if not ok:
        for fallback in ("https://hf-mirror.com", "https://huggingface.co"):
            if fallback != primary:
                ok = download(fallback) or ok
    sys.exit(0 if ok else 1)
