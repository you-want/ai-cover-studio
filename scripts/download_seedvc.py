"""下载 Seed-VC 所需模型权重并做一次最小推理验证。

背景：venv 里 transformers 4.x 要求 huggingface_hub<1.0，而旧版 hub 在
hf-mirror 冷缓存时会 308 跳转 huggingface.co 导致下载失败；新版 hub (>=1.0)
下载正常但与 transformers 不兼容。因此分两阶段：

  1) 预热（隔离垫片 PYTHONPATH=hub_shim，只用 huggingface_hub 下载文件，
     不 import torch/transformers，不受版本检查影响）；
  2) 验证（不带垫片，走真实运行时环境 hub 0.36 + transformers 4.57，
     全部命中本地缓存跑一次 diffusion-steps=2 的自转换）。

成功后在 models/seedvc/ 写 ready.flag。运行时权重全部在本地，
不再依赖网络与镜像状态。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

REPO_DIR = config.SEEDVC_DIR / "repo"
SHIM_DIR = config.SEEDVC_DIR / "hub_shim"
FLAG = config.SEEDVC_DIR / "ready.flag"
HF_CACHE = REPO_DIR / "checkpoints" / "hf_cache"   # inference.py 里 HF_HUB_CACHE 的相对路径落点

# VC 模式（f0_condition=False）与 SVC 歌唱模式（f0_condition=True）需要的文件
DIT_FILES = [("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
             ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
             ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
             ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
             # SVC 模式的 F0 提取器（RMVPE）
             ("lj1995/VoiceConversionWebUI", "rmvpe.pt")]
# 快照仓库 → 下载过滤（None 表示用通用排除表；bigvgan 仓库含 ~3GB 训练优化器
# 状态文件，推理只需要 config + generator，必须显式收窄）
SNAPSHOT_SPECS = [
    ("openai/whisper-small", None),
    ("nvidia/bigvgan_v2_22khz_80band_256x", ["config.json", "bigvgan_generator.pt"]),
]
# 排除冗余权重格式：transformers 优先用 safetensors，pytorch_model.bin 可省 ~900MB
EXCLUDE_PATTERNS = ["pytorch_model.bin", "*.h5", "*.msgpack", "rust_model.ot",
                    "flax_model*", "tf_model*", "*.onnx"]

DOWNLOAD_ONLY = r"""
import os, sys
sys.path.insert(0, os.environ["ACS_SHIM"])   # 下载垫片：新版 hub（仅本子进程生效）
from huggingface_hub import hf_hub_download, snapshot_download

for repo, fn in [("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
                 ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
                 ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
                 ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
                 ("lj1995/VoiceConversionWebUI", "rmvpe.pt")]:
    print(f"· 下载 {repo}/{fn} …", flush=True)
    hf_hub_download(repo_id=repo, filename=fn, cache_dir=os.environ["ACS_CKPT_DIR"])

# 按 config yml 动态快照两个模式各自的 vocoder（44k SVC 用 44k bigvgan）
import yaml
seen = set()
for cfg in ["config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"]:
    import glob as _glob
    hits = _glob.glob(os.path.join(os.environ["ACS_CKPT_DIR"], "models--Plachta--Seed-VC",
                                   "snapshots", "*", cfg))
    if not hits:
        continue
    with open(hits[0]) as f:
        params = yaml.safe_load(f)
    voc = (params.get("vocoder") or {}).get("name")
    tok = (params.get("speech_tokenizer") or {}).get("name")
    for name in (voc, tok):
        if name and "/" in name and name not in seen:
            seen.add(name)
            print(f"· 快照下载 {name}（config 依赖）…", flush=True)
            snapshot_download(repo_id=name, cache_dir=os.environ["ACS_HF_CACHE"],
                              allow_patterns=["config.json", "bigvgan_generator.pt"]
                              if "bigvgan" in name else None)

for repo, allow in [("openai/whisper-small", None),
                    ("nvidia/bigvgan_v2_22khz_80band_256x",
                     ["config.json", "bigvgan_generator.pt"])]:
    if allow:
        print(f"· 快照下载 {repo}（仅 {allow}）…", flush=True)
        snapshot_download(repo_id=repo, cache_dir=os.environ["ACS_HF_CACHE"],
                          allow_patterns=allow)
    else:
        print(f"· 快照下载 {repo} …", flush=True)
        snapshot_download(repo_id=repo, cache_dir=os.environ["ACS_HF_CACHE"],
                          ignore_patterns=os.environ["ACS_EXCLUDE"].split(","))
print("DOWNLOAD_OK", flush=True)
"""


def ensure_shim() -> bool:
    if (SHIM_DIR / "huggingface_hub").exists():
        return True
    print("· 安装下载垫片（隔离目录内的新版 huggingface_hub，不影响运行环境）…")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--target", str(SHIM_DIR), "huggingface_hub==2.3.0"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ 垫片安装失败: {r.stderr.strip()[-300:]}")
        return False
    return True


def prewarm_download() -> bool:
    """阶段 1：垫片环境纯下载（无需 torch）。"""
    if not ensure_shim():
        return False
    env = dict(os.environ)
    env.update({
        "HF_ENDPOINT": config.HF_ENDPOINT,
        "HF_HUB_DISABLE_XET": "1",
        "PYTHONPATH": str(SHIM_DIR),
        "ACS_SHIM": str(SHIM_DIR),
        "ACS_CKPT_DIR": str(REPO_DIR / "checkpoints"),
        "ACS_HF_CACHE": str(HF_CACHE),
        "ACS_EXCLUDE": ",".join(EXCLUDE_PATTERNS),
    })
    proc = subprocess.run([sys.executable, "-c", DOWNLOAD_ONLY],
                          env=env, cwd=str(REPO_DIR))
    return proc.returncode == 0


def validate() -> bool:
    """阶段 2：真实运行时（无垫片）跑一次最小自转换，验证本地缓存完整。"""
    env = dict(os.environ)
    env["HF_ENDPOINT"] = config.HF_ENDPOINT
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    demo = config.BASE_DIR / "demo"
    src = demo / "demo_ref.wav"
    if not src.exists():
        print(f"✗ 缺少验证音频 {src}")
        return False
    out_dir = config.SEEDVC_DIR / "_prewarm_out"
    cmd = [sys.executable, str(REPO_DIR / "inference.py"),
           "--source", str(src), "--target", str(src),
           "--output", str(out_dir), "--diffusion-steps", "2", "--fp16", "false"]
    print("· 最小推理验证（CPU，diffusion-steps=2，数分钟）…")
    proc = subprocess.run(cmd, cwd=str(REPO_DIR), env=env)
    ok = proc.returncode == 0 and any(out_dir.glob("*.wav"))
    shutil.rmtree(out_dir, ignore_errors=True)
    return ok


def main() -> int:
    if not REPO_DIR.exists():
        print("✗ 未找到 seed-vc 仓库（bash setup.sh --full 安装）")
        return 1
    if FLAG.exists():
        print(f"✓ 权重已就绪（{FLAG}），如需重新下载请删除该文件")
        return 0

    print(f"· 阶段 1/2：下载权重（约 1.5GB，走 {config.HF_ENDPOINT}）…")
    if not prewarm_download():
        print("✗ 权重下载失败（引擎保持实验性，转换失败会自动回退，不影响其他功能）")
        return 1
    print("✓ 权重下载完成")

    if not validate():
        print("✗ 最小推理验证失败（详见上方日志）")
        return 1

    FLAG.write_text("seed-vc weights downloaded & runtime-verified\n", encoding="utf-8")
    print(f"✓ Seed-VC 权重就绪并通过运行时验证（{REPO_DIR / 'checkpoints'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
