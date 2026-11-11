#!/usr/bin/env bash
# AI翻唱工坊 环境安装脚本
#   bash setup.sh          仅安装核心（演示引擎 + 中心声道分离，离线可用）
#   bash setup.sh --full   追加安装高质量组件（Demucs 分离 + OpenVoice 音色克隆 + Seed-VC）
set -uo pipefail
cd "$(dirname "$0")"

if command -v python3.12 >/dev/null 2>&1; then PY=python3.12
elif command -v python3.11 >/dev/null 2>&1; then PY=python3.11
else PY=python3; fi
echo "== 使用 Python: $($PY --version 2>&1) =="

# 双架构版本矩阵（Apple Silicon=arm64 用 numpy 2.x 新 torch/MPS 生态；Intel Mac=x86_64 锁旧矩阵）
# 注意：含 >= 的版本约束必须放数组并整体引用，否则会被 shell 解释成重定向
ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ] && [ "$(uname -s)" = "Darwin" ]; then
  NUMBA_PKGS=('numba==0.60.0' 'llvmlite==0.43.0')   # 有 macOS x86_64 预编译轮子且兼容 numpy 1.x
  NUMPY_PKGS=('numpy==1.26.4')                       # torch 2.2.2（Intel Mac 最后版本）生态
  echo "== 架构: ${ARCH}（Intel Mac：torch 2.2.2 + numpy 1.x 旧矩阵）=="
else
  NUMBA_PKGS=('numba>=0.61' 'llvmlite>=0.44')        # numpy 2.x 支持
  NUMPY_PKGS=('numpy>=2.0')
  echo "== 架构: ${ARCH}（Apple Silicon/其他：最新 torch（MPS）+ numpy 2.x）=="
fi

if [ ! -x .venv/bin/python ]; then
  echo "== 创建虚拟环境 .venv =="
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install -q -U pip wheel

echo "== 安装核心依赖 =="
if pip install -q -r requirements-core.txt; then
  echo "✓ 核心依赖安装完成"
else
  echo "✗ 核心依赖安装失败"; exit 1
fi

if [ "${1:-}" != "--full" ]; then
  echo ""
  echo "核心环境就绪！启动: bash run.sh   （想要高质量引擎请运行: bash setup.sh --full）"
  exit 0
fi

echo ""
echo "== 安装可选高质量组件（某项失败不影响其他功能）=="

echo "-- 1/4 torch + Demucs 人声分离 --"
pip install torch torchaudio && pip install demucs \
  && echo "✓ demucs 安装成功" || echo "✗ demucs 安装失败（分离将使用中心声道兜底）"
echo "-- 预下载 demucs 模型 htdemucs（约 80MB）--"
TORCH_HOME="$PWD/models/torch" "$PY" -c "from demucs.pretrained import get_model; get_model('htdemucs')" \
  && echo "✓ demucs 模型下载完成" || echo "✗ demucs 模型下载失败（首次分离时会自动重试）"

echo "-- 2/4 下载 OpenVoice v2 模型权重（约 500MB，走 hf-mirror 镜像）--"
pip install -q huggingface_hub
"$PY" scripts/download_openvoice.py || echo "✗ OpenVoice 权重下载失败"

echo "-- 3/4 安装 openvoice 包（--no-deps 避免 gradio 依赖冲突）--"
# GitHub 直连常被墙：先试直连，失败则走加速代理拉 tarball
pip install -q "${NUMBA_PKGS[@]}"   # 按架构锁定（见顶部矩阵说明）
ok_pkg=0
for i in 1 2; do
  if pip install -q --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"; then ok_pkg=1; break; fi
  echo "  直连 GitHub 失败，改用代理下载源码…"
  TMP_OV="$(mktemp -d)"
  for proxy in "https://ghproxy.net/https://github.com" "https://gh-proxy.com/https://github.com"; do
    if curl -sL -m 180 "$proxy/myshell-ai/OpenVoice/archive/refs/heads/main.tar.gz" -o "$TMP_OV/ov.tar.gz" \
       && tar -xzf "$TMP_OV/ov.tar.gz" -C "$TMP_OV" 2>/dev/null; then
      OV_DIR=$(find "$TMP_OV" -maxdepth 1 -type d -name "OpenVoice-*" | head -1)
      pip install -q --no-deps "$OV_DIR" && ok_pkg=1 && break
    fi
  done
  [ "$ok_pkg" = "1" ] && break
  sleep 3
done
if [ "$ok_pkg" = "1" ]; then
  # numba/llvmlite/numpy 必须与依赖在同一条命令中锁版本，否则 pip 会拉到破坏 torch 生态的组合
  pip install -q "${NUMBA_PKGS[@]}" "${NUMPY_PKGS[@]}" \
    librosa unidecode pypinyin cn2an eng_to_ipa inflect matplotlib jieba pydub langid \
    faster-whisper whisper-timestamped wavmark \
    && "$PY" -c "from openvoice.api import ToneColorConverter" 2>/dev/null \
    && echo "✓ OpenVoice 可用" || echo "✗ OpenVoice 依赖安装失败（引擎将显示不可用，不影响其他功能）"
else
  echo "✗ OpenVoice 源码获取失败（引擎将显示不可用，不影响其他功能）"
fi

echo "-- 4/4 安装 seed-vc（实验性：仓库 clone + 权重下载）--"
# seedvc_engine 按目录约定查找 models/seedvc/repo + models/seedvc/ready.flag
# 国内网络：优先 codeload 下载源码 tarball（比 git clone 直连稳定），gh 代理兜底
SEEDVC_DIR="$PWD/models/seedvc"
if [ ! -f "$SEEDVC_DIR/repo/app_svc.py" ]; then
  TMP_SC="$(mktemp -d)"
  if curl -sL -m 600 "https://codeload.github.com/Plachtaa/seed-vc/tar.gz/refs/heads/main" -o "$TMP_SC/sc.tar.gz" \
     && tar -tzf "$TMP_SC/sc.tar.gz" >/dev/null 2>&1; then
    tar -xzf "$TMP_SC/sc.tar.gz" -C "$TMP_SC"
    SRC=$(find "$TMP_SC" -maxdepth 1 -type d -name "seed-vc-*" | head -1)
    [ -n "$SRC" ] && rm -rf "$SEEDVC_DIR/repo" && mv "$SRC" "$SEEDVC_DIR/repo"
  fi
  if [ ! -f "$SEEDVC_DIR/repo/app_svc.py" ]; then
    echo "  codeload 失败，尝试 gh 代理…"
    for proxy in "https://ghproxy.net/https://github.com" "https://gh-proxy.com/https://github.com"; do
      if curl -sL -m 300 "$proxy/Plachtaa/seed-vc/archive/refs/heads/main.tar.gz" -o "$TMP_SC/sc.tar.gz" \
         && tar -tzf "$TMP_SC/sc.tar.gz" >/dev/null 2>&1; then
        tar -xzf "$TMP_SC/sc.tar.gz" -C "$TMP_SC"
        SRC=$(find "$TMP_SC" -maxdepth 1 -type d -name "seed-vc-*" | head -1)
        [ -n "$SRC" ] && rm -rf "$SEEDVC_DIR/repo" && mv "$SRC" "$SEEDVC_DIR/repo"
        break
      fi
    done
  fi
  rm -rf "$TMP_SC"
fi
if [ -f "$SEEDVC_DIR/repo/app_svc.py" ]; then
  "$PY" scripts/download_seedvc.py \
    && echo "✓ Seed-VC 权重下载完成（约 2GB）" \
    || echo "✗ Seed-VC 权重下载失败（引擎不可用，不影响其他功能）"
else
  echo "✗ seed-vc 仓库获取失败（可选，跳过）"
fi

echo "-- 5/5 安装引擎运行时依赖（锁定版本矩阵，见 requirements-optional.txt 注释）--"
# 步骤 A：主体依赖一次解出（不含 audio-separator，见下）
pip install -q -r requirements-optional.txt "${NUMPY_PKGS[@]}" \
  && echo "✓ 主体依赖完成" || echo "✗ 主体依赖安装失败（相关引擎自动降级）"
# 步骤 B：audio-separator 与 descript-audiotools 的 protobuf 约束互斥，必须独立安装；
# 实测共存时 dac 推理路径正常（pypi 解析器会打依赖冲突警告，可忽略）
pip install -q audio-separator "protobuf>=4,<5" \
  && "$PY" -c "import audio_separator, dac" 2>/dev/null \
  && echo "✓ 去混响依赖完成（共存验证通过）" \
  || echo "✗ 去混响依赖安装失败（DEREVERB 自动降级，不影响其他功能）"

echo ""
echo "== 安装结果汇总 =="
"$PY" -m app.cli engines
