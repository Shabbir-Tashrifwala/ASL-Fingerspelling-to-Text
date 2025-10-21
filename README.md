[![Hugging Face – Model](https://img.shields.io/badge/HuggingFace-Model-yellow.svg)](https://huggingface.co/Shabbir1/asl-fingerspelling-text)
# ASL Fingerspelling → Text (Transformer + MediaPipe)

Real-time American Sign Language (ASL) **fingerspelling** recognition that streams webcam video, extracts landmarks with **MediaPipe**, and decodes character sequences with a **Transformer encoder–decoder**. Includes a simple CLI webcam app.

---

## Table of Contents
- [Demo (Webcam)](#demo-webcam)
- [Project Highlights](#project-highlights)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data & Feature Engineering](#data--feature-engineering)
- [Model & Training](#model--training)
- [Real-Time Inference](#real-time-inference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Download weights once

curl -L -o assets/model_phase2_best.pt "https://github.com/Shabbir-Tashrifwala/asl-fingerspelling/releases/download/v1.0.0/model_phase2_best.pt"

---

## Demo (Webcam)

**CLI: **

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
# or: source .venv/bin/activate                    # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt

# Put your trained weights here:
# assets/model_phase2_best.pt
python -m aslfs_rt.app_cli --bundle . --source 0 --flip
```
- `--source 0` = default webcam (try `1` if you have multiple).
- Press **ESC** to quit.

---

## Project Highlights

- **Landmarks, not pixels.** MediaPipe provides 3D pose and hand keypoints; I normalize and feed sequences to a Transformer.
- **Tight, deterministic feature spec (F=144).**
  - Pose joints: **[11, 12, 13, 14, 15, 16]** (L/R shoulder, elbow, wrist)
  - **Left hand:** 21 landmarks (0–20)
  - **Right hand:** 21 landmarks (0–20)
  - **Order:** pose → left hand → right hand; each as (x, y, z) ⇒ **(6 + 21 + 21) × 3 = 144** features per frame.
- **Robust per-frame normalization.**
  - **Center:** mid-shoulders if pose present; otherwise mid-wrists.
  - **Scale:** shoulder distance if pose present; otherwise wrist distance (hand span).
  - Sanitization: **NaN/±Inf → 0**, clamp to **[-1e6, 1e6]**, `float32`.
- **No sample dropping.** I **interpolate within sequences** (linear + ffill/bfill) and train with masks instead of discarding data.
- **Transformer encoder–decoder** with sinusoidal PEs; **greedy decoding** for validation and streaming.
- **Temporal smoothing** in real time: moving average over next-token probs + stability gate to kill “jitter”.

**Result:** **CER ≈ 0.36** on validation in ~**3.5 hours** (2× T4).

---

## Repo Structure

```
asl-fingerspelling/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ assets/
│  ├─ char2idx.json
│  └─ idx2char.json
├─ notebooks/
│  └─ asl-fingerspelling-project.ipynb
└─ src/
   └─ aslfs_rt/
      ├─ __init__.py
      ├─ config.py
      ├─ model.py
      ├─ preprocess.py
      ├─ decoder.py
      └─ app_cli.py
                  
```
---

## Installation

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip wheel setuptools
# CPU-only torch (simplest):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**macOS/Linux**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 
```
numpy
opencv-python
mediapipe==0.10.14
torch
```

---

## Quick Start

1. Place **`model_phase2_best.pt`** in the `assets/` folder.
2. Activate your venv and install requirements.
3. Run:
   ```bash
   python -m aslfs_rt.app_cli --bundle . --source 0 --flip
   ```
4. Make the letters with your hand; the **predicted text** appears on the stream.

If you use a `src/` layout:
```bash
# macOS/Linux
export PYTHONPATH=src:$PYTHONPATH
# Windows PowerShell
$env:PYTHONPATH="src;$env:PYTHONPATH"
python -m aslfs_rt.app_cli --bundle . --source 0 --flip
```

---

## Data & Feature Engineering

**Dataset:** Google - American Sign Language Fingerspelling Recognition. I used `train_landmarks/`, `supplemental_landmarks/`, `train.csv`, and `character_to_prediction_index.json`.

**Pipeline key points**
- **Column selection:** auto-discovers available `(x, y, z)` triplets for:
  - Pose joints **11–16**; hands **0–20** (both).
- **Ordering (strict):** pose → left hand → right hand; each as (x, y, z).
- **Interpolation per sequence**: `linear` + `ffill`/`bfill` to fill gaps; **no dropping**.
- **Normalization per frame:**  
  - **Center:** mid(shoulder-11, shoulder-12) if pose exists; else mid(wrist-0 left, wrist-0 right).  
  - **Scale:** shoulder distance if pose exists; else wrist distance (hand span).  
  - **Sanitize:** `np.nan_to_num` (NaN/±Inf→0), `np.clip` to [-1e6, 1e6], `float32`.
- **Artifacts:**
  - `sequences/<sequence_id>.npz` with `frames: (T, 144)` and `frame_index`.
  - `manifest.csv` joining with `train.csv` to attach `phrase`.
- **Tokenizer (character-level):**
  - Start from competition map; **ensure `PAD=0`**, add **`<bos>`** and **`<eos>`**.
  - Functions: `text_to_ids` / `ids_to_text`.

---

## Model & Training

**Architecture**
- **Input:** `(T, F=144)` → linear **input projection** to `d_model`.
- **Positional encodings:** **sinusoidal** on both encoder **src** and decoder **tgt**.
- **Transformer (PyTorch `nn.Transformer`, `batch_first=True`):**
  - `d_model=512`, `nhead=8`
  - `num_encoder_layers=6`, `num_decoder_layers=6`
  - `dim_feedforward=2048`, `dropout=0.1`
- **Decoder embedding:** `nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)`
- **Output head:** `Linear(d_model → vocab_size)`

**Loss / Optimization**
- **Loss:** `CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=0.1)` 
- **Optimizer:** `AdamW(lr=2e-4, weight_decay=0.01, betas=(0.9, 0.98))`
- **Schedule:** `OneCycleLR(max_lr=2e-4, epochs=30, steps_per_epoch=len(dl_train))`
- **AMP:** `torch.cuda.amp.GradScaler(enabled=cuda)`
- **Grad clipping:** `clip_norm=1.0`
- **Teacher forcing:** standard `y_in = y[:, :-1]`, predict `y_tgt = y[:, 1:]`

**Data Loading**
- **Bucketed batching** by sequence length (reduces padding).
- `BASE_BATCH=64` scaled by GPU count, `num_workers=4`, `prefetch_factor=4`, `pin_memory=True`.
- Masks:
  - `src_key_padding_mask` for frames
  - `tgt_key_padding_mask` for tokens
  - Causal `tgt_mask` via square subsequent mask.

**Validation**
- **Greedy decode** until `<eos>`, then compute **CER** (Levenshtein distance on characters).

**Checkpoint**
- Saved at: `assets/model_phase2_best.pt` with:
  - `{"model": state_dict, "char2idx", "idx2char", "config": {"in_feat": 144, "vocab_size": V}}`

**Observed result**
- **CER ≈ 0.36** after ~**3.5 hours** on **2× T4**.

---

## Real-Time Inference 

**Runtime steps loop**
1. Capture frame from webcam.
2. **MediaPipe Pose + Hands** → landmarks.
3. **Same normalization** as training (center/scale + sanitize + strict feature order).
4. Maintain a **sliding window** (default **64** frames).
5. Encode and **greedy decode** the next token.
6. Apply **temporal smoothing** and **stability gate** before committing characters.
7. Draw text on the video.

**Temporal smoothing**
- Moving average over last **K=5** next-token probability vectors (EMA optional).
- **Commit gate:** probability ≥ **0.60** for **≥ 3** consecutive frames, ignore `<pad>/<bos>` and de-dup repeats.
- Produces **stable text** with minimal flicker.

**Files**
- `aslfs_rt/preprocess.py` — MediaPipe init + extraction + normalization (exact match to training).
- `aslfs_rt/decoder.py` — `TemporalSmoother`, `RealTimeDecoder`.
- `aslfs_rt/app_cli.py` — simple webcam app (ESC quits).

---

## Troubleshooting

- **“Could not open source”**: the webcam is busy; close other apps or try `--source 1`.
- **Windows install looks stuck**: install Torch first, then `mediapipe`/`opencv`:
  ```powershell
  pip install --upgrade pip wheel setuptools
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install --prefer-binary mediapipe==0.10.14 opencv-python numpy
  ```
- **Mismatched features**: ensure your weights were trained with **F=144** features (pose 11–16 + both hands 0–20 with (x,y,z) each).
- **Slow CPU performance**: it runs on CPU, but GPU is recommended.

---

## License

MIT (see `LICENSE`). MediaPipe is licensed separately; please respect its license.
