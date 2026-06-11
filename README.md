# neovad

Small, streaming, CPU-friendly **Voice Activity Detection** with pluggable modern
backbones. Built at [Neovision](https://neovision.fr) to fire **only on the
foreground speaker** — background noise and secondary voices should *not* trigger
activation — and to beat [Silero VAD](https://github.com/snakers4/silero-vad) on
streaming decision latency while staying under a ~2 MB / sub-millisecond-per-chunk
CPU envelope.

Unlike a plain speech/non-speech VAD, neovad emits a per-frame 3-way decision —
`non-speech` / `primary` / `secondary` — so a real-time telephony agent can gate on
the locked primary speaker and ignore an interfering voice in the room or on the line.

## Why

In production telephony the denoiser lets non-stationary noise and background voices
leak through as false negatives, and speaker-isolation models are no longer enough.
A *stateful* model can instead lock onto the current dominant speaker and reject the
rest. neovad makes that the design centre and lets you A/B the architectures that
might deliver it.

## The one pluggable axis: the backbone

Every model is `frontend → N × ResidualBlock(RMSNorm → mixer → RMSNorm → SwiGLU) → head`.
The only thing that varies is the **sequence mixer**, and every mixer satisfies one
hard contract: a parallel causal `forward` (training) and a recurrent `step`
(streaming) that are provably equivalent. Swapping backbones is a one-line config
change.

| backbone   | what it is | streaming state | why it's here |
|------------|------------|-----------------|---------------|
| `gru`      | causal GRU | hidden `(L,B,H)` | proven-simple baseline (Silero-class control) |
| `gqa`      | grouped-query attention + RoPE, sliding window | windowed KV ring buffer | conventional efficient-attention reference |
| `mla`      | DeepSeek multi-head **latent** attention | compressed latent KV cache | smallest KV cache; the DeepSeek-V3 signature |
| `diffattn` | **Differential Attention** (two-softmax) | windowed KV ring buffer | common-mode cancellation kills background/secondary-voice leakage — the primary attention backbone |
| `mamba2`   | **Mamba-2** selective SSD, pure-PyTorch CPU step | `O(1)` SSM + conv state | constant per-step cost over a multi-minute call; its state implicitly tracks *who the dominant speaker is* |

All share the same modern building blocks (RMSNorm, SwiGLU, RoPE) and the same
streaming-state machinery. Add a backbone = subclass `StreamingMixer`, declare a
`MixerConfig`, done.

## Install

```bash
# inference only (torch + numpy + pydantic, CPU-friendly)
pip install "git+https://github.com/NeovisionSAS/neovad.git"

# with the training engine and dataset tooling
pip install "neovad[train] @ git+https://github.com/NeovisionSAS/neovad.git"

# with the Silero comparison + ONNX export harness
pip install "neovad[bench] @ git+https://github.com/NeovisionSAS/neovad.git"
```

The pretrained `mamba2` weights ship **inside the wheel** (`neovad/weights/mamba2.pt`),
so `from_pretrained` works offline right after install — no download step. Checkpoints
published after your installed version are resolved from the HuggingFace Hub
(`NeovisionTech/neovad`) as a fallback.

## Use as a library

### Streaming inference (the deployment path)

```python
from neovad import StreamingVAD

# the pretrained model ships with the package — no download, no config
vad = StreamingVAD.from_pretrained("mamba2", input_sample_rate=8000)  # e.g. 8 kHz telephony

# feed audio chunks as they arrive
for chunk in audio_chunks(hop=160):
    probs = vad.push(chunk)        # foreground-speech probability per 10 ms frame
    if vad.is_speaking:            # hysteresis-smoothed gate
        ...                        # forward audio to STT
vad.reset()                        # at the call boundary
```

`VADModel.from_pretrained(name)` loads weights bundled in the wheel; `from_config(yaml)`
or `load(checkpoint)` build/restore your own.

### Train a model

```python
from neovad import NeoVADConfig, train

cfg = NeoVADConfig.load("configs/mamba2.yaml")
train(cfg)                          # Lightning under the hood; multi-GPU aware
```

### From the CLI

```bash
neovad list-backbones                       # gru gqa mla diffattn mamba2
neovad download --root /disk/manual         # fetch training + eval datasets
neovad train  configs/mamba2.yaml                  # rich TensorBoard logs (audio, mel, figures)
neovad bench  --all-backbones --silero             # latency / size / RTF vs Silero
neovad infer  audio.wav --backbone mamba2
neovad export model.onnx --backbone mamba2         # or --fmt int8 / jit for CPU deploy
```

Training logs to TensorBoard with folded categories: `train/`, `val/` (loss, primary
F1/precision/recall, `secondary_false_fire`), `lr/`, `audio/` (clean primary **and** the
augmented mixture, so you can hear the augmentation), `media/` (PCEN mel + per-frame
label-vs-prediction figures), and `hist/` (weight/grad histograms).

## Benchmarks vs Silero (measured)

All numbers below are produced by this repo's own harness — `neovad eval` scores both
models on the **same audio against the same per-frame labels** (threshold-free ROC-AUC,
validated against sklearn), and `neovad bench` measures latency on the same single CPU
thread. Reproduce with:

```bash
neovad eval  --source voxconverse --silero     # accuracy, neutral external set
neovad eval  --source synthetic   --silero     # accuracy, noisy multi-speaker synthesis
neovad bench --all-backbones --silero          # streaming latency / size / RTF
```

### Accuracy — speech/non-speech, identical audio + labels

neovad is scored by its *any-speech* probability, i.e. on Silero's own task, not its
foreground advantage.

| eval set | neovad (mamba2, bundled) | silero-v6 |
|---|---|---|
| **VoxConverse test** (76 min real conversational, neither model trained on) — ROC-AUC | 0.883 | **0.935** |
| VoxConverse test — best-threshold frame F1 | 0.970 | 0.970 |
| Synthetic noisy multi-speaker (neovad's training distribution) — ROC-AUC | **0.960** | 0.935 |

Honest read: **on the neutral set Silero still leads on ROC-AUC** (its training corpus
is vastly larger and more diverse); frame-F1 at the operating point is tied. An ongoing
ablation study (SpecAugment +1.7 AUC, conversational data with clean-label handling)
is closing the gap — see `docs/ARCHITECTURE.md` §5.

### Foreground-speaker gating — the task Silero cannot do

| metric (synthetic primary + interferers + noise) | neovad | silero-v6 |
|---|---|---|
| primary-speaker frame F1 | **0.955** | n/a (speaker-agnostic) |
| false-fire rate on interferer-only frames | **0.19** | n/a — fires on any voice |

A speaker-agnostic VAD passes every background voice to the STT; neovad's whole point
is that it does not.

### Latency & size — 1 CPU thread

| model | size | streaming RTF | offline RTF (2 s windows) |
|---|---|---|---|
| neovad mamba2 (torch fp32, 30 ms chunks) | 3.6 MB | 0.089 | 0.008 |
| neovad mamba2 (**ONNX fp32**) | 2.7 MB | — | **0.005** |
| neovad mamba2 (torch int8) | **0.96 MB** | ≈ fp32 (size win, not speed) | — |
| neovad gru (torch fp32, 30 ms chunks) | 3.6 MB | **0.046** | — |
| silero-v6 (fused JIT, 32 ms chunks) | 2.2 MB | **0.009** | 0.009 |

Honest read: in **offline/batch** mode neovad's ONNX export is **~2× faster than
Silero** (RTF 0.005 vs 0.009) and the int8 export is **half Silero's size**. In
**streaming** mode Silero's fused graph still wins per-chunk (0.009 vs 0.046–0.089);
neovad is comfortably real-time (≥11×) and emits decisions at 10 ms granularity vs
Silero's 32 ms. The remaining streaming gap is per-call overhead on a tiny eager
graph — the documented fix is exporting the `step` graph itself.

## Datasets

Training data is synthesized on the fly (no pre-rendered set): one **clean primary**
speaker (LibriSpeech, Common Voice fr/en) + 1–3 interfering voices (AMI real meeting
speech, MUSAN babble — crosstalk-y corpora live in the *interferer* pool only, so they
never pollute the primary labels) + MUSAN noise/music + VocalSound laughs/coughs as
hard non-speech + room impulse response + telephony codec degradation. Labels derive
from the clean primary reference. `neovad download` fetches everything into
`/disk/manual` (direct OpenSLR archives + HuggingFace sources materialized to 16 kHz
FLAC). All sources are commercial-safe (CC BY / CC0 / CC BY-SA).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design and the research
behind every choice.

## License

Apache-2.0 © Neovision
