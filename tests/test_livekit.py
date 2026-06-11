import asyncio

import numpy as np
import pytest
import torch

pytest.importorskip("livekit.agents")

from livekit import rtc  # noqa: E402
from livekit.agents.vad import VADEventType  # noqa: E402

from neovad.config import ModelConfig  # noqa: E402
from neovad.livekit import VAD  # noqa: E402
from neovad.models.vad import VADModel  # noqa: E402


def make_frames(wav_16k: np.ndarray, rate: int = 48000, frame_ms: int = 10):
    # Upsample to the LiveKit room rate and slice into AudioFrames.
    n48 = int(len(wav_16k) * rate / 16000)
    wav = np.interp(np.linspace(0, len(wav_16k) - 1, n48), np.arange(len(wav_16k)), wav_16k)
    i16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    step = rate * frame_ms // 1000
    return [
        rtc.AudioFrame(
            data=i16[i : i + step].tobytes(),
            sample_rate=rate,
            num_channels=1,
            samples_per_channel=step,
        )
        for i in range(0, len(i16) - step + 1, step)
    ]


async def collect_events(vad, frames):
    stream = vad.stream()
    for f in frames:
        stream.push_frame(f)
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    return events


def test_livekit_stream_event_machine():
    # Force the gate with a deterministic stand-in model: the probability source is
    # the audio energy itself, so loud tone -> speech, silence -> non-speech.
    torch.manual_seed(0)
    model = VADModel(ModelConfig(dim=32, depth=1, mixer={"kind": "gru"}))
    vad = VAD.load(
        model=model,
        gate="any_speech",
        activation_threshold=0.0,
        min_speech_duration=0.03,
        min_silence_duration=0.2,
    )
    rng = np.random.default_rng(0)
    wav = np.concatenate([np.zeros(8000), 0.5 * rng.standard_normal(16000), np.zeros(16000)])
    events = asyncio.run(collect_events(vad, make_frames(wav.astype(np.float32))))

    types = [e.type for e in events]
    assert VADEventType.INFERENCE_DONE in types
    # threshold 0.0 means everything is speech: exactly one start, no end (gate held)
    assert types.count(VADEventType.START_OF_SPEECH) == 1
    start = next(e for e in events if e.type == VADEventType.START_OF_SPEECH)
    assert start.speaking and start.frames and start.frames[0].sample_rate == 48000
    # INFERENCE_DONE cadence: one event per 30 ms of audio (2.5 s -> ~83)
    inf = [e for e in events if e.type == VADEventType.INFERENCE_DONE]
    assert 70 <= len(inf) <= 90
    assert all(0.0 <= e.probability <= 1.0 for e in inf)
    # timestamps advance monotonically on the inference clock
    ts = [e.timestamp for e in inf]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))


def test_livekit_capabilities_and_metadata():
    model = VADModel(ModelConfig(dim=32, depth=1, mixer={"kind": "gru"}))
    vad = VAD.load(model=model)
    assert abs(vad.capabilities.update_interval - 0.03) < 1e-9  # 3 x 10 ms hops
    assert vad.model.startswith("neovad-")
    assert vad.provider == "neovision"
