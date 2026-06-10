"""Neural building blocks. Importing this package registers every mixer and head
into their registries (the import side effect is what populates ``StreamingMixer`` and
``VADHead``), so config-driven construction can look them up by ``kind``."""

from neovad.nn.attention import (
    DiffAttnConfig,
    DiffAttnMixer,
    GQAConfig,
    GQAMixer,
    MLAConfig,
    MLAMixer,
)
from neovad.nn.block import ResidualBlock
from neovad.nn.conv import CausalDepthwiseConv1d
from neovad.nn.convmixer import ConvMixer, ConvMixerConfig
from neovad.nn.deltanet import GatedDeltaNetConfig, GatedDeltaNetMixer
from neovad.nn.gru import GRUConfig, GRUMixer
from neovad.nn.head import (
    AttractorHead,
    AttractorHeadConfig,
    HeadConfig,
    LinearHead,
    LinearHeadConfig,
    SpeechClass,
    VADHead,
)
from neovad.nn.linear_rnn import MinGRUConfig, MinGRUMixer, RGLRUConfig, RGLRUMixer
from neovad.nn.mamba import Mamba2Config, Mamba2Mixer
from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer
from neovad.nn.mlp import SwiGLU
from neovad.nn.norm import RMSNorm, RMSNormGated
from neovad.nn.rope import RotaryEmbedding
from neovad.nn.specaug import SpecAugment, SpecAugmentConfig

__all__ = [
    "AttractorHead",
    "AttractorHeadConfig",
    "CausalDepthwiseConv1d",
    "ConvMixer",
    "ConvMixerConfig",
    "DiffAttnConfig",
    "DiffAttnMixer",
    "GQAConfig",
    "GatedDeltaNetConfig",
    "GatedDeltaNetMixer",
    "GQAMixer",
    "GRUConfig",
    "GRUMixer",
    "HeadConfig",
    "LinearHead",
    "LinearHeadConfig",
    "MLAConfig",
    "MLAMixer",
    "Mamba2Config",
    "Mamba2Mixer",
    "MinGRUConfig",
    "MinGRUMixer",
    "MixerConfig",
    "ModuleState",
    "RGLRUConfig",
    "RGLRUMixer",
    "RMSNorm",
    "RMSNormGated",
    "ResidualBlock",
    "RotaryEmbedding",
    "SpecAugment",
    "SpecAugmentConfig",
    "SpeechClass",
    "StreamingMixer",
    "SwiGLU",
    "VADHead",
]
