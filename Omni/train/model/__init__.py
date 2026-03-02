"""
Omni 全模态模型模块 —— 基于 ms-swift 框架的多模态大模型训练核心组件。

导出：
    OmniModelConfig, StageConfig, STAGE_CONFIGS  — 模型与训练阶段配置
    MLPProjector, ConvGMLPProjector              — 模态投影层
    SpeechDecoder                                — 语音解码头
    OmniModel                                    — 全模态模型主类
"""

from .config import OmniModelConfig, StageConfig, STAGE_CONFIGS
from .projectors import MLPProjector, ConvGMLPProjector
from .speech_decoder import SpeechDecoder
from .omni_model import OmniModel

__all__ = [
    "OmniModelConfig",
    "StageConfig",
    "STAGE_CONFIGS",
    "MLPProjector",
    "ConvGMLPProjector",
    "SpeechDecoder",
    "OmniModel",
]
