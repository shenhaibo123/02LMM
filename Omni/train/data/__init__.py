"""
Omni 数据模块 —— 数据下载、预处理、格式转换、统计分析与 Dataset 加载。

导出：
    OmniDataset, MultiSourceDataset, build_dataloader — PyTorch Dataset 与加载器
    DATASET_REGISTRY                                  — 各阶段数据集注册表
    DatasetStats, analyze_directory                    — 数据统计分析
"""

from .dataset import OmniDataset, MultiSourceDataset, build_dataloader
from .prepare_data import DATASET_REGISTRY
from .data_stats import DatasetStats, analyze_directory

__all__ = [
    "OmniDataset",
    "MultiSourceDataset",
    "build_dataloader",
    "DATASET_REGISTRY",
    "DatasetStats",
    "analyze_directory",
]
