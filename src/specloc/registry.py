"""注册 SpecLoc 的数据集指标与实验追溯钩子。"""

from .engine import ExperimentMetadataHook
from .evaluation import AITODMetric

__all__ = ['AITODMetric', 'ExperimentMetadataHook']
