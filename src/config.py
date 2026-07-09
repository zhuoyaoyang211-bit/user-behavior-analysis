"""项目配置管理模块。

使用 dataclass 集中管理数据路径、分块大小、清洗参数等配置项。
所有模块通过导入 ProjectConfig 获取统一配置，避免散落的硬编码。

使用方式:
    from config import get_config
    config = get_config()
    chunk_size = config.chunk_size
"""

from dataclasses import dataclass, field
from pathlib import Path

from common.exceptions import ConfigError


@dataclass
class ProjectConfig:
    """项目全局配置。

    集中管理数据文件路径、处理参数、输出路径等配置项。
    使用 dataclass 确保类型安全和默认值管理。

    Attributes:
        project_root: 项目根目录路径
        raw_data_path: 原始 CSV 数据文件路径
        cleaned_data_path: 清洗后数据的 Parquet 输出路径
        quality_report_path: 数据质量报告输出路径
        chunk_size: 分块读取的行数，控制内存占用
        log_path: 日志文件输出路径
        dedup_subset: 去重时依据的列名列表（用户-商品-行为-时间四元组）
    """

    # 路径配置 — 全部使用相对路径，相对于项目根目录
    # project_root 通过本文件位置推导，不硬编码绝对路径，保证项目可移植
    project_root: Path = Path(__file__).parent.parent
    raw_data_path: Path = Path("user_behavior_processed.csv")
    cleaned_data_path: Path = Path("output/cleaned_data.parquet")
    quality_report_path: Path = Path("output/quality_report.txt")
    log_path: Path = Path("output/project.log")

    # 数据处理参数
    chunk_size: int = 1_000_000
    dedup_subset: list[str] = field(
        default_factory=lambda: ["user_id", "item_id", "behavior_type", "time"]
    )

    # 数据类型优化映射：减少内存占用
    dtype_map: dict[str, str] = field(
        default_factory=lambda: {
            "user_id": "int32",
            "item_id": "int32",
            "item_category": "int16",
            "behavior_type": "int8",
        }
    )

    def __post_init__(self) -> None:
        """校验原始数据文件是否存在，并确保输出目录可用。"""
        # 切换工作目录到项目根，确保所有相对路径生效
        import os

        os.chdir(self.project_root)

        # 确保输出目录存在
        Path("output").mkdir(exist_ok=True)

        if not self.raw_data_path.exists():
            raise ConfigError(
                "原始数据文件不存在",
                detail=f"期望路径: {self.raw_data_path}",
            )


# 全局配置单例，首次调用时初始化
_config_instance: ProjectConfig | None = None


def get_config() -> ProjectConfig:
    """获取项目全局配置单例。

    Returns:
        ProjectConfig 实例

    Raises:
        ConfigError: 原始数据文件不存在时抛出
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ProjectConfig()
    return _config_instance
