"""数据加载模块。

负责大规模用户行为数据的分块读取、类型压缩与格式转换。
原始 CSV 约 1225 万行、492MB，采用分块读取策略控制内存占用。

主要功能:
    - 分块读取 CSV 数据（每块 100 万行）
    - 读取时直接指定 dtype 压缩内存
    - 将清洗后的数据保存为 Parquet 格式
    - 从 Parquet 快速加载后续处理

典型使用流程:
    loader = DataLoader(config)
    for chunk in loader.load_csv_chunks():
        # 对每个分块进行清洗处理
        process(chunk)
"""

from pathlib import Path

import pandas as pd

from common.exceptions import DataLoadError
from common.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """大数据文件加载器。

    封装分块读取、类型压缩、格式转换等数据加载逻辑，
    避免一次性读取大文件导致内存溢出。

    Attributes:
        raw_data_path: 原始 CSV 文件路径
        cleaned_data_path: 清洗后 Parquet 文件路径
        chunk_size: 每次读取的行数
        dtype_map: 列名到数据类型的映射
    """

    def __init__(self, config) -> None:
        """初始化数据加载器。

        Args:
            config: ProjectConfig 实例，提供路径和参数配置
        """
        self.raw_data_path: Path = config.raw_data_path
        self.cleaned_data_path: Path = config.cleaned_data_path
        self.chunk_size: int = config.chunk_size
        self.dtype_map: dict[str, str] = config.dtype_map

    def load_csv_chunks(self) -> pd.io.parsers.TextFileReader:
        """分块读取 CSV 文件，返回可迭代的分块读取器。

        使用 pandas 的 chunksize 参数实现惰性读取，
        每次只在内存中保留一个分块，适合处理超大数据集。

        Returns:
            TextFileReader 迭代器，每次迭代返回一个 DataFrame 分块

        Raises:
            DataLoadError: 文件读取失败时抛出

        Examples:
            >>> loader = DataLoader(config)
            >>> for chunk in loader.load_csv_chunks():
            ...     print(chunk.shape)
        """
        logger.info(
            "开始分块读取 CSV: %s, 每块 %d 行",
            self.raw_data_path,
            self.chunk_size,
        )
        try:
            reader = pd.read_csv(
                self.raw_data_path,
                chunksize=self.chunk_size,
                dtype=self.dtype_map,
                parse_dates=["time"],
                date_format="%Y-%m-%d %H",
            )
            return reader
        except (FileNotFoundError, pd.errors.ParserError) as e:
            raise DataLoadError("CSV 文件读取失败", detail=str(e)) from e

    def load_full_csv(self) -> pd.DataFrame:
        """一次性读取完整 CSV（仅在内存充足时使用）。

        Returns:
            完整的 DataFrame

        Raises:
            DataLoadError: 读取失败时抛出

        Note:
            对于 1225 万行数据，此方法会占用约 1GB+ 内存，
            生产环境优先使用 load_csv_chunks()。
        """
        logger.warning("正在一次性加载完整 CSV，可能占用大量内存")
        try:
            df = pd.read_csv(
                self.raw_data_path,
                dtype=self.dtype_map,
                parse_dates=["time"],
                date_format="%Y-%m-%d %H",
            )
            logger.info("完整加载完成, 行数: %d, 列数: %d", df.shape[0], df.shape[1])
            return df
        except (FileNotFoundError, pd.errors.ParserError) as e:
            raise DataLoadError("CSV 完整加载失败", detail=str(e)) from e

    def save_parquet(self, df: pd.DataFrame, path: Path | None = None) -> Path:
        """将 DataFrame 保存为 Parquet 格式。

        Parquet 是列式存储格式，相比 CSV 读取速度快 5-10 倍，
        文件体积减小约 70%，且自动保留数据类型信息。

        Args:
            df: 待保存的 DataFrame
            path: 输出路径，为 None 时使用默认 cleaned_data_path

        Returns:
            实际保存的文件路径

        Raises:
            DataLoadError: 保存失败时抛出
        """
        save_path = path if path is not None else self.cleaned_data_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("保存 Parquet 文件: %s, 行数: %d", save_path, len(df))
        try:
            df.to_parquet(save_path, engine="pyarrow", index=False)
            logger.info("Parquet 保存成功")
            return save_path
        except Exception as e:
            raise DataLoadError("Parquet 保存失败", detail=str(e)) from e

    def load_parquet(self, path: Path | None = None) -> pd.DataFrame:
        """从 Parquet 文件加载数据。

        Args:
            path: Parquet 文件路径，为 None 时使用默认 cleaned_data_path

        Returns:
            加载的 DataFrame

        Raises:
            DataLoadError: 文件不存在或读取失败时抛出
        """
        load_path = path if path is not None else self.cleaned_data_path
        if not load_path.exists():
            raise DataLoadError("Parquet 文件不存在", detail=str(load_path))
        logger.info("从 Parquet 加载数据: %s", load_path)
        try:
            df = pd.read_parquet(load_path, engine="pyarrow")
            logger.info("Parquet 加载完成, 行数: %d, 列数: %d", df.shape[0], df.shape[1])
            return df
        except Exception as e:
            raise DataLoadError("Parquet 加载失败", detail=str(e)) from e
