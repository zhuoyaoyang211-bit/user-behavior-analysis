"""日志配置模块。

提供统一的日志器配置，支持控制台和文件双通道输出。
所有业务模块通过 get_logger() 获取日志器实例，保持一致的日志格式。
"""

import logging
import sys
from pathlib import Path

# 日志格式：时间 | 级别 | 模块名 | 行号 | 消息
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 全局标记，避免重复添加 handler
_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "user_behavior",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """获取统一配置的日志器实例。

    Args:
        name: 日志器名称，通常传模块名 __name__
        log_file: 日志文件路径，为 None 时仅输出到控制台
        level: 日志级别，默认为 INFO

    Returns:
        配置好的 logging.Logger 实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("数据加载开始")
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger
