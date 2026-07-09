"""自定义异常模块。

建立项目统一的异常层级体系，所有业务异常继承自 DataAnalysisError，
便于上层统一捕获和日志记录。异常命名遵循大驼峰 + Error 后缀规范。
"""


class DataAnalysisError(Exception):
    """数据分析项目基础异常类。

    所有项目自定义异常的基类，提供统一的错误信息格式。

    Attributes:
        message: 异常描述信息
        detail: 额外的上下文信息，用于辅助排查问题
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        full_message = f"{message}" if detail is None else f"{message} | 详情: {detail}"
        super().__init__(full_message)


class DataLoadError(DataAnalysisError):
    """数据加载异常。

    在数据文件读取、格式解析、路径校验等环节发生错误时抛出。
    """


class DataCleanError(DataAnalysisError):
    """数据清洗异常。

    在去重、异常值处理、类型转换等清洗环节发生错误时抛出。
    """


class DataQualityError(DataAnalysisError):
    """数据质量校验异常。

    在数据质量校验中发现严重问题（如缺失率超标、数据为空）时抛出。
    """


class ConfigError(DataAnalysisError):
    """配置异常。

    在配置项缺失、格式错误、路径不存在等场景下抛出。
    """
