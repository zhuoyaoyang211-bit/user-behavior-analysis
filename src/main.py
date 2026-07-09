"""数据工程阶段主入口（第一阶段）。

串联数据加载 → 清洗 → 质量校验 → Parquet 持久化的完整流程。
运行此文件即可完成第一阶段的数据工程任务。

使用方式:
    cd src/
    python main.py
"""

import sys
from pathlib import Path

# 将 src 目录加入 Python 路径，支持直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from common.exceptions import DataAnalysisError
from common.logger import get_logger
from config import get_config
from data_cleaner import DataCleaner
from data_loader import DataLoader
from data_quality import DataQualityChecker

logger = get_logger("main")


def run_data_engineering() -> None:
    """执行第一阶段数据工程完整流程。

    流程步骤:
        1. 分块加载原始 CSV 数据
        2. 逐块清洗（完整性校验 + 字段筛选 + 重复检测 + 业务规则异常过滤 + 时间标准化）
        3. 合并清洗结果
        4. 全量数据完整性校验（兜底：验证跨块的商品-类目一致性）
        5. 全局统计异常检测（IQR 法则 + 爬虫画像验证）
           - 统计异常用户经爬虫画像区分：爬虫删除，重度买家打标记保留
        6. 执行数据质量校验，生成报告
        7. 保存为 Parquet 格式

    注: 四元组重复检测保留在流程中，但经确认重复为合理现象，不删除重复记录。

    Raises:
        DataAnalysisError: 任何阶段发生不可恢复错误时抛出
    """
    config = get_config()
    logger.info("=" * 60)
    logger.info("第一阶段：数据工程体系搭建")
    logger.info("=" * 60)

    # 初始化各组件
    loader = DataLoader(config)
    cleaner = DataCleaner(dedup_subset=config.dedup_subset)
    checker = DataQualityChecker(report_path=config.quality_report_path)

    cleaned_chunks: list[pd.DataFrame] = []
    total_stats: dict[str, int] = {
        "original_rows": 0,
        "after_dedup": 0,
        "after_behavior_filter": 0,
        "after_time_clean": 0,
        "final_rows": 0,
    }

    try:
        # 步骤 1 + 2: 分块加载并逐块清洗（业务规则层）
        reader = loader.load_csv_chunks()
        for i, chunk in enumerate(reader, start=1):
            logger.info("--- 处理第 %d 块 ---", i)
            cleaned_chunk, stats = cleaner.clean(chunk)
            cleaned_chunks.append(cleaned_chunk)
            for key in total_stats:
                total_stats[key] += stats[key]

        logger.info(
            "全部分块清洗完成: 原始 %d → 清洗后 %d",
            total_stats["original_rows"],
            total_stats["final_rows"],
        )

        # 步骤 3: 合并所有分块
        df_cleaned = pd.concat(cleaned_chunks, ignore_index=True)
        logger.info("合并完成, 总行数: %d", len(df_cleaned))

        # 步骤 3.5: 全量重复检测（分块检测只能发现块内重复，此处统计跨块重复）
        before_cross_check = len(df_cleaned)
        cross_dup_count = df_cleaned.duplicated(
            subset=config.dedup_subset, keep="first"
        ).sum()
        total_stats["cross_dup_count"] = cross_dup_count
        logger.info(
            "全量重复检测: 跨块重复 %d 条, 保留全部记录不删除",
            cross_dup_count,
        )

        # 步骤 4: 全量数据完整性校验（兜底）
        # 分块校验只能发现块内不一致，此处验证跨块的商品-类目一致性
        cleaner.validate_data_integrity(df_cleaned)

        # 步骤 5: 全局统计异常检测（IQR 法则 + 爬虫画像验证）
        # 必须在合并后执行：单块内的分布不具全局代表性
        # 统计异常用户经爬虫画像区分：爬虫删除，真实重度买家打标记保留
        df_cleaned, outlier_stats = cleaner.handle_statistical_outliers(
            df_cleaned, method="iqr"
        )
        logger.info(
            "统计异常处理: 异常用户 %d 名 → 爬虫删除 %d 名, "
            "重度买家标记 %d 名保留, 删除记录 %d 条",
            outlier_stats["outlier_user_count"],
            outlier_stats["crawler_count"],
            outlier_stats["power_user_count"],
            outlier_stats["removed_rows"],
        )

        # 步骤 6: 数据质量校验
        cleaning_stats = {
            **total_stats,
            "outlier_user_count": outlier_stats["outlier_user_count"],
            "crawler_count": outlier_stats["crawler_count"],
            "power_user_count": outlier_stats["power_user_count"],
            "crawler_removed_rows": outlier_stats["removed_rows"],
        }
        report = checker.run_all_checks(df_cleaned, cleaning_stats=cleaning_stats)
        print(report)

        # 步骤 7: 保存为 Parquet
        loader.save_parquet(df_cleaned)
        logger.info("第一阶段数据工程全部完成")

    except DataAnalysisError as e:
        logger.error("数据工程流程失败: %s", e)
        raise
    finally:
        # 释放内存
        cleaned_chunks.clear()


if __name__ == "__main__":
    run_data_engineering()
