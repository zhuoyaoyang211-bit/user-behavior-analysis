"""数据质量校验模块。

对清洗后的数据进行多维度质量检查，生成结构化的质量报告，
帮助快速发现数据问题并评估数据是否满足后续分析需求。

检查维度:
    1. 清洗摘要：原始/清洗后行数、各步骤删除明细、保留率
    2. 缺失值检查：各列的缺失数量与比例
    3. 取值范围校验：ID 无负数、行为类型合法
    4. 重复率统计：四元组重复检测，统计但不删除（经确认重复为合理现象）
    5. 分布统计：行为类型分布、时间范围、去重计数
    6. 重度买家标记：is_power_user 标记的分布与行为占比
    7. 时间完整性：检查数据时间范围内是否有日期断档
"""

from pathlib import Path

import pandas as pd

from common.constants import (
    BEHAVIOR_NAME_MAP,
    VALID_BEHAVIOR_TYPES,
)
from common.exceptions import DataQualityError
from common.logger import get_logger

logger = get_logger(__name__)


class DataQualityChecker:
    """数据质量校验器。

    对 DataFrame 执行多维度质量检查，生成质量报告。
    报告包含缺失值、范围校验、分布统计等维度的检查结果。

    Attributes:
        report_path: 质量报告输出路径
    """

    def __init__(self, report_path: Path | None = None) -> None:
        """初始化数据质量校验器。

        Args:
            report_path: 质量报告文本文件输出路径，
                为 None 时不写文件仅返回报告字符串
        """
        self.report_path: Path | None = report_path

    def run_all_checks(
        self, df: pd.DataFrame, cleaning_stats: dict | None = None
    ) -> str:
        """执行全部质量检查并生成报告。

        依次执行清洗摘要、缺失值检查、范围校验、重复率验证、
        分布统计、重度买家标记、时间完整性，汇总为一份完整报告。

        Args:
            df: 待检查的 DataFrame（建议为清洗后的数据）
            cleaning_stats: 清洗流程统计字典，由 main.py 从清洗过程
                收集。包含各步骤行数、爬虫/重度买家数量等。
                为 None 时跳过清洗摘要段。

        Returns:
            质量报告字符串

        Raises:
            DataQualityError: 数据为空或检查过程异常时抛出
        """
        if df.empty:
            raise DataQualityError("数据为空，无法执行质量检查")

        logger.info("开始数据质量校验, 行数: %d", len(df))

        report_lines: list[str] = []
        report_lines.append("=" * 60)
        report_lines.append("用户行为数据质量校验报告")
        report_lines.append("=" * 60)
        report_lines.append(f"总行数: {len(df):,}")
        report_lines.append(f"列数: {len(df.columns)}")
        report_lines.append("")

        # 检查 1: 清洗摘要（如有清洗统计）
        if cleaning_stats is not None:
            report_lines.extend(self._check_cleaning_summary(df, cleaning_stats))
            report_lines.append("")

        # 检查 2: 缺失值
        report_lines.extend(self._check_missing(df))
        report_lines.append("")

        # 检查 3: 取值范围
        report_lines.extend(self._check_ranges(df))
        report_lines.append("")

        # 检查 4: 重复率验证
        report_lines.extend(self._check_duplicates(df))
        report_lines.append("")

        # 检查 5: 分布统计
        report_lines.extend(self._check_distribution(df))
        report_lines.append("")

        # 检查 6: 重度买家标记
        if "is_power_user" in df.columns:
            report_lines.extend(self._check_power_users(df))
            report_lines.append("")

        # 检查 7: 时间完整性
        report_lines.extend(self._check_time_completeness(df))
        report_lines.append("")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)
        logger.info("质量校验完成")

        # 写入报告文件
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_path.write_text(report, encoding="utf-8")
            logger.info("质量报告已保存: %s", self.report_path)

        return report

    def _check_missing(self, df: pd.DataFrame) -> list[str]:
        """检查各列缺失值情况。

        Args:
            df: 待检查的 DataFrame

        Returns:
            缺失值检查结果的文本行列表
        """
        lines = ["[缺失值检查]"]
        missing = df.isnull().sum()
        for col, count in missing.items():
            ratio = count / len(df) * 100
            status = "OK" if count == 0 else "WARN"
            lines.append(f"  {col}: 缺失 {count:,} 条 ({ratio:.4f}%) [{status}]")
        return lines

    def _check_ranges(self, df: pd.DataFrame) -> list[str]:
        """检查数值列取值范围是否合法。

        Args:
            df: 待检查的 DataFrame

        Returns:
            范围校验结果的文本行列表
        """
        lines = ["[取值范围校验]"]

        # user_id 范围检查
        uid_min, uid_max = df["user_id"].min(), df["user_id"].max()
        uid_neg = (df["user_id"] < 0).sum()
        lines.append(f"  user_id: 范围 [{uid_min:,}, {uid_max:,}], 负值 {uid_neg} 条")

        # item_id 范围检查
        iid_min, iid_max = df["item_id"].min(), df["item_id"].max()
        iid_neg = (df["item_id"] < 0).sum()
        lines.append(f"  item_id: 范围 [{iid_min:,}, {iid_max:,}], 负值 {iid_neg} 条")

        # behavior_type 合法性检查
        invalid_behaviors = ~df["behavior_type"].isin(VALID_BEHAVIOR_TYPES)
        invalid_count = invalid_behaviors.sum()
        status = "OK" if invalid_count == 0 else "FAIL"
        lines.append(f"  behavior_type: 非法值 {invalid_count} 条 [{status}]")

        return lines

    def _check_distribution(self, df: pd.DataFrame) -> list[str]:
        """统计行为类型分布与时间范围。

        Args:
            df: 待检查的 DataFrame

        Returns:
            分布统计结果的文本行列表
        """
        lines = ["[分布统计]"]

        # 行为类型分布
        behavior_counts = df["behavior_type"].value_counts().sort_index()
        for btype, count in behavior_counts.items():
            name = BEHAVIOR_NAME_MAP.get(btype, "未知")
            ratio = count / len(df) * 100
            lines.append(f"  {btype}({name}): {count:,} 条 ({ratio:.2f}%)")

        # 时间范围
        time_min = df["time"].min()
        time_max = df["time"].max()
        lines.append(f"  时间范围: {time_min} ~ {time_max}")

        # 去重计数
        unique_users = df["user_id"].nunique()
        unique_items = df["item_id"].nunique()
        unique_cats = df["item_category"].nunique()
        lines.append(f"  独立用户数: {unique_users:,}")
        lines.append(f"  独立商品数: {unique_items:,}")
        lines.append(f"  独立类目数: {unique_cats:,}")

        return lines

    def _check_cleaning_summary(
        self, df: pd.DataFrame, stats: dict
    ) -> list[str]:
        """生成清洗流程摘要。

        汇总分块清洗与全量异常处理的统计信息，展示数据从原始到最终
        的变化过程，回答"删了什么、为什么删、留了多少"。

        Args:
            df: 清洗后的 DataFrame
            stats: 清洗统计字典，由 main.py 从清洗过程收集

        Returns:
            清洗摘要的文本行列表
        """
        lines = ["[清洗摘要]"]

        original = stats.get("original_rows", 0)
        after_dedup = stats.get("after_dedup", 0)
        after_behavior = stats.get("after_behavior_filter", 0)
        after_time = stats.get("after_time_clean", 0)
        cross_dup = stats.get("cross_dup_count", 0)
        crawler_removed = stats.get("crawler_removed_rows", 0)
        final_rows = len(df)

        behavior_removed = after_dedup - after_behavior
        time_removed = after_behavior - after_time
        total_removed = original - final_rows
        retention = final_rows / original * 100 if original > 0 else 0

        lines.append(f"  原始行数:       {original:,}")
        lines.append(f"  四元组重复检测: 检出 {cross_dup:,} 条 (保留不删除)")
        lines.append(f"  非法行为删除:   {behavior_removed:,} 条")
        lines.append(f"  时间无效删除:   {time_removed:,} 条")
        lines.append(f"  爬虫删除:       {crawler_removed:,} 条")
        lines.append(f"  最终行数:       {final_rows:,}")
        lines.append(
            f"  总删除:         {total_removed:,} 条 ({100 - retention:.1f}%)"
        )
        lines.append(f"  保留率:         {retention:.1f}%")

        outlier_count = stats.get("outlier_user_count", 0)
        crawler_count = stats.get("crawler_count", 0)
        power_user_count = stats.get("power_user_count", 0)
        lines.append(
            f"  异常用户处理:   检出 {outlier_count} 名 -> "
            f"爬虫删除 {crawler_count} 名, "
            f"重度买家标记 {power_user_count} 名保留"
        )

        return lines

    def _check_duplicates(self, df: pd.DataFrame) -> list[str]:
        """统计四元组重复情况，检测但不删除。

        对四元组（user_id, item_id, behavior_type, time）进行重复检测，
        统计重复行数、重复率、最大重复次数和各行为类型重复分布。

        设计原因:
            大纲要求"按四元组去重"，但经实际数据验证发现重复率达 49.3%，
            且最高重复22次。经指导老师确认，高重复次数属于合理现象
           （如用户在同一小时内多次浏览同一商品），不做删除处理。
            因此本段改为统计报告，不判定 FAIL。

        Args:
            df: 待检查的 DataFrame

        Returns:
            重复率统计结果的文本行列表
        """
        lines = ["[重复率统计]"]

        dup_count = int(
            df.duplicated(
                subset=["user_id", "item_id", "behavior_type", "time"]
            ).sum()
        )
        ratio = dup_count / len(df) * 100 if len(df) > 0 else 0

        # 统计最大重复次数
        group_sizes = df.groupby(
            ["user_id", "item_id", "behavior_type", "time"]
        ).size()
        max_repeat = int(group_sizes.max())

        lines.append(
            f"  四元组重复: {dup_count:,} 条 ({ratio:.2f}%)"
        )
        lines.append(f"  最大重复次数: {max_repeat} 次")
        lines.append(f"  处理策略: 检测并统计, 保留全部记录不删除")

        # 各行为类型重复分布
        lines.append("  各行为类型重复率:")
        names = {1: "浏览", 2: "收藏", 3: "加购", 4: "购买"}
        for bt in [1, 2, 3, 4]:
            sub = df[df["behavior_type"] == bt]
            if len(sub) == 0:
                continue
            sub_dup = sub.duplicated(
                subset=["user_id", "item_id", "behavior_type", "time"]
            ).sum()
            sub_ratio = sub_dup / len(sub) * 100
            lines.append(
                f"    {names.get(bt, '未知')}: {sub_dup:,} / {len(sub):,} ({sub_ratio:.1f}%)"
            )

        return lines

    def _check_power_users(self, df: pd.DataFrame) -> list[str]:
        """统计重度买家标记的分布情况。

        is_power_user 是爬虫检测的产出：IQR 检出的异常用户中，
        非爬虫的标记为重度买家保留。本方法统计标记分布与行为占比，
        帮助下游了解高价值用户的规模。

        Args:
            df: 待检查的 DataFrame（须包含 is_power_user 列）

        Returns:
            重度买家标记统计的文本行列表
        """
        lines = ["[重度买家标记]"]

        power_user_count = int(
            df.loc[df["is_power_user"], "user_id"].nunique()
        )
        normal_user_count = int(
            df.loc[~df["is_power_user"], "user_id"].nunique()
        )
        power_user_rows = int(df["is_power_user"].sum())
        power_user_ratio = (
            power_user_rows / len(df) * 100 if len(df) > 0 else 0
        )

        lines.append(f"  is_power_user=True:  {power_user_count:,} 人")
        lines.append(f"  is_power_user=False: {normal_user_count:,} 人")
        lines.append(
            f"  重度买家行为量: {power_user_rows:,} 条 "
            f"(占总行为 {power_user_ratio:.1f}%)"
        )

        return lines

    def _check_time_completeness(self, df: pd.DataFrame) -> list[str]:
        """检查数据时间范围内是否有日期断档。

        统计数据覆盖的起止日期，计算应覆盖天数与实际有数据天数，
        若存在缺失日期则列出，供后续业务解释。

        Args:
            df: 待检查的 DataFrame（须包含 time 列）

        Returns:
            时间完整性检查结果的文本行列表
        """
        lines = ["[时间完整性]"]

        dates = df["time"].dt.date.unique()
        date_min = df["time"].min().date()
        date_max = df["time"].max().date()

        expected_days = (date_max - date_min).days + 1
        actual_days = len(dates)
        missing_days = expected_days - actual_days
        status = "OK" if missing_days == 0 else "WARN"

        lines.append(f"  时间范围: {date_min} ~ {date_max}")
        lines.append(f"  应覆盖天数: {expected_days}")
        lines.append(f"  实际有数据天数: {actual_days} [{status}]")

        if missing_days > 0:
            all_dates = pd.date_range(date_min, date_max, freq="D").date
            missing_dates = sorted(set(all_dates) - set(dates))
            missing_str = ", ".join(str(d) for d in missing_dates[:10])
            if len(missing_dates) > 10:
                missing_str += f" ... (共 {len(missing_dates)} 天)"
            lines.append(f"  缺失日期: {missing_str}")
        else:
            lines.append("  缺失日期: 无")

        return lines
