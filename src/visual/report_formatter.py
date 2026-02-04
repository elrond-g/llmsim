"""
性能报告格式化输出 - 支持多种输出格式（Console, Excel等）
"""

from src.visual.console_report import ConsoleReportFormatter
from src.visual.excel_report import ExcelReportFormatter
from src.visual.report_base import ReportFormatter


def create_formatter(format_type: str = "console") -> ReportFormatter:
    """
    工厂函数 - 创建对应格式的报告格式化器

    Args:
        format_type: 输出格式类型 ('console' 或 'excel')

    Returns:
        对应的ReportFormatter实例

    Raises:
        ValueError: 如果格式类型不支持
    """
    if format_type.lower() == "console":
        return ConsoleReportFormatter()
    elif format_type.lower() == "excel":
        return ExcelReportFormatter()
    else:
        raise ValueError(f"不支持的输出格式: {format_type}。支持的格式: console, excel")
