from src.arch.perf_calculator import ModelPerformance
from src.visual.report_base import ReportFormatter


class ConsoleReportFormatter(ReportFormatter):
    """Console output formatter - Beautified table format"""

    @staticmethod
    def _display_width(text: str) -> int:
        """Calculate display width of text (considering Chinese characters take 2 widths)"""
        width = 0
        for char in text:
            if ord(char) >= 0x4E00 and ord(char) <= 0x9FFF:  # Chinese character range
                width += 2
            else:
                width += 1
        return width

    @staticmethod
    def _pad_string(text: str, width: int, align: str = "left") -> str:
        """Pad string based on display width"""
        display_len = ConsoleReportFormatter._display_width(text)
        padding = width - display_len
        if padding <= 0:
            return text
        if align == "left":
            return text + " " * padding
        else:
            return " " * padding + text

    def format(self, model_perf: ModelPerformance) -> None:
        """
        Format and print performance report to console

        Args:
            model_perf: Model performance metrics
        """
        all_rows = self._collect_data(model_perf)

        # Calculate maximum width for each column (considering Chinese characters)
        if all_rows:
            col_widths = {}
            # Text columns
            col_widths["name"] = (
                max(
                    max(self._display_width(row["name"]) for row in all_rows),
                    self._display_width("Operator Name"),
                )
                + 2
            )
            col_widths["type"] = (
                max(
                    max(self._display_width(row["type"]) for row in all_rows),
                    self._display_width("Type"),
                )
                + 1
            )
            # Numeric columns
            col_widths["m"] = (
                max(max(len(str(row["m"])) for row in all_rows), len("m")) + 1
            )
            col_widths["n"] = (
                max(max(len(str(row["n"])) for row in all_rows), len("n")) + 1
            )
            col_widths["k"] = (
                max(max(len(str(row["k"])) for row in all_rows), len("k")) + 1
            )
            col_widths["batch"] = (
                max(max(len(str(row["batch"])) for row in all_rows), len("batch")) + 1
            )
            col_widths["layers"] = (
                max(max(len(str(row["layers"])) for row in all_rows), len("layers")) + 1
            )
            col_widths["in_dtype"] = (
                max(
                    max(self._display_width(row["in_dtype"]) for row in all_rows),
                    self._display_width("Input"),
                )
                + 1
            )
            col_widths["out_dtype"] = (
                max(
                    max(self._display_width(row["out_dtype"]) for row in all_rows),
                    self._display_width("Output"),
                )
                + 1
            )
            col_widths["weight_dtype"] = (
                max(
                    max(self._display_width(row["weight_dtype"]) for row in all_rows),
                    self._display_width("Weight"),
                )
                + 1
            )
            # Float columns (width sufficient to display formatted values)
            col_widths["compute"] = max(len("Compute(us)"), 10) + 1
            col_widths["memory"] = max(len("Memory(us)"), 10) + 1
            col_widths["transfer"] = max(len("Transfer(us)"), 10) + 1
            col_widths["op_time_single_layer"] = (
                max(len("Single Layer Theory Latency(us)"), 15) + 1
            )
            col_widths["total"] = max(len("Total Time(ms)"), 10) + 1
            col_widths["percent"] = max(len("Percent(%)"), 8) + 1
            col_widths["op_weight_mem"] = max(len("Weight Memory/Single GPU"), 12) + 1
        else:
            col_widths = {
                "name": 20,
                "type": 10,
                "m": 6,
                "n": 6,
                "k": 6,
                "batch": 8,
                "layers": 8,
                "in_dtype": 8,
                "out_dtype": 8,
                "weight_dtype": 8,
                "compute": 12,
                "memory": 12,
                "transfer": 12,
                "op_time_single_layer": 12,
                "total": 12,
                "percent": 10,
                "op_weight_mem": 12,
            }

        # Calculate total width (considering separators)
        total_width = sum(col_widths.values()) + 16  # 16 = 分隔符号的宽度

        # Print header
        print()
        print("┌" + "─" * (total_width - 2) + "┐")
        header_text = f"Performance Analysis Report: {model_perf.model_name} ({model_perf.forward_mode})"
        padding = total_width - self._display_width(header_text) - 4
        print(f"│ {header_text}" + " " * max(padding, 1) + " │")
        print("├" + "─" * (total_width - 2) + "┤")

        # Print column headers
        headers = [
            ("name", "Operator Name"),
            ("type", "Type"),
            ("m", "m"),
            ("n", "n"),
            ("k", "k"),
            ("batch", "batch"),
            ("layers", "layers"),
            ("in_dtype", "Input"),
            ("out_dtype", "Output"),
            ("weight_dtype", "Weight"),
            ("compute", "Compute(us)"),
            ("memory", "Memory(us)"),
            ("transfer", "Transfer(us)"),
            ("op_time_single_layer", "Single Layer Theory Latency(us)"),
            ("total", "Total Time(ms)"),
            ("percent", "Percent(%)"),
            ("op_weight_mem", "Weight/Single GPU All Layers"),
        ]

        header_line = "│ "
        for key, label in headers:
            width = col_widths[key]
            if key in ["compute", "memory", "transfer", "total", "percent"]:
                # Numeric columns right-aligned
                padded = self._pad_string(label, width, "right")
            else:
                # Text columns left-aligned
                padded = self._pad_string(label, width, "left")
            header_line += f"{padded} │ "
        print(header_line)
        print("├" + "─" * (total_width - 2) + "┤")

        # Print data rows
        for row in all_rows:
            line = "│ "
            for key, _ in headers:
                width = col_widths[key]
                value = row[key]

                if key in ["compute", "memory", "transfer", "op_time_single_layer"]:
                    # Float, 3 decimal places
                    formatted = f"{value:.3f}"
                    padded = self._pad_string(formatted, width, "right")
                    line += f"{padded} │ "
                elif key == "total":
                    # Total time, 3 decimal places
                    formatted = f"{value:.3f}"
                    padded = self._pad_string(formatted, width, "right")
                    line += f"{padded} │ "
                elif key == "percent":
                    # Percentage, 2 decimal places
                    formatted = f"{value:.2f}"
                    padded = self._pad_string(formatted, width, "right")
                    line += f"{padded} │ "
                elif key in ["m", "n", "k", "batch", "layers", "op_weight_mem"]:
                    # Integer columns
                    formatted = str(value)
                    padded = self._pad_string(formatted, width, "right")
                    line += f"{padded} │ "
                else:
                    # Text columns
                    padded = self._pad_string(str(value), width, "left")
                    line += f"{padded} │ "
            print(line)

        print("├" + "─" * (total_width - 2) + "┤")

        # Summary statistics
        total_compute_ms = model_perf.total_compute_time / 1000.0
        total_memory_ms = model_perf.total_memory_time / 1000.0
        total_transfer_ms = model_perf.total_transfer_time / 1000.0
        total_ms = model_perf.total_time / 1000.0

        summary_lines = [
            "Summary Statistics / Single Layer",
            f"  Compute Time: {total_compute_ms:>10.3f} ms",
            f"  Memory Time: {total_memory_ms:>10.3f} ms",
            f"  Transfer Time: {total_transfer_ms:>10.3f} ms",
            f"  Total Time:   {total_ms:>10.3f} ms",
        ]

        for summary in summary_lines:
            padding = total_width - self._display_width(summary) - 4
            print(f"│ {summary}" + " " * max(padding, 1) + " │")

        # Performance bottleneck
        bottleneck = model_perf.get_bottleneck_op()
        if bottleneck:
            layer_name, op_name, op_perf = bottleneck
            print("├" + "─" * (total_width - 2) + "┤")
            bottleneck_text = f"Performance Bottleneck: {op_name} (Total Time: {op_perf.total_time:.3f} ms)"
            padding = total_width - self._display_width(bottleneck_text) - 4
            print(f"│ {bottleneck_text}" + " " * max(padding, 1) + " │")

        # TTFT and Throughput
        ttft = model_perf.get_ttft_or_tpot()
        print("├" + "─" * (total_width - 2) + "┤")
        ttft_text = (
            f"TTFT: (Time: {ttft:.3f} ms)"
            if model_perf.forward_mode == "EXTEND"
            else f"TPOT: (Time: {ttft:.3f} ms)"
        )
        padding = total_width - self._display_width(ttft_text) - 4
        print(f"│ {ttft_text}" + " " * max(padding, 1) + " │")

        throughput = model_perf.get_throughput()
        print("├" + "─" * (total_width - 2) + "┤")
        throughput_txt = f"TPS: (throughput: {throughput:.3f})"
        padding = total_width - self._display_width(throughput_txt) - 4
        print(f"│ {throughput_txt}" + " " * max(padding, 1) + " │")

        total_weight_mem_single_gpu = model_perf.model_total_mem_occupy / (1024**3)
        print("├" + "─" * (total_width - 2) + "┤")
        weight_mem_text = (
            f"Weight Memory/Single GPU: {total_weight_mem_single_gpu:.3f} GB"
        )
        padding = total_width - self._display_width(weight_mem_text) - 4
        print(f"│ {weight_mem_text}" + " " * max(padding, 1) + " │")

        print("└" + "─" * (total_width - 2) + "┘")

    def save(self, model_perf: ModelPerformance, output_path: str = None) -> None:
        """
        Save performance report to file

        Args:
            model_perf: Model performance metrics
            output_path: Output file path (optional)
        """
        if output_path:
            # Redirect stdout to file
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            self.format(model_perf)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Report saved to: {output_path}")
        else:
            # Print directly to console
            self.format(model_perf)
