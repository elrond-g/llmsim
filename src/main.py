import argparse
import sys
import traceback

from src.arch.config import ForwardMode, ModelConfig, ScheduleConfig
from src.arch.models_arch.model_arch import create_model_arch
from src.arch.perf_calculator import PerformanceCalculator
from src.hardware.hardware_config import (
    DEFAULT_HARDWARE,
    HardwareConfig,
    get_hardware_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Inference Performance Analysis Tool"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model configuration file path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size",
    )
    parser.add_argument(
        "--max_seqlen",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="extend",
        choices=["extend", "decode"],
        help="Forward pass mode [extend, decode]",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=4,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=4,
        help="Data parallelism size",
    )
    parser.add_argument(
        "--ep_size",
        type=int,
        default=16,
        help="Expert parallelism size",
    )
    parser.add_argument(
        "--enable_mtp",
        action="store_true",
        help="Enable multi-token prediction",
    )
    parser.add_argument(
        "--enable_deepep",
        action="store_true",
        help="Enable deep expert parallelism",
    )
    parser.add_argument(
        "--enable_moe_dense_fully_dp",
        action="store_true",
        help="Enable MoE dense layer fully data parallelism",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=["default", "h20", "h800", "gb200", "klx_p800", "custom"],
        default="default",
        help="Hardware configuration preset (default, h20, h800, gb200, klx_p800, custom)",
    )
    parser.add_argument(
        "--hardware_config",
        type=str,
        default=None,
        help="Custom hardware configuration file path (used when --hardware=custom)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["console", "excel"],
        default="console",
        help="Output format (console, excel)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (recommended when --output_format=excel)",
    )

    args = parser.parse_args()
    return args


def validate_args(args) -> None:
    """Validate command line arguments"""
    if args.max_seqlen % args.tp_size != 0:
        raise ValueError(
            f"max_seqlen ({args.max_seqlen}) must be divisible by tp_size ({args.tp_size})"
        )

    if args.batch_size > args.tp_size:
        if args.batch_size % args.tp_size != 0:
            raise ValueError(
                f"batch_size ({args.batch_size}) must be divisible by tp_size ({args.tp_size})"
            )


def main() -> None:
    """Main function"""
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        print(f"Argument validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Load model configuration
    print(f"Loading model configuration: {args.model_path}")
    try:
        model_config = ModelConfig.from_json(args.model_path)
    except Exception as e:
        print(f"Failed to load model configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Create schedule configuration
    schedule_config = ScheduleConfig(
        batch_size=args.batch_size,
        max_seqlen=args.max_seqlen,
        mode=ForwardMode.EXTEND if args.mode == "extend" else ForwardMode.DECODE,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        ep_size=args.ep_size,
        is_mtp=args.enable_mtp,
        deepep=args.enable_deepep,
        enable_moe_dense_fully_dp=args.enable_moe_dense_fully_dp,
    )

    # Select hardware configuration
    if args.hardware == "custom":
        # Custom configuration file
        if args.hardware_config is None:
            print(
                "Error: --hardware_config must be specified when using --hardware=custom",
                file=sys.stderr,
            )
            sys.exit(1)
        hardware_config = HardwareConfig.from_json(args.hardware_config)
    elif args.hardware in ["h20", "h800", "gb200", "klx_p800"]:
        # Load from predefined configuration
        hardware_config = get_hardware_config(args.hardware)
    else:
        # Default configuration
        hardware_config = DEFAULT_HARDWARE

    # Create model architecture
    print(f"Model type: {model_config.model_type}")
    print(f"Forward mode: {schedule_config.mode.name}")
    print(f"Hardware configuration: {hardware_config.name}")
    print()

    try:
        model_arch = create_model_arch(model_config, schedule_config)
    except Exception as e:
        print(f"Failed to create model architecture: {e}", file=sys.stderr)
        sys.exit(1)

    # Calculate performance
    print("Calculating performance metrics...")
    calculator = PerformanceCalculator(hardware_config)

    try:
        model_perf = calculator.calculate_model_performance(model_arch)
    except Exception as e:
        print(f"Performance calculation failed: {e}", file=sys.stderr)
        print("\nDetailed error stack:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Print performance report
    calculator.print_performance_report(
        model_perf, output_format=args.output_format, output_path=args.output_file
    )


if __name__ == "__main__":
    main()
