from dataclasses import dataclass
from pathlib import Path
from typing import Any
from io import StringIO

import pandas as pd
from jinja2 import Environment, FileSystemLoader


@dataclass(frozen=True)
class BenchmarkTableConfig:
    csv_path: Path = Path("results/data/reduction.csv")
    out_path: Path = Path("results/markdown_tables/reduction_bench_tables.md")
    kernel_name: str = "Reduction"
    large_n: int = 1 << 24
    main_input_pattern: str = "ones"


@dataclass(frozen=True)
class BenchmarkReportConfig(BenchmarkTableConfig):
    template_dir: Path = Path("results/templates")
    template_name: str = "reduction_report.md.j2"
    report_out_path: Path = Path("results/reduction.md")

    @property
    def md_table_rel_path(self) -> Path:
        return self.out_path.relative_to("results")

    @property
    def csv_rel_path(self) -> Path:
        return self.csv_path.relative_to("results")


FLOAT_COLUMNS = [
    "avg_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "result",
    "ref",
    "abs_error",
    "rel_error",
    "bw_GB_s",
    "gflops",
]

TABLE_COLUMNS = {
    "main_table": [
        "Version",
        "Block Size",
        "Grid Size",
        "H2D Time",
        "Kernel Time [ms]",
        "D2H Time [ms]",
        "CPU finalize",
        "Post-H2D Time",
        "E2E Time [ms]",
        "Post-H2D BW [GB/s]",
        "E2E BW [GB/s]",
        "Post-H2D GFLOP/s",
        "E2E GFLOP/s",
        "Correct",
        "Post-H2D Speedup",
        "E2E Speedup",
    ],
    "block_size_sweep": [
        "Block Size",
        "Grid Size",
        "Avg Time [ms]",
        "Min Time [ms]",
        "Max Time [ms]",
        "Std Dev",
        "Effective BW [GB/s]",
        "GFLOP/s",
        "Result",
        "Ref",
        "Abs Error",
        "Rel Error",
        "Correct",
        "Kernel-stage Speedup",
    ],
    "problem_size_sweep": [
        "N",
        "Single Vector Size [MiB]",
        "CPU Time [ms]",
        "CPU BW [GB/s]",
        "Best GPU Block Size",
        "GPU Grid Size",
        "GPU Post-H2D Time [ms]",
        "GPU Post-H2D BW [GB/s]",
        "GPU Post-H2D GFLOP/s",
        "GPU E2E Time [ms]",
        "GPU E2E BW [GB/s]",
        "GPU E2E GFLOP/s",
        "Result",
        "Ref",
        "Correct",
        "GPU Post-H2D Speedup",
        "GPU E2E Speedup",
    ],
    "problem_size_sweep_cpu_optimized": [
        "N",
        "Single Vector Size [MiB]",
        "CPU serial Time [ms]",
        "CPU serial BW [GB/s]",
        "CPU optimized/parallel Time [ms]",
        "CPU optimized/parallel BW [GB/s]",
        "Result",
        "Ref",
        "Correct",
        "Speedup",
    ],
}


def to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def bandwidth_GB_s(num_bytes: float, time_ms: float) -> float:
    return float(num_bytes) / (time_ms / 1000.0 * 1e9)


def calculate_gflops(n_flop: float, time_ms: float) -> float:
    return float(n_flop) / (time_ms / 1000.0 * 1e9)


def num_flop_reduction(n: int) -> int:
    return max(int(n) - 1, 0)


def format_ms(value: Any) -> str:
    return "N/A" if value == "N/A" else f"{float(value):.3f}"


def format_float(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def format_speedup(value: Any) -> str:
    return "" if value == "" else f"{float(value):.2f}x"


def make_markdown_table(rows: list[dict[str, Any]], table_name: str) -> str:
    columns = TABLE_COLUMNS[table_name]
    table = pd.DataFrame(rows, columns=columns)
    return table.to_markdown(index=False)


def read_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "input_pattern" not in df.columns:
        df["input_pattern"] = "ones"

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    if "correct" in df.columns:
        df["correct"] = df["correct"].map(to_bool)

    return df


def filter_main_input_pattern(
    df: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> pd.DataFrame:
    return df.loc[df["input_pattern"] == config.main_input_pattern].copy()


def get_cpu_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["mode"] == "cpu"].copy()


def get_cpu_optimized_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df["mode"] == "cpu") & (df["version"] != "cpu_serial")].copy()


def get_cpu_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[(df["n"] == int(n)) & (df["mode"] == "cpu")].copy()


def get_cpu_serial_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[
        (df["n"] == int(n)) & (df["mode"] == "cpu") & (df["version"] == "cpu_serial")
    ].copy()


def get_gpu_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["mode"] != "cpu"].copy()


def get_kernel_versions(df_gpu: pd.DataFrame) -> list[str]:
    if df_gpu.empty:
        return []

    versions = sorted(df_gpu.loc[df_gpu["mode"] == "cuda_kernel", "version"].unique())

    return list(versions)


def get_cpu_versions(df_cpu: pd.DataFrame) -> list[str]:
    return list(df_cpu["version"].unique())


def get_cpu_optimized_versions(df_cpu_optimized: pd.DataFrame) -> list[str]:
    return [v for v in df_cpu_optimized["version"].unique() if v != "cpu_serial"]


def get_h2d_time_at_n(df_gpu: pd.DataFrame, n: int) -> float:
    rows = df_gpu.loc[(df_gpu["n"] == int(n)) & (df_gpu["mode"] == "h2d")]

    if rows.empty:
        return 0.0

    return float(rows.iloc[0]["avg_ms"])


def get_best_kernel_row(rows: pd.DataFrame) -> pd.Series:
    kernel_rows = rows.loc[rows["mode"] == "cuda_kernel"]

    if kernel_rows.empty:
        raise RuntimeError("No cuda_kernel row found.")

    return kernel_rows.sort_values("avg_ms").iloc[0]


def find_matching_stage_rows(
    gpu_rows_for_version_at_n: pd.DataFrame,
    mode: str,
    best_kernel_row: pd.Series,
) -> pd.DataFrame:
    rows = gpu_rows_for_version_at_n.loc[gpu_rows_for_version_at_n["mode"] == mode]

    if rows.empty:
        return rows

    best_block_size = best_kernel_row["block_size"]
    best_grid_size = best_kernel_row["grid_size"]

    block_matched = rows.loc[rows["block_size"] == best_block_size]

    if not block_matched.empty:
        grid_matched = block_matched.loc[block_matched["grid_size"] == best_grid_size]

        if not grid_matched.empty:
            return grid_matched

        return block_matched

    return rows


def get_matching_stage_time(
    gpu_rows_for_version_at_n: pd.DataFrame,
    mode: str,
    best_kernel_row: pd.Series,
) -> tuple[float, pd.Series | None]:
    rows = find_matching_stage_rows(
        gpu_rows_for_version_at_n=gpu_rows_for_version_at_n,
        mode=mode,
        best_kernel_row=best_kernel_row,
    )

    if rows.empty:
        return 0.0, None

    row = rows.iloc[0]
    return float(row["avg_ms"]), row


def get_cpu_finalize_info(
    gpu_rows_for_version_at_n: pd.DataFrame,
    best_kernel_row: pd.Series,
) -> tuple[Any, bool, Any]:
    time_cpu_finalize, row_cpu_finalize = get_matching_stage_time(
        gpu_rows_for_version_at_n=gpu_rows_for_version_at_n,
        mode="cpu_finalize",
        best_kernel_row=best_kernel_row,
    )

    if row_cpu_finalize is None:
        return "N/A", to_bool(best_kernel_row["correct"]), best_kernel_row["result"]

    return (
        time_cpu_finalize,
        to_bool(row_cpu_finalize["correct"]),
        row_cpu_finalize["result"],
    )


def build_cpu_main_row(
    cpu_rows: pd.DataFrame,
    version: str,
) -> tuple[dict[str, Any] | None, float | None]:
    if cpu_rows.empty:
        return None, None

    version_rows = cpu_rows.loc[cpu_rows["version"] == version]
    if version_rows.empty:
        return None, None

    cpu_serial_rows = cpu_rows.loc[cpu_rows["version"] == "cpu_serial"]
    if cpu_serial_rows.empty:
        raise RuntimeError("cpu_serial row missing.")

    row = version_rows.iloc[0]
    cpu_ref_time = float(cpu_serial_rows.iloc[0]["avg_ms"])

    time_post_h2d = float(row["avg_ms"])
    time_e2e = float(row["avg_ms"])

    speedup_post_h2d = cpu_ref_time / time_post_h2d
    speedup_e2e = cpu_ref_time / time_e2e

    main_row = {
        "Version": row["version"],
        "Block Size": "N/A",
        "Grid Size": "N/A",
        "H2D Time": "N/A",
        "Kernel Time [ms]": "N/A",
        "D2H Time [ms]": "N/A",
        "CPU finalize": "N/A",
        "Post-H2D Time": format_ms(row["avg_ms"]),
        "E2E Time [ms]": format_ms(row["avg_ms"]),
        "Post-H2D BW [GB/s]": format_float(row["bw_GB_s"], 2),
        "E2E BW [GB/s]": format_float(row["bw_GB_s"], 2),
        "Post-H2D GFLOP/s": format_float(row["gflops"], 2),
        "E2E GFLOP/s": format_float(row["gflops"], 2),
        "Correct": to_bool(row["correct"]),
        "Post-H2D Speedup": format_speedup(speedup_post_h2d),
        "E2E Speedup": format_speedup(speedup_e2e),
    }

    return main_row, cpu_ref_time


def build_main_gpu_row(
    cpu_rows: pd.DataFrame,
    cpu_ref_time: float | None,
    df_gpu: pd.DataFrame,
    gpu_rows_for_version_at_large_n: pd.DataFrame,
    version: str,
    large_n: int,
) -> dict[str, Any]:
    cpu_serial_row = cpu_rows.loc[cpu_rows["version"] == "cpu_serial"].iloc[0]

    best_gpu = get_best_kernel_row(gpu_rows_for_version_at_large_n)

    time_h2d = get_h2d_time_at_n(df_gpu, large_n)
    time_kernel = float(best_gpu["avg_ms"])

    time_d2h, _ = get_matching_stage_time(
        gpu_rows_for_version_at_n=gpu_rows_for_version_at_large_n,
        mode="d2h",
        best_kernel_row=best_gpu,
    )

    time_cpu_finalize, correct, _ = get_cpu_finalize_info(
        gpu_rows_for_version_at_n=gpu_rows_for_version_at_large_n,
        best_kernel_row=best_gpu,
    )

    time_cpu_finalize_for_sum = (
        0.0 if time_cpu_finalize == "N/A" else float(time_cpu_finalize)
    )

    time_post_h2d = time_kernel + time_d2h + time_cpu_finalize_for_sum
    time_e2e = time_h2d + time_post_h2d

    if cpu_ref_time is not None:
        speedup_post_h2d = cpu_ref_time / time_post_h2d
        speedup_e2e = cpu_ref_time / time_e2e
    else:
        speedup_post_h2d = ""
        speedup_e2e = ""

    useful_bytes = float(cpu_serial_row["bytes"])
    bw_post_h2d = bandwidth_GB_s(useful_bytes, time_post_h2d)
    bw_e2e = bandwidth_GB_s(useful_bytes, time_e2e)

    useful_n_flop = num_flop_reduction(large_n)
    gflops_post_h2d = calculate_gflops(useful_n_flop, time_post_h2d)
    gflops_e2e = calculate_gflops(useful_n_flop, time_e2e)

    return {
        "Version": version,
        "Block Size": int(best_gpu["block_size"]),
        "Grid Size": int(best_gpu["grid_size"]),
        "H2D Time": format_ms(time_h2d),
        "Kernel Time [ms]": format_ms(time_kernel),
        "D2H Time [ms]": format_ms(time_d2h),
        "CPU finalize": format_ms(time_cpu_finalize),
        "Post-H2D Time": format_ms(time_post_h2d),
        "E2E Time [ms]": format_ms(time_e2e),
        "Post-H2D BW [GB/s]": format_float(bw_post_h2d, 2),
        "E2E BW [GB/s]": format_float(bw_e2e, 2),
        "Post-H2D GFLOP/s": format_float(gflops_post_h2d, 2),
        "E2E GFLOP/s": format_float(gflops_e2e, 2),
        "Correct": correct,
        "Post-H2D Speedup": format_speedup(speedup_post_h2d),
        "E2E Speedup": format_speedup(speedup_e2e),
    }


def build_main_table_rows(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    large_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    cpu_rows = get_cpu_rows_at_n(df, large_n)

    _, cpu_ref_time = build_cpu_main_row(cpu_rows, "cpu_serial")

    for version in get_cpu_versions(cpu_rows):
        cpu_row, _ = build_cpu_main_row(cpu_rows, version)
        if cpu_row is not None:
            rows.append(cpu_row)

    for version in get_kernel_versions(df_gpu):
        gpu_rows_for_version_at_large_n = df_gpu.loc[
            (df_gpu["n"] == large_n) & (df_gpu["version"] == version)
        ].copy()

        if gpu_rows_for_version_at_large_n.empty:
            continue

        rows.append(
            build_main_gpu_row(
                cpu_rows=cpu_rows,
                cpu_ref_time=cpu_ref_time,
                df_gpu=df_gpu,
                gpu_rows_for_version_at_large_n=gpu_rows_for_version_at_large_n,
                version=version,
                large_n=large_n,
            )
        )

    return rows


def get_block_sweep_rows(df: pd.DataFrame, large_n: int) -> pd.DataFrame:
    return df.loc[(df["n"] == large_n) & (df["mode"] == "cuda_kernel")].copy()


def build_block_sweep_table_rows(
    each_version_sweep: pd.DataFrame,
    cpu_ref_time: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for _, row in each_version_sweep.sort_values("block_size").iterrows():
        speedup = cpu_ref_time / float(row["avg_ms"])

        rows.append(
            {
                "Block Size": int(row["block_size"]),
                "Grid Size": int(row["grid_size"]),
                "Avg Time [ms]": format_ms(row["avg_ms"]),
                "Min Time [ms]": format_ms(row["min_ms"]),
                "Max Time [ms]": format_ms(row["max_ms"]),
                "Std Dev": format_ms(row["std_ms"]),
                "Effective BW [GB/s]": format_float(row["bw_GB_s"], 2),
                "GFLOP/s": format_float(row["gflops"], 2),
                "Result": format_float(row["result"], 3),
                "Ref": format_float(row["ref"], 3),
                "Abs Error": format_float(row["abs_error"], 3),
                "Rel Error": format_float(row["rel_error"], 3),
                "Correct": to_bool(row["correct"]),
                "Kernel-stage Speedup": format_speedup(speedup),
            }
        )

    return rows


def build_problem_size_row_cpu_optimized(
    df: pd.DataFrame,
    cpu_version_rows: pd.DataFrame,
    n: int,
) -> dict[str, Any] | None:
    cpu_serial_n = get_cpu_serial_rows_at_n(df, n)
    cpu_optimized_n = cpu_version_rows.loc[cpu_version_rows["n"] == n]

    if cpu_serial_n.empty or cpu_optimized_n.empty:
        return None

    cpu_serial_row = cpu_serial_n.iloc[0]
    cpu_optimized_row = cpu_optimized_n.iloc[0]

    single_vector_size_MiB = int(cpu_serial_row["size_of_dtype"]) * n / (1 << 20)

    cpu_serial_time_n = float(cpu_serial_row["avg_ms"])
    bw_cpu_serial_n = float(cpu_serial_row["bw_GB_s"])

    ref = (
        cpu_serial_row["ref"]
        if "ref" in cpu_serial_row and pd.notna(cpu_serial_row["ref"])
        else cpu_serial_row["result"]
    )

    cpu_optimized_time_n = float(cpu_optimized_row["avg_ms"])
    bw_cpu_optimized_n = float(cpu_optimized_row["bw_GB_s"])

    result = float(cpu_optimized_row["result"])
    correct = to_bool(cpu_optimized_row["correct"])

    speedup_n = cpu_serial_time_n / cpu_optimized_time_n

    return {
        "N": f"{n:,}",
        "Single Vector Size [MiB]": format_float(single_vector_size_MiB, 2),
        "CPU serial Time [ms]": format_ms(cpu_serial_time_n),
        "CPU serial BW [GB/s]": format_float(bw_cpu_serial_n, 2),
        "CPU optimized/parallel Time [ms]": format_ms(cpu_optimized_time_n),
        "CPU optimized/parallel BW [GB/s]": format_float(bw_cpu_optimized_n, 2),
        "Result": format_float(result, 3),
        "Ref": format_float(ref, 3),
        "Correct": correct,
        "Speedup": format_speedup(speedup_n),
    }


def build_problem_size_table_rows_for_cpu_version(
    df: pd.DataFrame,
    df_cpu_optimized: pd.DataFrame,
    version: str,
) -> list[dict[str, Any]]:
    cpu_version_rows = df_cpu_optimized.loc[
        df_cpu_optimized["version"] == version
    ].copy()

    rows: list[dict[str, Any]] = []

    for n in sorted(cpu_version_rows["n"].unique()):
        row = build_problem_size_row_cpu_optimized(
            df=df,
            cpu_version_rows=cpu_version_rows,
            n=int(n),
        )

        if row is not None:
            rows.append(row)

    return rows


def build_problem_size_row_gpu(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    gpu_version_rows: pd.DataFrame,
    n: int,
) -> dict[str, Any] | None:
    cpu_n = get_cpu_serial_rows_at_n(df, n)
    gpu_n = gpu_version_rows.loc[gpu_version_rows["n"] == n]

    if cpu_n.empty or gpu_n.empty:
        return None

    cpu_row = cpu_n.iloc[0]

    best_gpu_n = get_best_kernel_row(gpu_n)
    best_block_size = int(best_gpu_n["block_size"])
    best_grid_size = int(best_gpu_n["grid_size"])

    single_vector_size_MiB = int(cpu_row["size_of_dtype"]) * n / (1 << 20)

    cpu_time_n = float(cpu_row["avg_ms"])
    bw_cpu_n = float(cpu_row["bw_GB_s"])
    ref = (
        cpu_row["ref"]
        if "ref" in cpu_row and pd.notna(cpu_row["ref"])
        else cpu_row["result"]
    )

    time_h2d_n = get_h2d_time_at_n(df_gpu, n)
    time_kernel = float(best_gpu_n["avg_ms"])

    time_d2h, _ = get_matching_stage_time(
        gpu_rows_for_version_at_n=gpu_n,
        mode="d2h",
        best_kernel_row=best_gpu_n,
    )

    time_cpu_finalize, correct, result = get_cpu_finalize_info(
        gpu_rows_for_version_at_n=gpu_n,
        best_kernel_row=best_gpu_n,
    )

    time_cpu_finalize_for_sum = (
        0.0 if time_cpu_finalize == "N/A" else float(time_cpu_finalize)
    )

    time_post_h2d = time_kernel + time_d2h + time_cpu_finalize_for_sum
    time_e2e = time_h2d_n + time_post_h2d

    speedup_post_h2d = cpu_time_n / time_post_h2d
    speedup_e2e = cpu_time_n / time_e2e

    useful_bytes = float(cpu_row["bytes"])
    bw_post_h2d = bandwidth_GB_s(useful_bytes, time_post_h2d)
    bw_e2e = bandwidth_GB_s(useful_bytes, time_e2e)

    useful_n_flop = num_flop_reduction(n)
    gflops_post_h2d = calculate_gflops(useful_n_flop, time_post_h2d)
    gflops_e2e = calculate_gflops(useful_n_flop, time_e2e)

    return {
        "N": f"{n:,}",
        "Single Vector Size [MiB]": format_float(single_vector_size_MiB, 2),
        "CPU Time [ms]": format_ms(cpu_time_n),
        "CPU BW [GB/s]": format_float(bw_cpu_n, 2),
        "Best GPU Block Size": best_block_size,
        "GPU Grid Size": best_grid_size,
        "GPU Post-H2D Time [ms]": format_ms(time_post_h2d),
        "GPU Post-H2D BW [GB/s]": format_float(bw_post_h2d, 2),
        "GPU Post-H2D GFLOP/s": format_float(gflops_post_h2d, 2),
        "GPU E2E Time [ms]": format_ms(time_e2e),
        "GPU E2E BW [GB/s]": format_float(bw_e2e, 2),
        "GPU E2E GFLOP/s": format_float(gflops_e2e, 2),
        "Result": format_float(result, 3),
        "Ref": format_float(ref, 3),
        "Correct": correct,
        "GPU Post-H2D Speedup": format_speedup(speedup_post_h2d),
        "GPU E2E Speedup": format_speedup(speedup_e2e),
    }


def build_problem_size_table_rows_for_version(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    version: str,
) -> list[dict[str, Any]]:
    gpu_version_rows = df_gpu.loc[df_gpu["version"] == version].copy()

    rows: list[dict[str, Any]] = []

    for n in sorted(gpu_version_rows["n"].unique()):
        row = build_problem_size_row_gpu(
            df=df,
            df_gpu=df_gpu,
            gpu_version_rows=gpu_version_rows,
            n=int(n),
        )

        if row is not None:
            rows.append(row)

    return rows


def write_file_header(
    f,
    config: BenchmarkTableConfig,
) -> None:
    f.write(f"# Auto-Generated {config.kernel_name} Benchmark Tables\n\n")
    f.write(f"This file is automatically generated from `{config.csv_path}`.\n\n")


def write_main_benchmark_section(
    f,
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> None:
    large_n = config.large_n

    f.write("## Main Benchmark Results at Large N\n\n")
    f.write(f"N = {large_n:,}\n\n")
    f.write(f"DType = {df.iloc[0]['dtype']}\n\n")
    f.write(f"Size of dtype = {df.iloc[0]['size_of_dtype']} bytes\n\n")
    f.write(f"Input pattern: {df['input_pattern'].iloc[0]}\n\n")

    single_vector_size_MiB = large_n * int(df.iloc[0]["size_of_dtype"]) / (1 << 20)
    f.write(f"Single vector size = {format_float(single_vector_size_MiB, 2)} MiB\n\n")

    rows = build_main_table_rows(
        df=df,
        df_gpu=df_gpu,
        large_n=large_n,
    )

    f.write(make_markdown_table(rows, "main_table"))
    f.write("\n\n")


def write_block_size_sweep_section(
    f,
    df: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> None:
    large_n = config.large_n

    f.write("## CUDA Block Size Sweep\n\n")
    f.write(f"N = {large_n:,}\n\n")

    block_sweep = get_block_sweep_rows(df, large_n)

    if block_sweep.empty:
        f.write("No CUDA block-size sweep data available.\n\n")
        return

    f.write(f"DType = {block_sweep.iloc[0]['dtype']}\n\n")
    f.write(f"Size of dtype = {df.iloc[0]['size_of_dtype']} bytes\n\n")
    f.write(f"Input pattern: {df['input_pattern'].iloc[0]}\n\n")

    cpu_rows = get_cpu_rows_at_n(df, large_n)
    cpu_ref_time = float(
        cpu_rows.loc[cpu_rows["version"] == "cpu_serial", "avg_ms"].iloc[0]
    )

    for i_th_version, version in enumerate(
        sorted(block_sweep["version"].unique()),
        start=1,
    ):
        f.write(f"{i_th_version}. Version = {version}\n\n")

        each_version_sweep = block_sweep.loc[block_sweep["version"] == version].copy()

        rows = build_block_sweep_table_rows(
            each_version_sweep=each_version_sweep,
            cpu_ref_time=cpu_ref_time,
        )

        f.write(make_markdown_table(rows, "block_size_sweep"))
        f.write("\n\n")


def write_problem_size_sweep_section(
    f,
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> None:
    f.write("## Problem Size Sweep\n\n")
    f.write(f"DType = {df.iloc[0]['dtype']}\n\n")
    f.write(f"Size of dtype = {df.iloc[0]['size_of_dtype']} bytes\n\n")
    f.write(f"Input pattern: {df['input_pattern'].iloc[0]}\n\n")

    df_cpu_optimized = get_cpu_optimized_rows(df)
    cpu_opt_versions = get_cpu_optimized_versions(df_cpu_optimized)

    for i_th_version_cpu, version in enumerate(cpu_opt_versions, start=1):
        f.write(f"{i_th_version_cpu}. Version = {version}\n\n")

        rows = build_problem_size_table_rows_for_cpu_version(
            df=df,
            df_cpu_optimized=df_cpu_optimized,
            version=version,
        )

        f.write(make_markdown_table(rows, "problem_size_sweep_cpu_optimized"))
        f.write("\n\n")

    for i_th_version, version in enumerate(
        get_kernel_versions(df_gpu),
        start=len(cpu_opt_versions) + 1,
    ):
        f.write(f"{i_th_version}. Version = {version}\n\n")

        rows = build_problem_size_table_rows_for_version(
            df=df,
            df_gpu=df_gpu,
            version=version,
        )

        f.write(make_markdown_table(rows, "problem_size_sweep"))
        f.write("\n\n")


def write_markdown_tables(
    df: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> None:
    config.out_path.parent.mkdir(parents=True, exist_ok=True)

    df_gpu = get_gpu_rows(df)

    with open(config.out_path, "w") as f:
        write_file_header(f, config)

        write_main_benchmark_section(
            f=f,
            df=df,
            df_gpu=df_gpu,
            config=config,
        )

        write_block_size_sweep_section(
            f=f,
            df=df,
            config=config,
        )

        write_problem_size_sweep_section(
            f=f,
            df=df,
            df_gpu=df_gpu,
            config=config,
        )


def render_markdown_tables_to_string(
    df: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> str:
    df_gpu = get_gpu_rows(df)

    buffer = StringIO()

    write_file_header(buffer, config)

    write_main_benchmark_section(
        f=buffer,
        df=df,
        df_gpu=df_gpu,
        config=config,
    )

    write_block_size_sweep_section(
        f=buffer,
        df=df,
        config=config,
    )

    write_problem_size_sweep_section(
        f=buffer,
        df=df,
        df_gpu=df_gpu,
        config=config,
    )

    return buffer.getvalue()


def build_cpu_summary_record(
    row: pd.Series,
    cpu_ref_time: float,
) -> dict[str, Any]:
    version = row["version"]
    time_ms = float(row["avg_ms"])

    category = "cpu_serial" if version == "cpu_serial" else "cpu_optimized"
    speedup = cpu_ref_time / time_ms if time_ms > 0 else float("nan")

    return {
        "version": version,
        "category": category,
        "block_size": "N/A",
        "grid_size": "N/A",
        "h2d_time_ms": 0.0,
        "kernel_time_ms": 0.0,
        "d2h_time_ms": 0.0,
        "cpu_finalize_time_ms": 0.0,
        "post_h2d_time_ms": time_ms,
        "e2e_time_ms": time_ms,
        "post_h2d_bw_GB_s": float(row["bw_GB_s"]),
        "e2e_bw_GB_s": float(row["bw_GB_s"]),
        "post_h2d_gflops": float(row["gflops"]),
        "e2e_gflops": float(row["gflops"]),
        "post_h2d_speedup": speedup,
        "e2e_speedup": speedup,
        "correct": to_bool(row["correct"]),
    }


def build_main_summary_records(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    large_n: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    cpu_rows = get_cpu_rows_at_n(df, large_n)
    if cpu_rows.empty:
        return records

    cpu_serial_rows = cpu_rows.loc[cpu_rows["version"] == "cpu_serial"]
    if cpu_serial_rows.empty:
        raise RuntimeError(f"No cpu_serial row found at n = {large_n}")

    cpu_serial_row = cpu_serial_rows.iloc[0]
    cpu_ref_time = float(cpu_serial_row["avg_ms"])

    # CPU serial + optimized/parallel CPU versions
    for _, row in cpu_rows.iterrows():
        records.append(
            build_cpu_summary_record(
                row=row,
                cpu_ref_time=cpu_ref_time,
            )
        )

    # GPU versions
    for version in get_kernel_versions(df_gpu):
        gpu_rows_for_version_at_large_n = df_gpu.loc[
            (df_gpu["n"] == large_n) & (df_gpu["version"] == version)
        ].copy()

        if gpu_rows_for_version_at_large_n.empty:
            continue

        best_gpu = get_best_kernel_row(gpu_rows_for_version_at_large_n)

        time_h2d = get_h2d_time_at_n(df_gpu, large_n)
        time_kernel = float(best_gpu["avg_ms"])

        time_d2h, _ = get_matching_stage_time(
            gpu_rows_for_version_at_n=gpu_rows_for_version_at_large_n,
            mode="d2h",
            best_kernel_row=best_gpu,
        )

        time_cpu_finalize, correct, _ = get_cpu_finalize_info(
            gpu_rows_for_version_at_n=gpu_rows_for_version_at_large_n,
            best_kernel_row=best_gpu,
        )

        time_cpu_finalize_for_sum = (
            0.0 if time_cpu_finalize == "N/A" else float(time_cpu_finalize)
        )

        time_post_h2d = time_kernel + time_d2h + time_cpu_finalize_for_sum
        time_e2e = time_h2d + time_post_h2d

        useful_bytes = float(cpu_serial_row["bytes"])
        useful_n_flop = num_flop_reduction(large_n)

        bw_post_h2d = bandwidth_GB_s(useful_bytes, time_post_h2d)
        bw_e2e = bandwidth_GB_s(useful_bytes, time_e2e)

        gflops_post_h2d = calculate_gflops(useful_n_flop, time_post_h2d)
        gflops_e2e = calculate_gflops(useful_n_flop, time_e2e)

        speedup_post_h2d = cpu_ref_time / time_post_h2d
        speedup_e2e = cpu_ref_time / time_e2e

        records.append(
            {
                "version": version,
                "category": "gpu",
                "block_size": int(best_gpu["block_size"]),
                "grid_size": int(best_gpu["grid_size"]),
                "h2d_time_ms": time_h2d,
                "kernel_time_ms": time_kernel,
                "d2h_time_ms": time_d2h,
                "cpu_finalize_time_ms": time_cpu_finalize_for_sum,
                "post_h2d_time_ms": time_post_h2d,
                "e2e_time_ms": time_e2e,
                "post_h2d_bw_GB_s": bw_post_h2d,
                "e2e_bw_GB_s": bw_e2e,
                "post_h2d_gflops": gflops_post_h2d,
                "e2e_gflops": gflops_e2e,
                "post_h2d_speedup": speedup_post_h2d,
                "e2e_speedup": speedup_e2e,
                "correct": bool(correct),
            }
        )

    return records


def build_block_sweep_summaries(
    df: pd.DataFrame,
    large_n: int,
) -> list[dict[str, Any]]:
    block_sweep = get_block_sweep_rows(df, large_n)

    summaries: list[dict[str, Any]] = []

    if block_sweep.empty:
        return summaries

    for version in sorted(block_sweep["version"].unique()):
        rows = block_sweep.loc[block_sweep["version"] == version].copy()

        if rows.empty:
            continue

        rows = rows.sort_values("avg_ms")

        best = rows.iloc[0]
        slowest = rows.iloc[-1]

        min_time = float(best["avg_ms"])
        max_time = float(slowest["avg_ms"])

        time_ratio = max_time / min_time if min_time > 0 else float("nan")

        summaries.append(
            {
                "version": version,
                "best_block_size": int(best["block_size"]),
                "best_grid_size": int(best["grid_size"]),
                "best_kernel_time_ms": float(best["avg_ms"]),
                "best_kernel_bw_GB_s": float(best["bw_GB_s"]),
                "best_kernel_gflops": float(best["gflops"]),
                "best_result": float(best["result"]),
                "best_ref": float(best["ref"]),
                "best_abs_error": float(best["abs_error"]),
                "best_rel_error": float(best["rel_error"]),
                "best_correct": to_bool(best["correct"]),
                "kernel_time_min_ms": min_time,
                "kernel_time_max_ms": max_time,
                "kernel_time_ratio": time_ratio,
                "slowest_block_size": int(slowest["block_size"]),
                "num_block_sizes": int(rows["block_size"].nunique()),
            }
        )

    return summaries


# This function builds Jinja2 context, using numeric type instead of string as in the markdown table
def build_problem_size_numeric_row_gpu(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    gpu_version_rows: pd.DataFrame,
    n: int,
) -> dict[str, Any] | None:
    cpu_n = get_cpu_serial_rows_at_n(df, n)
    gpu_n = gpu_version_rows.loc[gpu_version_rows["n"] == n]

    if cpu_n.empty or gpu_n.empty:
        return None

    cpu_row = cpu_n.iloc[0]
    best_gpu_n = get_best_kernel_row(gpu_n)

    best_block_size = int(best_gpu_n["block_size"])
    best_grid_size = int(best_gpu_n["grid_size"])

    size_of_dtype = int(cpu_row["size_of_dtype"])
    single_vector_size_MiB = size_of_dtype * int(n) / (1 << 20)

    cpu_time_n = float(cpu_row["avg_ms"])
    cpu_bw_n = float(cpu_row["bw_GB_s"])

    ref = (
        cpu_row["ref"]
        if "ref" in cpu_row and pd.notna(cpu_row["ref"])
        else cpu_row["result"]
    )

    time_h2d_n = get_h2d_time_at_n(df_gpu, n)
    time_kernel = float(best_gpu_n["avg_ms"])

    time_d2h, _ = get_matching_stage_time(
        gpu_rows_for_version_at_n=gpu_n,
        mode="d2h",
        best_kernel_row=best_gpu_n,
    )

    time_cpu_finalize, correct, result = get_cpu_finalize_info(
        gpu_rows_for_version_at_n=gpu_n,
        best_kernel_row=best_gpu_n,
    )

    time_cpu_finalize_for_sum = (
        0.0 if time_cpu_finalize == "N/A" else float(time_cpu_finalize)
    )

    time_post_h2d = time_kernel + time_d2h + time_cpu_finalize_for_sum
    time_e2e = time_h2d_n + time_post_h2d

    speedup_post_h2d = cpu_time_n / time_post_h2d
    speedup_e2e = cpu_time_n / time_e2e

    useful_bytes = float(cpu_row["bytes"])
    bw_post_h2d = bandwidth_GB_s(useful_bytes, time_post_h2d)
    bw_e2e = bandwidth_GB_s(useful_bytes, time_e2e)

    useful_n_flop = num_flop_reduction(n)
    gflops_post_h2d = calculate_gflops(useful_n_flop, time_post_h2d)
    gflops_e2e = calculate_gflops(useful_n_flop, time_e2e)

    return {
        "n": int(n),
        "single_vector_size_MiB": single_vector_size_MiB,
        "cpu_time_ms": cpu_time_n,
        "cpu_bw_GB_s": cpu_bw_n,
        "best_block_size": best_block_size,
        "best_grid_size": best_grid_size,
        "kernel_time_ms": time_kernel,
        "h2d_time_ms": time_h2d_n,
        "d2h_time_ms": time_d2h,
        "cpu_finalize_time_ms": time_cpu_finalize_for_sum,
        "post_h2d_time_ms": time_post_h2d,
        "e2e_time_ms": time_e2e,
        "post_h2d_bw_GB_s": bw_post_h2d,
        "e2e_bw_GB_s": bw_e2e,
        "post_h2d_gflops": gflops_post_h2d,
        "e2e_gflops": gflops_e2e,
        "result": float(result),
        "ref": float(ref),
        "correct": bool(correct),
        "post_h2d_speedup": speedup_post_h2d,
        "e2e_speedup": speedup_e2e,
    }


def build_problem_size_numeric_records_for_version(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    version: str,
) -> list[dict[str, Any]]:
    gpu_version_rows = df_gpu.loc[df_gpu["version"] == version].copy()

    records: list[dict[str, Any]] = []

    for n in sorted(gpu_version_rows["n"].unique()):
        row = build_problem_size_numeric_row_gpu(
            df=df,
            df_gpu=df_gpu,
            gpu_version_rows=gpu_version_rows,
            n=int(n),
        )

        if row is not None:
            records.append(row)

    return records


def build_problem_size_summaries(
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for version in get_kernel_versions(df_gpu):
        records = build_problem_size_numeric_records_for_version(
            df=df,
            df_gpu=df_gpu,
            version=version,
        )

        if not records:
            continue

        records = sorted(records, key=lambda r: r["n"])

        first = records[0]
        last = records[-1]

        best_e2e = min(records, key=lambda r: r["e2e_time_ms"])
        best_post_h2d = min(records, key=lambda r: r["post_h2d_time_ms"])

        summaries.append(
            {
                "version": version,
                "num_sizes": len(records),
                "small": first,
                "large": last,
                "best_e2e": best_e2e,
                "best_post_h2d": best_post_h2d,
                "post_h2d_speedup_growth": (
                    last["post_h2d_speedup"] / first["post_h2d_speedup"]
                    if first["post_h2d_speedup"] > 0
                    else float("nan")
                ),
                "e2e_speedup_growth": (
                    last["e2e_speedup"] / first["e2e_speedup"]
                    if first["e2e_speedup"] > 0
                    else float("nan")
                ),
            }
        )

    return summaries


# This function builds Jinja2 context, using numeric type instead of string as in the markdown table
def build_problem_size_numeric_row_cpu_optimized(
    df: pd.DataFrame,
    cpu_version_rows: pd.DataFrame,
    n: int,
) -> dict[str, Any] | None:
    cpu_serial_n = get_cpu_serial_rows_at_n(df, n)
    cpu_optimized_n = cpu_version_rows.loc[cpu_version_rows["n"] == n]

    if cpu_serial_n.empty or cpu_optimized_n.empty:
        return None

    cpu_serial_row = cpu_serial_n.iloc[0]
    cpu_optimized_row = cpu_optimized_n.iloc[0]

    size_of_dtype = int(cpu_serial_row["size_of_dtype"])
    single_vector_size_MiB = size_of_dtype * int(n) / (1 << 20)

    cpu_serial_time_ms = float(cpu_serial_row["avg_ms"])
    cpu_serial_bw_GB_s = float(cpu_serial_row["bw_GB_s"])
    cpu_serial_gflops = float(cpu_serial_row["gflops"])

    cpu_optimized_time_ms = float(cpu_optimized_row["avg_ms"])
    cpu_optimized_bw_GB_s = float(cpu_optimized_row["bw_GB_s"])
    cpu_optimized_gflops = float(cpu_optimized_row["gflops"])

    speedup = cpu_serial_time_ms / cpu_optimized_time_ms

    ref = (
        cpu_serial_row["ref"]
        if "ref" in cpu_serial_row and pd.notna(cpu_serial_row["ref"])
        else cpu_serial_row["result"]
    )

    return {
        "n": int(n),
        "single_vector_size_MiB": single_vector_size_MiB,
        "cpu_serial_time_ms": cpu_serial_time_ms,
        "cpu_serial_bw_GB_s": cpu_serial_bw_GB_s,
        "cpu_serial_gflops": cpu_serial_gflops,
        "cpu_optimized_time_ms": cpu_optimized_time_ms,
        "cpu_optimized_bw_GB_s": cpu_optimized_bw_GB_s,
        "cpu_optimized_gflops": cpu_optimized_gflops,
        "result": float(cpu_optimized_row["result"]),
        "ref": float(ref),
        "correct": to_bool(cpu_optimized_row["correct"]),
        "speedup": speedup,
    }


def build_problem_size_numeric_records_for_cpu_optimized_version(
    df: pd.DataFrame,
    df_cpu_optimized: pd.DataFrame,
    version: str,
) -> list[dict[str, Any]]:
    cpu_version_rows = df_cpu_optimized.loc[
        df_cpu_optimized["version"] == version
    ].copy()

    records: list[dict[str, Any]] = []

    for n in sorted(cpu_version_rows["n"].unique()):
        row = build_problem_size_numeric_row_cpu_optimized(
            df=df,
            cpu_version_rows=cpu_version_rows,
            n=int(n),
        )

        if row is not None:
            records.append(row)

    return records


def build_cpu_optimized_problem_size_summaries(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    df_cpu_optimized = get_cpu_optimized_rows(df)

    summaries: list[dict[str, Any]] = []

    if df_cpu_optimized.empty:
        return summaries

    for version in get_cpu_optimized_versions(df_cpu_optimized):
        records = build_problem_size_numeric_records_for_cpu_optimized_version(
            df=df,
            df_cpu_optimized=df_cpu_optimized,
            version=version,
        )

        if not records:
            continue

        records = sorted(records, key=lambda r: r["n"])

        first = records[0]
        last = records[-1]
        best = min(records, key=lambda r: r["cpu_optimized_time_ms"])

        summaries.append(
            {
                "version": version,
                "num_sizes": len(records),
                "small": first,
                "large": last,
                "best": best,
                "speedup_growth": (
                    last["speedup"] / first["speedup"]
                    if first["speedup"] > 0
                    else float("nan")
                ),
            }
        )

    return summaries


# This function does formatting for Jinja2 context
def format_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)

    for key in [
        "h2d_time_ms",
        "kernel_time_ms",
        "d2h_time_ms",
        "cpu_finalize_time_ms",
        "post_h2d_time_ms",
        "e2e_time_ms",
    ]:
        if isinstance(out.get(key), (int, float)):
            out[key] = format_float(out[key], 3)

    for key in [
        "post_h2d_bw_GB_s",
        "e2e_bw_GB_s",
        "post_h2d_gflops",
        "e2e_gflops",
    ]:
        if isinstance(out.get(key), (int, float)):
            out[key] = format_float(out[key], 2)

    for key in [
        "post_h2d_speedup",
        "e2e_speedup",
    ]:
        if isinstance(out.get(key), (int, float)):
            out[key] = format_speedup(out[key])

    return out


def format_block_sweep_summary(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)

    for key in [
        "best_kernel_time_ms",
        "best_kernel_bw_GB_s",
        "best_kernel_gflops",
        "kernel_time_min_ms",
        "kernel_time_max_ms",
        "kernel_time_ratio",
        "best_result",
        "best_ref",
        "best_abs_error",
        "best_rel_error",
    ]:
        if isinstance(out.get(key), (int, float)):
            if key in [
                "best_kernel_time_ms",
                "kernel_time_min_ms",
                "kernel_time_max_ms",
            ]:
                out[key] = format_ms(out[key])
            elif key in ["kernel_time_ratio"]:
                out[key] = format_float(out[key], 2)
            elif key in ["best_abs_error", "best_rel_error"]:
                out[key] = f"{out[key]:.3e}"
            else:
                out[key] = format_float(out[key], 2)

    return out


def format_problem_size_record(r: dict[str, Any]) -> dict[str, Any]:
    out = dict(r)
    out["n"] = f"{int(out['n']):,}"

    for key in [
        "single_vector_size_MiB",
        "cpu_time_ms",
        "cpu_bw_GB_s",
        "kernel_time_ms",
        "h2d_time_ms",
        "d2h_time_ms",
        "cpu_finalize_time_ms",
        "post_h2d_time_ms",
        "e2e_time_ms",
        "post_h2d_bw_GB_s",
        "e2e_bw_GB_s",
        "post_h2d_gflops",
        "e2e_gflops",
        "post_h2d_speedup",
        "e2e_speedup",
        "result",
        "ref",
    ]:
        if isinstance(out.get(key), (int, float)):
            if key.endswith("_speedup"):
                out[key] = format_speedup(out[key])
            elif key in ["single_vector_size_MiB"]:
                out[key] = format_float(out[key], 2)
            elif key in ["result", "ref"]:
                out[key] = format_float(out[key], 3)
            else:
                out[key] = format_float(out[key], 3)

    return out


def format_problem_size_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)

    out["small"] = format_problem_size_record(summary["small"])
    out["large"] = format_problem_size_record(summary["large"])
    out["best_e2e"] = format_problem_size_record(summary["best_e2e"])
    out["best_post_h2d"] = format_problem_size_record(summary["best_post_h2d"])

    out["post_h2d_speedup_growth"] = format_float(summary["post_h2d_speedup_growth"], 2)
    out["e2e_speedup_growth"] = format_float(summary["e2e_speedup_growth"], 2)

    return out


def format_cpu_optimized_problem_size_record(
    r: dict[str, Any],
) -> dict[str, Any]:
    out = dict(r)
    out["n"] = f"{int(out['n']):,}"

    for key in [
        "single_vector_size_MiB",
        "cpu_serial_time_ms",
        "cpu_serial_bw_GB_s",
        "cpu_serial_gflops",
        "cpu_optimized_time_ms",
        "cpu_optimized_bw_GB_s",
        "cpu_optimized_gflops",
        "result",
        "ref",
        "speedup",
    ]:
        if isinstance(out.get(key), (int, float)):
            if key == "single_vector_size_MiB":
                out[key] = format_float(out[key], 2)
            elif key == "speedup":
                out[key] = format_speedup(out[key])
            elif key.endswith("_time_ms"):
                out[key] = format_ms(out[key])
            elif key in ["result", "ref"]:
                out[key] = format_float(out[key], 3)
            else:
                out[key] = format_float(out[key], 2)

    return out


def format_cpu_optimized_problem_size_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    out = dict(summary)

    out["small"] = format_cpu_optimized_problem_size_record(summary["small"])
    out["large"] = format_cpu_optimized_problem_size_record(summary["large"])
    out["best"] = format_cpu_optimized_problem_size_record(summary["best"])
    out["speedup_growth"] = format_float(summary["speedup_growth"], 2)

    return out


# This function builds non-correct cases in ninja constext
def build_correctness_failures_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "correct" not in df.columns:
        return []

    bad = df.loc[~df["correct"]].copy()

    failures: list[dict[str, Any]] = []

    for _, row in bad.iterrows():
        failures.append(
            {
                "source": "csv",
                "version": row.get("version", "N/A"),
                "mode": row.get("mode", "N/A"),
                "n": (
                    f"{int(row['n']):,}" if "n" in row and pd.notna(row["n"]) else "N/A"
                ),
                "block_size": (
                    int(row["block_size"])
                    if "block_size" in row and pd.notna(row["block_size"])
                    else "N/A"
                ),
                "grid_size": (
                    int(row["grid_size"])
                    if "grid_size" in row and pd.notna(row["grid_size"])
                    else "N/A"
                ),
                "result": (
                    f"{float(row['result']):.6g}"
                    if "result" in row and pd.notna(row["result"])
                    else "N/A"
                ),
                "ref": (
                    f"{float(row['ref']):.6g}"
                    if "ref" in row and pd.notna(row["ref"])
                    else "N/A"
                ),
                "abs_error": (
                    f"{float(row['abs_error']):.6g}"
                    if "abs_error" in row and pd.notna(row["abs_error"])
                    else "N/A"
                ),
                "rel_error": (
                    f"{float(row['rel_error']):.6g}"
                    if "rel_error" in row and pd.notna(row["rel_error"])
                    else "N/A"
                ),
            }
        )

    return failures


# This function build ninja2 contexts
def build_report_context(
    df: pd.DataFrame,
    config: BenchmarkReportConfig,
) -> dict[str, Any]:
    df_gpu = get_gpu_rows(df)
    large_n = config.large_n

    tables_markdown = render_markdown_tables_to_string(df, config)

    records = build_main_summary_records(
        df=df,
        df_gpu=df_gpu,
        large_n=large_n,
    )

    if not records:
        raise RuntimeError("No summary records were generated.")

    cpu_record = next(r for r in records if r["category"] == "cpu_serial")
    cpu_formatted = format_summary_record(cpu_record)
    cpu_formatted["time_ms"] = cpu_formatted["post_h2d_time_ms"]

    cpu_optimized_records_raw = [r for r in records if r["category"] == "cpu_optimized"]

    cpu_optimized_records = [
        format_summary_record(r) for r in cpu_optimized_records_raw
    ]

    if cpu_optimized_records_raw:
        best_cpu_optimized = min(
            cpu_optimized_records_raw,
            key=lambda r: r["e2e_time_ms"],
        )
        best_cpu_optimized_formatted = format_summary_record(best_cpu_optimized)
    else:
        best_cpu_optimized_formatted = None

    gpu_records = [r for r in records if r["category"] == "gpu"]

    if not gpu_records:
        raise RuntimeError("No GPU records were generated.")

    best_post_h2d = min(gpu_records, key=lambda r: r["post_h2d_time_ms"])
    best_post_h2d_formatted = format_summary_record(best_post_h2d)

    best_e2e = min(gpu_records, key=lambda r: r["e2e_time_ms"])
    best_e2e_formatted = format_summary_record(best_e2e)

    h2d_fraction = (
        float(best_e2e["h2d_time_ms"]) / float(best_e2e["e2e_time_ms"]) * 100.0
        if float(best_e2e["e2e_time_ms"]) > 0
        else 0.0
    )
    best_e2e_formatted["h2d_fraction_percent"] = format_float(h2d_fraction, 1)

    formatted_records = [format_summary_record(r) for r in records]

    versions = {r["version"]: format_summary_record(r) for r in records}

    block_sweep_summaries_raw = build_block_sweep_summaries(
        df=df,
        large_n=large_n,
    )

    block_sweep_summaries = [
        format_block_sweep_summary(r) for r in block_sweep_summaries_raw
    ]

    problem_size_summaries_raw = build_problem_size_summaries(
        df=df,
        df_gpu=df_gpu,
    )

    problem_size_summaries = [
        format_problem_size_summary(r) for r in problem_size_summaries_raw
    ]

    cpu_optimized_problem_size_summaries_raw = (
        build_cpu_optimized_problem_size_summaries(df)
    )

    cpu_optimized_problem_size_summaries = [
        format_cpu_optimized_problem_size_summary(r)
        for r in cpu_optimized_problem_size_summaries_raw
    ]

    dtype = df.iloc[0]["dtype"]
    size_of_dtype = int(df.iloc[0]["size_of_dtype"])
    single_vector_size_mib = large_n * size_of_dtype / (1 << 20)

    correctness_failures = build_correctness_failures_from_df(df)

    context = {
        "csv_path": str(config.csv_path),
        "md_table_out_path": str(config.md_table_rel_path),
        "csv_rel_path": str(config.csv_rel_path),
        "large_n": large_n,
        "large_n_formatted": f"{large_n:,}",
        "dtype": dtype,
        "size_of_dtype": size_of_dtype,
        "single_vector_size_mib": format_float(single_vector_size_mib, 2),
        "input_pattern": config.main_input_pattern,
        "cpu": cpu_formatted,
        "cpu_optimized_records": cpu_optimized_records,
        "best_cpu_optimized": best_cpu_optimized_formatted,
        "cpu_optimized_problem_size_summaries": cpu_optimized_problem_size_summaries,
        "best_post_h2d": best_post_h2d_formatted,
        "best_e2e": best_e2e_formatted,
        "records": formatted_records,
        "versions": versions,
        "block_sweep_summaries": block_sweep_summaries,
        "problem_size_summaries": problem_size_summaries,
        "correctness_failures": correctness_failures,
        "tables_markdown": tables_markdown,
    }

    return context


def render_report(
    df: pd.DataFrame,
    config: BenchmarkReportConfig,
) -> None:
    context = build_report_context(df, config)

    env = Environment(
        loader=FileSystemLoader(config.template_dir),
        autoescape=False,
    )

    template = env.get_template(config.template_name)
    rendered = template.render(**context)

    config.report_out_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_out_path.write_text(rendered)

    print(f"Wrote {config.report_out_path}")


def main() -> None:
    config = BenchmarkReportConfig()

    df_all = read_benchmark_csv(config.csv_path)
    df = filter_main_input_pattern(df_all, config)

    if df.empty:
        raise RuntimeError(
            f"No rows found for input_pattern = {config.main_input_pattern}"
        )

    write_markdown_tables(
        df=df,
        config=config,
    )

    print(f"Wrote {config.out_path}")

    render_report(
        df=df,
        config=config,
    )


if __name__ == "__main__":
    main()
