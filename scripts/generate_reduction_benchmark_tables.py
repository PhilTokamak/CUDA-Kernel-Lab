from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BenchmarkTableConfig:
    csv_path: Path = Path("results/data/reduction.csv")
    out_path: Path = Path("results/markdown_tables/reduction_bench_tables.md")
    kernel_name: str = "Reduction"
    large_n: int = 1 << 24


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
        "Speedup",
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
}


def bandwidth_GB_s(num_bytes: float, time_ms: float) -> float:
    return float(num_bytes) / (time_ms / 1000.0 * 1e9)


def calculate_gflops(n_flop: float, time_ms: float) -> float:
    return n_flop / (time_ms / 1000.0 * 1e9)


def num_flop_reduction(n: int) -> int:
    return max(int(n) - 1, 0)


def format_ms(value: Any) -> str:
    return "N/A" if value == "N/A" else f"{float(value):.3f}"


def format_float(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def format_speedup(value: Any) -> str:
    return "" if value == "" else f"{float(value):.2f}x"


def read_benchmark_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def make_markdown_table(rows: list[dict[str, Any]], table_name: str) -> str:
    columns = TABLE_COLUMNS[table_name]
    table = pd.DataFrame(rows, columns=columns)
    return table.to_markdown(index=False)


def get_cpu_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[(df["n"] == int(n)) & (df["mode"] == "cpu")]


def get_cpu_serial_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[
        (df["n"] == int(n)) & (df["mode"] == "cpu") & (df["version"] == "cpu_serial")
    ]


def get_gpu_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Extract all gpu versions
    return df.loc[~df["version"].str.contains("cpu", na=False)].copy()


def is_h2d_only_version(df_gpu: pd.DataFrame, version: str) -> bool:
    return not df_gpu.loc[
        (df_gpu["version"] == version) & (df_gpu["mode"] == "h2d")
    ].empty


def get_h2d_time_at_n(df_gpu: pd.DataFrame, n: int) -> float:
    return df_gpu.loc[(df_gpu["n"] == int(n)) & (df_gpu["mode"] == "h2d")].iloc[0][
        "avg_ms"
    ]


def get_kernel_versions(df_gpu: pd.DataFrame) -> list[str]:
    versions: list[str] = []

    for version in sorted(df_gpu["version"].unique()):
        # Skip h2d since its version is "all_gpu_version" and this "version" doesn't have an kernel
        if is_h2d_only_version(df_gpu, version):
            continue

        versions.append(version)

    return versions


def get_best_kernel_row(rows: pd.DataFrame) -> pd.Series:
    # choose fastest GPU kernel version among block sizes
    return rows.loc[rows["mode"] == "cuda_kernel"].sort_values("avg_ms").iloc[0]


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
    return row["avg_ms"], row


def build_cpu_main_row(
    cpu_rows: pd.DataFrame,
) -> tuple[dict[str, Any] | None, float | None]:
    if cpu_rows.empty:
        return None, None

    row = cpu_rows.iloc[0]
    cpu_ref_time = cpu_rows.loc[cpu_rows["version"] == "cpu_serial", "avg_ms"].iloc[0]

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
        "Correct": row["correct"],
        "Post-H2D Speedup": format_speedup(1.00),
        "E2E Speedup": format_speedup(1.00),
    }

    return main_row, cpu_ref_time


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
        return "N/A", best_kernel_row["correct"], best_kernel_row["result"]

    return (
        time_cpu_finalize,
        row_cpu_finalize["correct"],
        row_cpu_finalize["result"],
    )


def build_main_gpu_row(
    cpu_rows: pd.DataFrame,
    cpu_ref_time: float | None,
    df_gpu: pd.DataFrame,
    gpu_rows_for_version_at_large_n: pd.DataFrame,
    version: str,
    large_n: int,
) -> dict[str, Any]:
    best_gpu = get_best_kernel_row(gpu_rows_for_version_at_large_n)

    time_h2d = get_h2d_time_at_n(df_gpu, large_n)
    time_kernel = best_gpu["avg_ms"]

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

    useful_bytes = cpu_rows.iloc[0]["bytes"]
    bw_post_h2d = bandwidth_GB_s(useful_bytes, time_post_h2d)
    bw_e2e = bandwidth_GB_s(useful_bytes, time_e2e)

    # Attention: this is reduction specific, has to be changed for other kernel
    useful_n_flop = num_flop_reduction(large_n)
    gflops_post_h2d = calculate_gflops(useful_n_flop, time_post_h2d)
    gflops_e2e = calculate_gflops(useful_n_flop, time_e2e)

    return {
        "Version": version,
        "Block Size": best_gpu["block_size"],
        "Grid Size": best_gpu["grid_size"],
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
    cpu_row, cpu_ref_time = build_cpu_main_row(cpu_rows)

    if cpu_row is not None:
        rows.append(cpu_row)

    for version in get_kernel_versions(df_gpu):
        gpu_rows_for_version_at_large_n = df_gpu.loc[
            (df_gpu["n"] == large_n) & (df_gpu["version"] == version)
        ].copy()

        if gpu_rows_for_version_at_large_n.empty:
            continue

        # Add data of each kernel versions to the main table
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
    return df.loc[
        (df["n"] == large_n)
        & (~df["version"].str.contains("cpu", na=False))
        & (df["mode"] == "cuda_kernel")
    ].copy()


def build_block_sweep_table_rows(
    each_version_sweep: pd.DataFrame,
    cpu_ref_time: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for _, row in each_version_sweep.sort_values("block_size").iterrows():
        speedup = cpu_ref_time / row["avg_ms"]

        rows.append(
            {
                "Block Size": row["block_size"],
                "Grid Size": row["grid_size"],
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
                "Correct": row["correct"],
                "Speedup": format_speedup(speedup),
            }
        )

    return rows


def build_problem_size_row(
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
    best_block_size = best_gpu_n["block_size"]
    best_grid_size = best_gpu_n["grid_size"]

    single_vector_size_MiB = cpu_row["size_of_dtype"] * n / (1 << 20)

    cpu_time_n = cpu_row["avg_ms"]
    bw_cpu_n = cpu_row["bw_GB_s"]
    ref = cpu_row["result"]

    time_h2d_n = get_h2d_time_at_n(df_gpu, n)
    time_kernel = best_gpu_n["avg_ms"]

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

    useful_bytes = cpu_n.iloc[0]["bytes"]
    bw_post_h2d = bandwidth_GB_s(useful_bytes, time_post_h2d)
    bw_e2e = bandwidth_GB_s(useful_bytes, time_e2e)

    # Attention: this is reduction specific, has to be changed for other kernel.
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
        row = build_problem_size_row(
            df=df,
            df_gpu=df_gpu,
            gpu_version_rows=gpu_version_rows,
            n=n,
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

    # Main results at large N
    f.write("## Main Benchmark Results at Large N\n\n")
    f.write(f"N = {large_n:,}\n\n")
    f.write(f"DType = {df.iloc[0]['dtype']}\n\n")
    f.write(f"Size of dtype = {df.iloc[0]['size_of_dtype']} bytes\n\n")

    single_vector_size_MiB = large_n * df.iloc[0]["size_of_dtype"] / (1 << 20)
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

    # Block size sweep at largest N
    f.write("## CUDA Block Size Sweep\n\n")
    f.write(f"N = {large_n:,}\n\n")

    block_sweep = get_block_sweep_rows(df, large_n)

    f.write(f"DType = {block_sweep.iloc[0]['dtype']}\n\n")
    f.write(f"Size of dtype = {df.iloc[0]['size_of_dtype']} bytes\n\n")

    cpu_rows = get_cpu_rows_at_n(df, large_n)
    cpu_ref_time = cpu_rows.loc[cpu_rows["version"] == "cpu_serial", "avg_ms"].iloc[0]

    for i_th_version, version in enumerate(
        sorted(block_sweep["version"].unique()), start=1
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
    # Problem size sweep (Use CPU and best GPU per N)
    f.write("## Problem Size Sweep\n\n")
    f.write(f"DType = {df.iloc[0]['dtype']}\n\n")
    f.write(f"Size of dtype = {df.iloc[0]['size_of_dtype']} bytes\n\n")

    for i_th_version, version in enumerate(get_kernel_versions(df_gpu), start=1):
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


def main() -> None:
    config = BenchmarkTableConfig()

    df = read_benchmark_csv(config.csv_path)

    write_markdown_tables(
        df=df,
        config=config,
    )

    print(f"Wrote {config.out_path}")


if __name__ == "__main__":
    main()
