from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BenchmarkTableConfig:
    csv_path: Path = Path("results/data/vector_add_ref.csv")
    out_path: Path = Path("results/markdown_tables/vector_add_bench_tables.md")
    kernel_name: str = "Vector Add"
    # If large_n is None, use the largest N in the CSV file.
    large_n: int | None = None


FLOAT_COLUMNS = [
    "avg_ms",
    "min_ms",
    "max_ms",
    "std_ms",
    "bw_GB_s",
    "gflops",
]

TABLE_COLUMNS = {
    "main_table": [
        "Version",
        "Mode",
        "Effective Data Moved [MiB]",
        "Block Size",
        "Avg Time [ms]",
        "Min Time [ms]",
        "Max Time [ms]",
        "Std Dev",
        "Effective BW [GB/s]",
        "GFLOP/s",
        "Correct",
        "Speedup",
    ],
    "block_size_sweep": [
        "Block Size",
        "Avg Time [ms]",
        "Min Time [ms]",
        "Max Time [ms]",
        "Std Dev",
        "Effective BW [GB/s]",
        "GFLOP/s",
        "Correct",
        "Speedup",
    ],
    "problem_size_sweep": [
        "N",
        "Single Vector Size [MiB]",
        "CPU Time [ms]",
        "CPU BW [GB/s]",
        "Best GPU Block",
        "GPU Kernel Time [ms]",
        "GPU Kernel Effective BW [GB/s]",
        "Kernel Speedup",
        "GPU E2E Time [ms]",
        "E2E Speedup",
    ],
}


def bandwidth_GB_s(num_bytes: float, time_ms: float) -> float:
    return float(num_bytes) / (time_ms / 1000.0 * 1e9)


def calculate_gflops(n_flop: float, time_ms: float) -> float:
    return n_flop / (time_ms / 1000.0 * 1e9)


def format_ms(value: Any) -> str:
    return "N/A" if value == "N/A" else f"{float(value):.3f}"


def format_float(value: Any, digits: int = 2) -> str:
    return "N/A" if value == "N/A" else f"{float(value):.{digits}f}"


def format_speedup(value: Any) -> str:
    return "N/A" if value == "N/A" else f"{float(value):.2f}x"


def bytes_to_MiB(num_bytes: float) -> float:
    return float(num_bytes) / float(1 << 20)


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


def get_large_n(df: pd.DataFrame, config: BenchmarkTableConfig) -> int:
    if config.large_n is not None:
        return int(config.large_n)

    return int(df["n"].max())


def get_cpu_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[(df["n"] == int(n)) & (df["mode"] == "cpu")]


def get_cpu_serial_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[
        (df["n"] == int(n)) & (df["mode"] == "cpu") & (df["version"] == "cpu_serial")
    ]


def get_gpu_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Extract all gpu versions
    return df.loc[~df["version"].str.contains("cpu", na=False)].copy()


def get_cuda_kernel_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.loc[(df["n"] == int(n)) & (df["mode"] == "cuda_kernel")].copy()


def get_stage_rows_at_n(df: pd.DataFrame, n: int, mode: str) -> pd.DataFrame:
    return df.loc[(df["n"] == int(n)) & (df["mode"] == mode)].copy()


def get_h2d_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return get_stage_rows_at_n(df, n, "h2d")


def get_d2h_rows_at_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return get_stage_rows_at_n(df, n, "d2h")


def get_kernel_versions(df_gpu: pd.DataFrame) -> list[str]:
    return sorted(df_gpu.loc[df_gpu["mode"] == "cuda_kernel", "version"].unique())


def get_best_kernel_row(rows: pd.DataFrame) -> pd.Series:
    # choose fastest GPU kernel version among block sizes
    return rows.loc[rows["mode"] == "cuda_kernel"].sort_values("avg_ms").iloc[0]


def get_cpu_ref_time(cpu_rows: pd.DataFrame) -> float | None:
    if cpu_rows.empty:
        return None

    return cpu_rows.loc[cpu_rows["version"] == "cpu_serial", "avg_ms"].iloc[0]


def build_cpu_main_row(
    cpu_rows: pd.DataFrame,
) -> tuple[dict[str, Any] | None, float | None]:
    if cpu_rows.empty:
        return None, None

    row = cpu_rows.iloc[0]
    cpu_ref_time = get_cpu_ref_time(cpu_rows)

    data_moved_MiB = bytes_to_MiB(row["bytes"])

    main_row = {
        "Version": row["version"],
        "Mode": row["mode"],
        "Effective Data Moved [MiB]": format_float(data_moved_MiB, 2),
        "Block Size": "N/A",
        "Avg Time [ms]": format_ms(row["avg_ms"]),
        "Min Time [ms]": format_ms(row["min_ms"]),
        "Max Time [ms]": format_ms(row["max_ms"]),
        "Std Dev": format_ms(row["std_ms"]),
        "Effective BW [GB/s]": format_float(row["bw_GB_s"], 2),
        "GFLOP/s": format_float(row["gflops"], 2),
        "Correct": row["correct"],
        "Speedup": format_speedup(1.00),
    }

    return main_row, cpu_ref_time


def build_best_gpu_kernel_main_row(
    gpu_kernel_rows_at_large_n: pd.DataFrame,
    cpu_ref_time: float | None,
) -> dict[str, Any] | None:
    if gpu_kernel_rows_at_large_n.empty:
        return None

    best_gpu = get_best_kernel_row(gpu_kernel_rows_at_large_n)
    data_moved_MiB = bytes_to_MiB(best_gpu["bytes"])

    if cpu_ref_time is not None:
        speedup = cpu_ref_time / best_gpu["avg_ms"]
    else:
        speedup = "N/A"

    return {
        "Version": best_gpu["version"],
        "Mode": best_gpu["mode"],
        "Effective Data Moved [MiB]": format_float(data_moved_MiB, 2),
        "Block Size": best_gpu["block_size"],
        "Avg Time [ms]": format_ms(best_gpu["avg_ms"]),
        "Min Time [ms]": format_ms(best_gpu["min_ms"]),
        "Max Time [ms]": format_ms(best_gpu["max_ms"]),
        "Std Dev": format_ms(best_gpu["std_ms"]),
        "Effective BW [GB/s]": format_float(best_gpu["bw_GB_s"], 2),
        "GFLOP/s": format_float(best_gpu["gflops"], 2),
        "Correct": best_gpu["correct"],
        "Speedup": format_speedup(speedup),
    }


def build_copy_main_row(copy_rows_at_large_n: pd.DataFrame) -> dict[str, Any] | None:
    if copy_rows_at_large_n.empty:
        return None

    row = copy_rows_at_large_n.iloc[0]
    data_moved_MiB = bytes_to_MiB(row["bytes"])

    return {
        "Version": row["version"],
        "Mode": row["mode"],
        "Effective Data Moved [MiB]": format_float(data_moved_MiB, 2),
        "Block Size": "N/A",
        "Avg Time [ms]": format_ms(row["avg_ms"]),
        "Min Time [ms]": format_ms(row["min_ms"]),
        "Max Time [ms]": format_ms(row["max_ms"]),
        "Std Dev": format_ms(row["std_ms"]),
        "Effective BW [GB/s]": format_float(row["bw_GB_s"], 2),
        "GFLOP/s": "N/A",
        "Correct": "N/A",
        "Speedup": "N/A",
    }


def build_e2e_main_row(
    h2d_rows_at_large_n: pd.DataFrame,
    d2h_rows_at_large_n: pd.DataFrame,
    gpu_kernel_rows_at_large_n: pd.DataFrame,
    cpu_ref_time: float | None,
) -> dict[str, Any] | None:
    # Add synthetic GPU end-to-end row: H2D + best GPU kernel + D2H
    if (
        h2d_rows_at_large_n.empty
        or d2h_rows_at_large_n.empty
        or gpu_kernel_rows_at_large_n.empty
    ):
        return None

    best_gpu = get_best_kernel_row(gpu_kernel_rows_at_large_n)

    h2d_row = h2d_rows_at_large_n.iloc[0]
    d2h_row = d2h_rows_at_large_n.iloc[0]

    h2d_time = h2d_row["avg_ms"]
    kernel_time = best_gpu["avg_ms"]
    d2h_time = d2h_row["avg_ms"]

    e2e_time = h2d_time + kernel_time + d2h_time

    data_moved_bytes = h2d_row["bytes"] + d2h_row["bytes"]
    data_moved_MiB = bytes_to_MiB(data_moved_bytes)
    bw_GB_s = bandwidth_GB_s(data_moved_bytes, e2e_time)

    if cpu_ref_time is not None:
        e2e_speedup = cpu_ref_time / e2e_time
    else:
        e2e_speedup = "N/A"

    return {
        "Version": f"{best_gpu['version']}_e2e",
        "Mode": "h2d + kernel + d2h",
        "Effective Data Moved [MiB]": format_float(data_moved_MiB, 2),
        "Block Size": best_gpu["block_size"],
        "Avg Time [ms]": format_ms(e2e_time),
        "Min Time [ms]": "N/A",
        "Max Time [ms]": "N/A",
        "Std Dev": "N/A",
        "Effective BW [GB/s]": format_float(bw_GB_s, 2),
        "GFLOP/s": "N/A",
        "Correct": "N/A",
        "Speedup": format_speedup(e2e_speedup),
    }


def build_main_table_rows(
    df: pd.DataFrame,
    large_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    cpu_rows = get_cpu_rows_at_n(df, large_n)
    cpu_row, cpu_ref_time = build_cpu_main_row(cpu_rows)

    if cpu_row is not None:
        rows.append(cpu_row)

    gpu_kernel_rows_at_large_n = get_cuda_kernel_rows_at_n(df, large_n)

    best_gpu_row = build_best_gpu_kernel_main_row(
        gpu_kernel_rows_at_large_n=gpu_kernel_rows_at_large_n,
        cpu_ref_time=cpu_ref_time,
    )
    if best_gpu_row is not None:
        rows.append(best_gpu_row)

    h2d_rows_at_large_n = get_h2d_rows_at_n(df, large_n)
    d2h_rows_at_large_n = get_d2h_rows_at_n(df, large_n)

    h2d_row = build_copy_main_row(h2d_rows_at_large_n)
    if h2d_row is not None:
        rows.append(h2d_row)

    d2h_row = build_copy_main_row(d2h_rows_at_large_n)
    if d2h_row is not None:
        rows.append(d2h_row)

    e2e_row = build_e2e_main_row(
        h2d_rows_at_large_n=h2d_rows_at_large_n,
        d2h_rows_at_large_n=d2h_rows_at_large_n,
        gpu_kernel_rows_at_large_n=gpu_kernel_rows_at_large_n,
        cpu_ref_time=cpu_ref_time,
    )
    if e2e_row is not None:
        rows.append(e2e_row)

    return rows


def get_block_sweep_rows(
    df: pd.DataFrame,
    large_n: int,
    version: str,
) -> pd.DataFrame:
    return df.loc[
        (df["n"] == int(large_n))
        & (df["version"] == version)
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
                "Avg Time [ms]": format_ms(row["avg_ms"]),
                "Min Time [ms]": format_ms(row["min_ms"]),
                "Max Time [ms]": format_ms(row["max_ms"]),
                "Std Dev": format_ms(row["std_ms"]),
                "Effective BW [GB/s]": format_float(row["bw_GB_s"], 2),
                "GFLOP/s": format_float(row["gflops"], 2),
                "Correct": row["correct"],
                "Speedup": format_speedup(speedup),
            }
        )

    return rows


def build_problem_size_row(
    df: pd.DataFrame,
    version: str,
    n: int,
) -> dict[str, Any] | None:
    cpu_n = get_cpu_serial_rows_at_n(df, n)

    gpu_n = df.loc[
        (df["n"] == int(n)) & (df["mode"] == "cuda_kernel") & (df["version"] == version)
    ]

    if cpu_n.empty or gpu_n.empty:
        return None

    cpu_row = cpu_n.iloc[0]
    best_gpu_n = get_best_kernel_row(gpu_n)

    single_vector_size_MiB = cpu_row["bytes"] / 3 / (1 << 20)

    cpu_time_n = cpu_row["avg_ms"]
    gpu_time_n = best_gpu_n["avg_ms"]

    h2d_n = get_h2d_rows_at_n(df, n)
    d2h_n = get_d2h_rows_at_n(df, n)

    if not h2d_n.empty and not d2h_n.empty:
        e2e_time = h2d_n.iloc[0]["avg_ms"] + gpu_time_n + d2h_n.iloc[0]["avg_ms"]
        e2e_speedup = cpu_time_n / e2e_time
        e2e_time_formatted = format_ms(e2e_time)
        e2e_speedup_formatted = format_speedup(e2e_speedup)
    else:
        e2e_time_formatted = "N/A"
        e2e_speedup_formatted = "N/A"

    kernel_speedup = cpu_time_n / gpu_time_n

    return {
        "N": f"{n:,}",
        "Single Vector Size [MiB]": format_float(single_vector_size_MiB, 2),
        "CPU Time [ms]": format_ms(cpu_time_n),
        "CPU BW [GB/s]": format_float(cpu_row["bw_GB_s"], 2),
        "Best GPU Block": best_gpu_n["block_size"],
        "GPU Kernel Time [ms]": format_ms(gpu_time_n),
        "GPU Kernel Effective BW [GB/s]": format_float(best_gpu_n["bw_GB_s"], 2),
        "Kernel Speedup": format_speedup(kernel_speedup),
        "GPU E2E Time [ms]": e2e_time_formatted,
        "E2E Speedup": e2e_speedup_formatted,
    }


def build_problem_size_table_rows_for_version(
    df: pd.DataFrame,
    version: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    gpu_version_rows = df.loc[
        (df["version"] == version) & (df["mode"] == "cuda_kernel")
    ].copy()

    for n in sorted(gpu_version_rows["n"].unique()):
        row = build_problem_size_row(
            df=df,
            version=version,
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
    config: BenchmarkTableConfig,
) -> None:
    large_n = get_large_n(df, config)

    # Main results at largest N
    f.write("## Main Benchmark Results at Largest N\n\n")
    f.write(f"N = {large_n:,}\n\n")
    f.write(f"DType = {df.iloc[0]['dtype']}\n\n")

    cpu_rows = get_cpu_rows_at_n(df, large_n)
    single_vector_size_MiB = cpu_rows.iloc[0]["bytes"] / 3 / (1 << 20)
    f.write(f"Single vector size = {format_float(single_vector_size_MiB, 2)} MiB\n\n")

    rows = build_main_table_rows(
        df=df,
        large_n=large_n,
    )

    f.write(make_markdown_table(rows, "main_table"))
    f.write("\n\n")


def write_block_size_sweep_section(
    f,
    df: pd.DataFrame,
    df_gpu: pd.DataFrame,
    config: BenchmarkTableConfig,
) -> None:
    large_n = get_large_n(df, config)

    # Block size sweep at largest N
    f.write("## CUDA Block Size Sweep\n\n")
    f.write(f"N = {large_n:,}\n\n")

    cpu_rows = get_cpu_rows_at_n(df, large_n)
    cpu_ref_time = get_cpu_ref_time(cpu_rows)

    if cpu_ref_time is None:
        return

    for i_th_version, version in enumerate(get_kernel_versions(df_gpu), start=1):
        block_sweep = get_block_sweep_rows(
            df=df,
            large_n=large_n,
            version=version,
        )

        if block_sweep.empty:
            continue

        f.write(f"{i_th_version}. Version = {version}\n\n")
        f.write(f"DType = {block_sweep.iloc[0]['dtype']}\n\n")

        rows = build_block_sweep_table_rows(
            each_version_sweep=block_sweep,
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

    for i_th_version, version in enumerate(get_kernel_versions(df_gpu), start=1):
        f.write(f"{i_th_version}. Version = {version}\n\n")

        rows = build_problem_size_table_rows_for_version(
            df=df,
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
            config=config,
        )

        write_block_size_sweep_section(
            f=f,
            df=df,
            df_gpu=df_gpu,
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
