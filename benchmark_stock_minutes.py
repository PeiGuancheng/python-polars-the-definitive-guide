"""对比 Polars 和 DuckDB 扫描超内存规模分钟行情的能力。"""

import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, value):
    """以 UTF-8 保存测试记录。

    Args:
        path: 输出文件的 Path 对象。
        value: 可序列化为 JSON 的测试记录。
    """
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def quote_identifier(name):
    """转义并引用 SQL 标识符。

    Args:
        name: 列名或其他 SQL 标识符。

    Returns:
        使用双引号引用的标识符。
    """
    return '"' + name.replace('"', '""') + '"'


def run_worker(args):
    """在独立进程中执行查询，并保存结果与计时信息。

    Args:
        args: 包含引擎、输入清单、输出路径和资源配置的命令行参数。
    """
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    files = manifest["files"]
    extra_columns = manifest["extra_numeric_columns"]
    output = Path(args.output)
    engine = args.worker
    started = time.perf_counter()
    # 延迟导入计算引擎，确保仅初始化当前进程需要的线程池。
    if engine == "polars":
        import polars as pl

        query = (
            pl.scan_parquet(files, hive_partitioning=False)
            .select(["stk_code", "trade_time", "close"] + extra_columns)
            .group_by("stk_code")
            .agg(
                pl.len().cast(pl.Int64).alias("rows"),
                pl.col("trade_time").min().alias("first_time"),
                pl.col("trade_time").max().alias("last_time"),
                pl.col("close").cast(pl.Float64).min().alias("min_close"),
                pl.col("close").cast(pl.Float64).max().alias("max_close"),
                pl.col("close").cast(pl.Float64).mean().alias("mean_close"),
                *[
                    pl.col(name).cast(pl.Float64).sum().alias(f"sum_{name}")
                    for name in extra_columns
                ],
            )
            .sort("stk_code")
        )
        result = query.collect(engine="streaming")
        query_seconds = time.perf_counter() - started
        result.write_parquet(output)
        version = pl.__version__
        actual_threads = pl.thread_pool_size()
    else:
        import duckdb

        connection = duckdb.connect(
            config={
                "threads": args.threads,
                "memory_limit": args.memory_limit,
                "temp_directory": str(
                    output.parent / (output.stem + "_spill")
                ),
                "max_temp_directory_size": args.spill_limit,
            }
        )
        connection.read_parquet(files, hive_partitioning=False).create_view(
            "minutes"
        )
        extra_sql = "".join(
            f", SUM(CAST({quote_identifier(name)} AS DOUBLE)) "
            f"AS {quote_identifier(f'sum_{name}')}"
            for name in extra_columns
        )
        query_sql = """
            SELECT stk_code, COUNT(*) AS rows,
                MIN(trade_time) AS first_time,
                MAX(trade_time) AS last_time,
                MIN(CAST(close AS DOUBLE)) AS min_close,
                MAX(CAST(close AS DOUBLE)) AS max_close,
                AVG(CAST(close AS DOUBLE)) AS mean_close
        """ + extra_sql + """
            FROM minutes
            GROUP BY stk_code
            ORDER BY stk_code
        """
        result = connection.sql(query_sql).to_arrow_table()
        query_seconds = time.perf_counter() - started

        pq.write_table(result, output, compression="zstd")
        connection.close()
        version = duckdb.__version__
        actual_threads = args.threads
    write_json(
        output.with_suffix(".metrics.json"),
        {
            "engine": engine,
            "version": version,
            "actual_threads": actual_threads,
            "query_seconds_including_import": query_seconds,
            "worker_seconds_including_output": time.perf_counter() - started,
        },
    )


def measure(args, manifest, engine, output):
    """采样工作进程 RSS，并应用内存和超时保护。

    Args:
        args: 包含资源限制与线程数的命令行参数。
        manifest: 输入文件清单的 Path 对象。
        engine: 待测试的引擎名称。
        output: 聚合结果文件的 Path 对象。

    Returns:
        包含退出状态、耗时、采样内存峰值和输出路径的字典。
    """

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker", engine,
        "--manifest", str(manifest),
        "--output", str(output),
        "--threads", str(args.threads),
        "--memory-limit", args.memory_limit,
        "--spill-limit", args.spill_limit,
    ]
    environment = os.environ.copy()
    environment["POLARS_MAX_THREADS"] = str(args.threads)
    environment["POLARS_TEMP_DIR"] = str(output.parent / "polars_temp")
    Path(environment["POLARS_TEMP_DIR"]).mkdir(exist_ok=True)
    started = time.perf_counter()
    peak = 0
    status = "ok"
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=log,
        )
        watched = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, watched.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if peak > args.rss_limit_gib * 2**30:
                    status = "rss_limit"
                    process.kill()
                    break
                if time.perf_counter() - started > args.timeout:
                    status = "timeout"
                    process.kill()
                    break
                time.sleep(0.1)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if status == "ok" and process.returncode:
        status = "error"
    record = {
        "engine": engine,
        "status": status,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "sampled_peak_rss_gib": round(peak / 2**30, 3),
        "output": str(output),
    }
    metrics = output.with_suffix(".metrics.json")
    if status == "ok":
        record.update(json.loads(metrics.read_text(encoding="utf-8")))
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def parse_arguments():
    """解析并检查命令行参数。

    Returns:
        通过基本合法性检查的 argparse.Namespace 对象。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("D:/SFData/parquet/stocks/stock_mins"),
    )
    parser.add_argument("--target-gib", type=float, default=51)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GiB")
    parser.add_argument("--spill-limit", default="20GiB")
    available_gib = psutil.virtual_memory().available / 2**30
    parser.add_argument(
        "--rss-limit-gib",
        type=float,
        default=min(8, available_gib * 0.65),
    )
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", choices=["polars", "duckdb"])
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.target_gib <= 0 or args.threads < 1 or args.repeats < 1:
        parser.error("数据大小、线程数和重复次数必须为正数")
    if args.rss_limit_gib <= 0 or args.timeout <= 0:
        parser.error("内存保护阈值和超时时间必须为正数")
    if args.worker and (not args.manifest or not args.output):
        parser.error("工作进程必须指定 --manifest 和 --output")
    return args


def create_smoke_data(folder):
    """创建两份用于检查执行流程的小样本文件。

    Args:
        folder: 测试结果目录。

    Returns:
        保存小样本文件的目录。
    """
    sample_root = folder / "sample"
    sample_root.mkdir()
    for index in range(2):
        table = pa.table({
            "stk_code": ["000001.SZ", "600000.SH", "000001.SZ"],
            "trade_time": [
                datetime.datetime(2025, 1, 2, 9, 31 + index)
            ] * 3,
            "close": [10.0 + index, None, 12.0],
        })
        pq.write_table(table, sample_root / f"{index}.parquet")
    return sample_root


def prepare_manifest(args, folder, input_files=None):
    """选择输入文件并记录数据范围与运行配置。

    Args:
        args: 包含数据目录和目标大小的命令行参数。
        folder: 保存文件清单的测试结果目录。
        input_files: 可选的固定文件列表；指定后保留全部文件及其顺序。

    Returns:
        文件清单的 Path 对象。

    Raises:
        ValueError: 数据量不足或输入文件缺少必要字段。
    """
    files = (
        sorted(args.root.rglob("*.parquet"))
        if input_files is None
        else input_files
    )
    selected = []
    total_bytes = 0
    for path in files:
        selected.append(str(path.resolve()))
        total_bytes += path.stat().st_size
        if (
            input_files is None
            and total_bytes > args.target_gib * 2**30
            and not args.smoke
        ):
            break
    if not selected or total_bytes <= args.target_gib * 2**30:
        raise ValueError("目录中的 Parquet 文件不足目标大小")
    metadata = pq.ParquetFile(selected[0])
    schema = metadata.schema_arrow
    for name in ("stk_code", "trade_time", "close"):
        if name not in schema.names:
            raise ValueError(f"缺少字段 {name}，实际字段为 {schema.names}")
    manifest = folder / "manifest.json"
    extra_columns = [
        field.name
        for field in schema
        if field.name != "close"
        and (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_decimal(field.type)
        )
    ]
    info = {
        "files": selected,
        "file_count": len(selected),
        "source_file_gib": total_bytes / 2**30,
        "total_ram_gib": psutil.virtual_memory().total / 2**30,
        "available_ram_gib": psutil.virtual_memory().available / 2**30,
        "schema_first_file": str(schema),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("polars", "duckdb", "pyarrow", "psutil")
        },
        "columns_read": ["stk_code", "trade_time", "close"] + extra_columns,
        "extra_numeric_columns": extra_columns,
        "settings": vars(args) | {"root": str(args.root)},
    }
    write_json(manifest, info)
    print(
        json.dumps(
            info | {"files": "见 manifest.json"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return manifest


def validate_results(results):
    """核验两个引擎的聚合结果，允许浮点累加的微小误差。

    Args:
        results: 引擎名称到结果文件路径的映射。

    Raises:
        AssertionError: 两个结果的结构或数值不一致。
    """
    # 父进程仅在所有工作进程结束后导入 Polars，以验证小型聚合结果。
    import polars as pl
    import polars.testing

    polars.testing.assert_frame_equal(
        pl.read_parquet(results["polars"]),
        pl.read_parquet(results["duckdb"]),
        check_dtypes=False,
        rel_tol=1e-8,
        abs_tol=1e-8,
    )


def run_benchmarks(args, manifest, folder):
    """交替引擎顺序运行各轮测试，并保存指标及核验结果。

    Args:
        args: 包含重复次数和资源配置的命令行参数。
        manifest: 输入文件清单的路径。
        folder: 测试结果目录。
    """
    records = []
    for repeat in range(args.repeats):
        engines = ["polars", "duckdb"]
        if repeat % 2:
            engines.reverse()
        results = {}
        for engine in engines:
            output = folder / f"{repeat + 1}_{engine}.parquet"
            record = measure(args, manifest, engine, output)
            records.append(record)
            write_json(folder / "summary.json", records)
            if record["status"] == "ok":
                results[engine] = output
        if len(results) == 2:
            validate_results(results)
            print(f"第 {repeat + 1} 轮结果一致", flush=True)
    print(f"结果目录：{folder}")


def main():
    """准备数据清单，并按命令行选项执行检查或基准测试。"""
    args = parse_arguments()
    if args.worker:
        run_worker(args)
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = Path(__file__).resolve().parent / "benchmark" / timestamp
    folder.mkdir(parents=True)
    if args.smoke:
        args.root = create_smoke_data(folder)
        args.target_gib = 1e-12
        args.run = True
        args.repeats = 1

    manifest = prepare_manifest(args, folder)
    if args.run:
        run_benchmarks(args, manifest, folder)
    else:
        print(f"仅检查完成；加 --run 执行大数据测试。清单：{manifest}")


if __name__ == "__main__":
    main()
