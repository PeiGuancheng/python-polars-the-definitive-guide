"""使用本地真实股票分钟行情比较 Polars eager 与 lazy 的耗时。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import datetime
import gc
import json
import pathlib
import platform
import random
import statistics
import sys
import time

import polars as pl
from polars import testing

COLUMNS = [
    "stk_code",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
]


def find_minute_files(
    root: pathlib.Path, start: datetime.date, end: datetime.date
) -> list[pathlib.Path]:
    """查找指定年月的文件，兼容 month=9 和 month=09。

    Args:
        root: 股票分钟数据根目录。
        start: 起始日期，包含当天。
        end: 截止日期，包含当天。

    Returns:
        去重并按路径排序的 Parquet 文件列表。

    Raises:
        FileNotFoundError: 根目录不存在，或指定年月没有匹配文件。
    """
    if not root.is_dir():
        raise FileNotFoundError(f"分钟数据根目录不存在：{root}；请检查 --root")

    months = set()
    day = start
    while day <= end:
        months.add((day.year, day.month))
        day += datetime.timedelta(days=1)

    directories = sorted(
        {
            root / f"year={year}" / f"month={name}"
            for year, month in months
            for name in {str(month), f"{month:02d}"}
        }
    )
    files = sorted(
        {p for directory in directories for p in directory.rglob("*.parquet")}
    )

    if not files:
        searched = "\n".join(str(p) for p in directories)
        raise FileNotFoundError(
            f"指定日期范围 {start} 至 {end} 未发现分钟 Parquet 文件。已检查：\n{searched}"
        )
    return files


def operations(codes: list[str]) -> dict[str, Callable]:
    """创建使用相同表达式的 eager 和 lazy 测试操作。

    Args:
        codes: 当前样本的股票代码，用于过滤条件与连接映射表。

    Returns:
        操作名称到处理函数的映射，函数接收 DataFrame 或 LazyFrame。
    """

    def select_columns(frame):
        return frame.select("stk_code", "trade_time", "close")

    def filter_select(frame):
        filtered = frame.filter(pl.col("stk_code").is_in(codes[:10]))
        return select_columns(filtered)

    def features(frame):
        bar_return = pl.col("close") / pl.col("open") - 1
        amplitude = (pl.col("high") - pl.col("low")) / pl.col("open")
        amount_per_volume = (
            pl.when(pl.col("vol") > 0)
            .then(pl.col("amount") / pl.col("vol"))
            .otherwise(None)
        )
        return frame.with_columns(
            bar_return.alias("bar_return"),
            amplitude.alias("amplitude"),
            amount_per_volume.alias("amount_per_volume"),
        )

    def aggregate(frame):
        aggregated = frame.group_by("stk_code").agg(
            pl.len().alias("bars"),
            pl.col("close").mean().alias("mean_close"),
            pl.col("vol").sum(),
            pl.col("amount").sum(),
        )
        return aggregated.sort("stk_code")

    def daily(frame):
        ordered = frame.sort(["stk_code", "trade_time"])
        aggregated = ordered.group_by(
            "stk_code", pl.col("trade_time").dt.date().alias("date")
        ).agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("vol").sum(),
            pl.col("amount").sum(),
        )
        return aggregated.sort(["stk_code", "date"])

    def rolling(frame):
        ordered = frame.sort(["stk_code", "trade_time"])
        return ordered.with_columns(
            pl.col("close").rolling_mean(20).over("stk_code").alias("ma20"),
            pl.col("close")
            .pct_change()
            .over("stk_code")
            .alias("minute_return"),
        )

    def resample(frame):
        ordered = frame.sort(["stk_code", "trade_time"])
        aggregated = ordered.group_by_dynamic(
            "trade_time",
            every="5m",
            group_by="stk_code",
            closed="left",
            label="left",
        ).agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("vol").sum(),
        )
        return aggregated.sort(["stk_code", "trade_time"])

    # 使用样本内代码构建唯一映射表，仅用于比较连接操作。
    lookup = pl.DataFrame(
        {
            "stk_code": codes,
            "sample_group": [index % 10 for index in range(len(codes))],
        }
    )

    def join(frame):
        right = lookup.lazy() if isinstance(frame, pl.LazyFrame) else lookup
        joined = frame.join(right, on="stk_code", how="left")
        return joined.sort(["stk_code", "trade_time"])

    def pipeline(frame):
        filtered = frame.filter(pl.col("stk_code").is_in(codes[:10]))
        enriched = features(filtered)
        aggregated = enriched.group_by("stk_code").agg(
            pl.col("bar_return").mean(),
            pl.col("amplitude").mean(),
            pl.col("amount").sum(),
        )
        return aggregated.sort("stk_code")

    def sort_rows(frame):
        return frame.sort(["close", "stk_code", "trade_time"])

    def top_k(frame):
        ordered = frame.sort(
            ["amount", "stk_code", "trade_time"],
            descending=[True, False, False],
        )
        return ordered.head(100)

    def unique_dates(frame):
        selected = frame.select("stk_code", pl.col("trade_time").dt.date())
        return selected.unique().sort(["stk_code", "trade_time"])

    return {
        "select": select_columns,
        "filter_select": filter_select,
        "with_columns": features,
        "group_by": aggregate,
        "daily_ohlcv": daily,
        "sort": sort_rows,
        "top_k": top_k,
        "unique": unique_dates,
        "rolling_window": rolling,
        "resample_5m": resample,
        "left_join": join,
        "filter_features_aggregate": pipeline,
    }


def materialize(q: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """lazy 的计时必须包括 collect，确保测到实际执行。"""
    return q.collect() if isinstance(q, pl.LazyFrame) else q


def measure(eager, lazy, repeats: int, warmups: int, seed: int):
    """验证并预热查询，再随机交替计时，结果销毁和回收不计时。

    Args:
        eager: 返回 eager 查询结果的无参数函数。
        lazy: 返回已 collect 的 lazy 查询结果的无参数函数。
        repeats: 正式计时次数。
        warmups: 正式计时前的预热次数。
        seed: 执行顺序随机化的种子。

    Returns:
        各模式的逐次耗时（毫秒）及查询结果行数。
    """
    a, b = eager(), lazy()
    testing.assert_frame_equal(
        a, b, check_exact=False, rel_tol=1e-9, abs_tol=1e-9
    )
    output_rows = a.height
    del a, b

    funcs = {"eager": eager, "lazy": lazy}
    for _ in range(warmups):
        for fn in funcs.values():
            result = fn()
            del result

    times = {name: [] for name in funcs}
    rng = random.Random(seed)
    for _ in range(repeats):
        order = list(funcs)
        rng.shuffle(order)
        for name in order:
            gc.collect()
            start = time.perf_counter()
            result = funcs[name]()
            elapsed = time.perf_counter() - start
            times[name].append(elapsed * 1000)
            del result

    return times, output_rows


def parse_arguments() -> argparse.Namespace:
    """解析并验证命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("D:/SFData/parquet/stocks/stock_mins"),
    )
    parser.add_argument(
        "--start",
        type=datetime.date.fromisoformat,
        default=datetime.date(2026, 10, 8),
    )
    parser.add_argument(
        "--end",
        type=datetime.date.fromisoformat,
        default=datetime.date(2026, 10, 15),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10_000, 100_000, 1_000_000, 5_000_000],
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("benchmark_results")
    )
    parser.add_argument("--cases", nargs="+", help="只运行指定操作名称")
    args = parser.parse_args()
    if (
        args.start > args.end
        or min(args.sizes) <= 0
        or args.repeats < 1
        or args.warmups < 0
    ):
        parser.error("日期、数据量或重复次数无效")

    if args.cases:
        unknown = set(args.cases) - operations([]).keys()
        if unknown:
            parser.error(f"未知操作：{unknown}")
    return args


def save_results(output_dir, records, raw, metadata) -> None:
    """保存汇总、逐次计时和运行元数据。"""
    pl.DataFrame(records).write_csv(output_dir / "summary.csv")
    pl.DataFrame(raw).write_csv(output_dir / "timings.csv")
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_runners(operation, frame, fixture, scope):
    """构建执行入口，确保读取与 collect 位于计时范围内。"""

    def eager():
        source = frame if scope == "memory" else pl.read_parquet(fixture)
        return materialize(operation(source))

    def lazy():
        source = frame.lazy() if scope == "memory" else pl.scan_parquet(fixture)
        return materialize(operation(source))

    return eager, lazy


def main() -> None:
    """准备各量级样本，运行测试并持续保存结果。"""
    args = parse_arguments()

    # 仅查找指定年月的文件，时间过滤进一步限制到指定日期。
    files = find_minute_files(args.root, args.start, args.end)
    args.out.mkdir(parents=True, exist_ok=True)

    source = (
        pl.scan_parquet(files, hive_partitioning=False)
        .select(COLUMNS)
        .filter(
            pl.col("trade_time") >= args.start,
            pl.col("trade_time") < args.end + datetime.timedelta(days=1),
        )
    )

    metadata = {
        "python": sys.version,
        "executable": sys.executable,
        "polars": pl.__version__,
        "platform": platform.platform(),
        "threads": pl.thread_pool_size(),
        "source_files": [str(p) for p in files],
        "start": str(args.start),
        "end": str(args.end),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "sampling": "文件路径排序后的前 N 行，未随机抽样，不复制扩充",
        "cache": "已验证并预热的查询；操作系统文件缓存未清除，不代表冷盘读取",
        "samples": [],
    }
    records, raw = [], []

    for requested in sorted(set(args.sizes)):
        # 每次仅保留当前量级，准备数据和写入测试文件均不计时。
        df = source.head(requested).collect()
        if df.height < requested:
            print(
                f"SKIP {requested:,}: only {df.height:,} rows available",
                flush=True,
            )
            del df
            continue

        codes = df["stk_code"].unique().sort().to_list()
        metadata["samples"].append(
            {
                "rows": df.height,
                "stocks": len(codes),
                "min_time": str(df["trade_time"].min()),
                "max_time": str(df["trade_time"].max()),
            }
        )
        cases = operations(codes)
        if args.cases:
            cases = {k: v for k, v in cases.items() if k in args.cases}

        fixture = args.out / f"sample_{requested}.parquet"
        if fixture.exists():
            raise FileExistsError(
                f"测试样本已存在，请换一个 --out 目录：{fixture}"
            )
        df.write_parquet(fixture, compression="zstd", row_group_size=100_000)

        try:
            for scope in ["memory", "parquet"]:
                for name, operation in cases.items():
                    eager, lazy = build_runners(operation, df, fixture, scope)
                    times, output_rows = measure(
                        eager, lazy, args.repeats, args.warmups, 42
                    )
                    eager_ms = statistics.median(times["eager"])
                    lazy_ms = statistics.median(times["lazy"])
                    records.append(
                        {
                            "rows": requested,
                            "scope": scope,
                            "operation": name,
                            "output_rows": output_rows,
                            "eager_median_ms": eager_ms,
                            "lazy_median_ms": lazy_ms,
                            "eager_min_ms": min(times["eager"]),
                            "lazy_min_ms": min(times["lazy"]),
                            "speedup_eager_over_lazy": eager_ms / lazy_ms,
                        }
                    )
                    for mode, values in times.items():
                        raw.extend(
                            {
                                "rows": requested,
                                "scope": scope,
                                "operation": name,
                                "mode": mode,
                                "repeat": i + 1,
                                "ms": t,
                            }
                            for i, t in enumerate(values)
                        )
                    print(
                        f"{requested:>10,} {scope:7} {name:26} "
                        f"eager={eager_ms:9.3f}ms lazy={lazy_ms:9.3f}ms "
                        f"speedup={eager_ms / lazy_ms:.2f}x",
                        flush=True,
                    )
                    # 每组及时保存，长时间运行被中断时仍可保留已完成结果。
                    save_results(args.out, records, raw, metadata)
        finally:
            # 只删除本次运行创建的样本文件，不修改原始数据湖。
            fixture.unlink(missing_ok=True)
        del df
        gc.collect()


if __name__ == "__main__":
    main()
