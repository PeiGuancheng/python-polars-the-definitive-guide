"""使用固定的真实分钟行情文件，对比 Polars 的 4 和 12 线程。"""

import datetime
import json
import pathlib
import statistics

import benchmark_stock_minutes as benchmark


def main():
    """选择小样本或复用文件清单，交替测试线程数并核验结果。"""
    args = benchmark.parse_arguments()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = pathlib.Path(__file__).resolve().parent / "benchmark"
    folder = folder / f"polars_threads_{timestamp}"
    folder.mkdir(parents=True)

    # 优先选择完整的 2025 年数据，按实际年月排序，避免月份字典序混乱。
    files = list((args.root / "year=2025").rglob("*.parquet"))
    files.sort(
        key=lambda path: (int(path.parent.name.split("=")[1]), path.name)
    )
    selected = []
    size_bytes = 0
    for path in files:
        selected.append(str(path.resolve()))
        size_bytes += path.stat().st_size
        if size_bytes >= 8 * 2**30:
            break
    if args.manifest:
        source_info = json.loads(
            pathlib.Path(args.manifest).read_text(encoding="utf-8")
        )
        selected = source_info["files"]
        size_bytes = sum(pathlib.Path(path).stat().st_size for path in selected)
    elif not 5 * 2**30 <= size_bytes <= 10 * 2**30:
        raise ValueError("2025 年文件无法组成 5–10 GiB 的测试样本")

    # 复用原基准的字段发现和版本记录，保留选定的全部文件。
    args.target_gib = min(5, size_bytes / 2**30 * 0.99)
    args.repeats = 2
    manifest = benchmark.prepare_manifest(
        args, folder, input_files=[pathlib.Path(path) for path in selected]
    )
    info = json.loads(manifest.read_text(encoding="utf-8"))
    info.update({
        "files": selected,
        "file_count": len(selected),
        "source_file_gib": size_bytes / 2**30,
        "thread_order": [4, 12, 12, 4],
        "cache_note": "不清除系统缓存；按 4/12/12/4 交替顺序运行",
    })
    benchmark.write_json(manifest, info)
    print(
        f"固定样本：{len(selected)} 文件，{size_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    print(f"文件范围：{selected[0]} 至 {selected[-1]}", flush=True)
    print(f"结果目录：{folder}", flush=True)

    if size_bytes <= 10 * 2**30:
        print("先用 4 线程预热文件缓存；预热不计入正式对照", flush=True)
        args.threads = 4
        warmup = benchmark.measure(
            args, manifest, "polars", folder / "warmup_4threads.parquet"
        )
        benchmark.write_json(folder / "warmup.json", warmup)
        if warmup["status"] != "ok":
            raise RuntimeError("预热失败，请检查日志")

    records = []
    reference = None
    for index, threads in enumerate(info["thread_order"], start=1):
        args.threads = threads
        output = folder / f"{index}_polars_{threads}threads.parquet"
        print(f"开始第 {index}/4 次：{threads} 线程", flush=True)
        record = benchmark.measure(args, manifest, "polars", output)
        record["threads"] = threads
        records.append(record)
        benchmark.write_json(folder / "summary.json", records)
        if record["status"] != "ok":
            raise RuntimeError(f"测试失败，请检查 {output.with_suffix('.log')}")
        if record["actual_threads"] != threads:
            raise AssertionError("Polars 实际线程数与请求不一致")
        if reference is None:
            reference = output
        else:
            benchmark.validate_results({"polars": reference, "duckdb": output})
        print(f"第 {index} 次结果核验通过", flush=True)

    comparison = {}
    for threads in (4, 12):
        runs = [record for record in records if record["threads"] == threads]
        comparison[str(threads)] = {
            "median_seconds": statistics.median(
                record["elapsed_seconds"] for record in runs
            ),
            "max_sampled_rss_gib": max(
                record["sampled_peak_rss_gib"] for record in runs
            ),
        }
    comparison["speedup_4_over_12"] = (
        comparison["4"]["median_seconds"]
        / comparison["12"]["median_seconds"]
    )
    benchmark.write_json(folder / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
