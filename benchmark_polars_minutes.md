# 股票分钟行情：Polars eager / lazy 性能对比

脚本采用 Google Python 风格组织：4 空格缩进、80 字符目标行宽、
Google 格式的中文文档字符串，并将复杂查询拆为有名称的步骤。
参数解析、执行入口和结果保存分别由独立函数负责。
可使用 `python -m black --line-length 80 benchmark_polars_minutes.py` 统一排版。

运行当前目录下的脚本：

```powershell
python benchmark_polars_minutes.py
```

默认使用 `D:/SFData/parquet/stocks/stock_mins`，限定 2026-10-08 至 2026-10-15，比较 1 万、10 万、100 万、500 万行，每组预热 1 次，正式重复 5 次。需要安装 Polars；本机验证使用系统 `python`（Polars 1.41.2），项目 `.venv` 当前缺少 Polars。

分区查找兼容 `month=9` 和 `month=09`。如终端解释器不同，可将下面命令中的 `python` 替换为 `& "D:\Programs\anaconda3\envs\py314\python.exe"`。

自定义参数示例：

```powershell
python benchmark_polars_minutes.py --start 2026-09-01 --end 2026-09-30 --sizes 10000 100000 1000000 10000000 --repeats 7 --out benchmark_september
python benchmark_polars_minutes.py --sizes 1000000 --cases group_by rolling_window resample_5m --out benchmark_subset
```

## 测试内容

12 类操作：列选择、过滤后选列、衍生列计算、分组聚合、日 OHLCV、排序、Top K、去重、分组滚动与收益率、5 分钟重采样、左连接、过滤→计算→聚合组合查询。

每种操作分别测试两种范围：

| scope | eager | lazy |
|---|---|---|
| memory | 对已有 DataFrame 执行操作 | 对同一 DataFrame 调用 lazy 后执行并 collect |
| parquet | read_parquet 后执行操作 | scan_parquet 后执行并 collect |

数据准备、测试样本写盘和结果校验均不计时。每轮随机决定两种模式的先后顺序，输出中位数、最小值与每次原始耗时。每组先比较实际结果，数值允许 1e-9 浮点误差；不一致直接报错。

## 输出与解释

- `summary.csv`：每组中位数、最小值、输出行数、加速倍数。
- `timings.csv`：每次计时，便于查看波动。
- `metadata.json`：解释器、Polars 版本、线程数、源文件、时间范围及实际样本覆盖。

`speedup_eager_over_lazy = eager 中位耗时 / lazy 中位耗时`，大于 1 表示 lazy 更快。简单选列可能共享底层内存，极短耗时容易受噪声影响。单操作的 eager 也可能使用 Polars 内部查询执行器，因此两者计算耗时接近是正常结果。

Parquet 测试使用同一份 Zstd 压缩样本，固定为 10 万行的 row group，lazy 可以进行列裁剪、谓词下推等优化。这体现完整查询的优化能力。文件刚写入且查询已经预热，因此属于系统缓存下的读取，不代表冷盘 I/O，也不能直接外推到原始数据湖所有文件布局。

样本是指定日期文件按路径排序后的前 N 行，未随机抽样、未复制扩充；不同量级的股票与时间覆盖可能不同，详见元数据。数据不足的量级会跳过。分钟数据的实际最大时间以本次样本为准。原始字段使用未复权 OHLC；成交额与成交量沿用本地原始单位，`amount_per_volume` 仅为运算测试，不解释为成交均价。

滚动窗口按股票跨交易日连续计算；如研究策略需要每日重置，应增加日期分组。5 分钟重采样按自然时钟左闭右开窗口，不额外处理集合竞价和交易时段。左连接的分组标签是测试构造的映射，不是真实行业。

脚本只读取原始数据；临时样本写入 `--out`，完成当前量级后删除。中断后的样本可能保留，重新运行可使用新的输出目录。大数据量下排序与连接会增加内存需求，可先跑小量级，再逐步增加；该脚本未测量峰值内存。
