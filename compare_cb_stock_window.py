"""用 Polars 和 DuckDB 查询成交额第一的可转债正股分钟区间。"""

from datetime import date, datetime
from pathlib import Path
import sys

import duckdb
import polars as pl
from psycopg2.extensions import make_dsn
from polars.testing import assert_frame_equal


# 复用本地技能的私有配置读取器，不在代码中写入连接凭据。
SKILL_SCRIPTS = (
    Path.home() / '.codex/skills/local-finance-data/scripts'
)
if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))
import local_query as local


TRADE_DATE = date(2026, 10, 15)
START = datetime(2026, 10, 15, 10, 35)
END = datetime(2026, 10, 15, 11, 25)

# 全日成交额排名；仅包含当日有报价的沪深上市可转债。
# 排除本地规则中的 132 前缀交换债；并列时按代码选一只。
TOP_SQL = f"""
    SELECT
        q.trade_date,
        TRIM(q.cb_code) AS cb_code,
        b.cb_name,
        TRIM(b.stk_code) AS stk_code,
        b.stk_short_name,
        CAST(q.amount AS DOUBLE PRECISION) AS cb_daily_amount
    FROM cbonds.cb_daily_quote AS q
    JOIN cbonds.cb_basic AS b ON b.cb_code = q.cb_code
    WHERE q.trade_date = DATE '{TRADE_DATE.isoformat()}'
      AND b.exchange IN ('SH', 'SZ')
      AND b.list_date <= q.trade_date
      AND (b.delist_date IS NULL OR b.delist_date > q.trade_date)
      AND q.cb_code NOT LIKE '132%'
      AND q.amount IS NOT NULL
    ORDER BY q.amount DESC, q.cb_code
    LIMIT 1
"""


def query_polars(config, files):
    """先在 PostgreSQL 选目标，再用 Polars 处理对应正股分钟数据。"""
    connection = local.pg(config)
    try:
        with connection.cursor() as cursor:
            cursor.execute(TOP_SQL)
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError('指定日期没有符合条件的可转债日行情')
    target = pl.DataFrame(rows, schema=columns, orient='row')
    code = target['stk_code'][0]
    if not code:
        raise ValueError('成交额第一的转债缺少正股映射')

    stats = (
        pl.scan_parquet(files, hive_partitioning=False)
        .filter(
            pl.col('stk_code') == code,
            pl.col('trade_time') >= START,
            pl.col('trade_time') <= END,
        )
        .select(
            pl.len().cast(pl.Int64).alias('minute_rows'),
            pl.col('trade_time').n_unique().cast(pl.Int64)
            .alias('unique_minutes'),
            pl.col('trade_time').min().alias('first_time'),
            pl.col('trade_time').max().alias('last_time'),
            pl.col('high').max().alias('highest_price'),
            pl.col('low').min().alias('lowest_price'),
            pl.when(pl.col('amount').count() > 0)
            .then(pl.col('amount').cast(pl.Float64).sum())
            .otherwise(None).alias('stock_window_amount'),
        )
        .collect(engine='streaming')
    )
    return target.hstack(stats)


def query_duckdb(config, files):
    """用只读 PostgreSQL 连接与 Parquet 完成联合 SQL 查询。"""
    dsn = make_dsn(
        host=config['DB_HOST'],
        port=config['DB_PORT'],
        dbname=config['DB_DATABASE'],
        user=config['READ_ONLY_USER'],
        password=config['READ_ONLY_PASSWORD'],
        options='-c default_transaction_read_only=on',
    )
    # SQL 字面量转义，连接信息仅留在进程内，不打印。
    dsn_literal = "'" + dsn.replace("'", "''") + "'"
    top_literal = "'" + TOP_SQL.replace("'", "''") + "'"

    with duckdb.connect() as connection:
        connection.execute('LOAD postgres')
        connection.execute(
            f'ATTACH {dsn_literal} AS local_pg '
            '(TYPE postgres, READ_ONLY)'
        )
        result = connection.execute(
            f"""
            WITH target AS MATERIALIZED (
                SELECT * FROM postgres_query('local_pg', {top_literal})
            ),
            stats AS (
                SELECT
                    m.stk_code,
                    COUNT(*) AS minute_rows,
                    COUNT(DISTINCT m.trade_time) AS unique_minutes,
                    MIN(m.trade_time) AS first_time,
                    MAX(m.trade_time) AS last_time,
                    MAX(m.high) AS highest_price,
                    MIN(m.low) AS lowest_price,
                    SUM(CAST(m.amount AS DOUBLE)) AS stock_window_amount
                FROM read_parquet(?, hive_partitioning=false) AS m
                JOIN target AS t ON m.stk_code = t.stk_code
                WHERE m.trade_time >= ? AND m.trade_time <= ?
                GROUP BY m.stk_code
            )
            SELECT
                t.*,
                COALESCE(s.minute_rows, 0) AS minute_rows,
                COALESCE(s.unique_minutes, 0) AS unique_minutes,
                s.first_time,
                s.last_time,
                s.highest_price,
                s.lowest_price,
                s.stock_window_amount
            FROM target AS t
            LEFT JOIN stats AS s ON t.stk_code = s.stk_code
            """,
            [files, START, END],
        ).pl()
    return result


def main():
    """执行两套查询并核对返回结果。"""
    config = local.load_config()
    month_dir = (
        local.parquet_root(config) / 'stocks/stock_mins'
        / f'year={TRADE_DATE.year}' / f'month={TRADE_DATE.month}'
    )
    files = sorted(path.as_posix() for path in month_dir.rglob('*.parquet'))
    if not files:
        raise FileNotFoundError(f'目标月份无分钟文件：{month_dir}')
    polars_result = query_polars(config, files)
    duckdb_result = query_duckdb(config, files)
    assert_frame_equal(
        polars_result,
        duckdb_result,
        check_dtypes=False,
        check_exact=False,
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    print('结果一致；时间标签包含 10:35 与 11:25。')
    print(polars_result.to_dicts())
    return polars_result, duckdb_result


if __name__ == '__main__':
    polars_result, duckdb_result = main()
