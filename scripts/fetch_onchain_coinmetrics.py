"""
Daily on-chain metrics for BTC and ETH from Coin Metrics' free community API
(community-api.coinmetrics.io, no key). Stored in the LOCAL research database
only (table onchain_daily), for studies, never for the live bot.

Which metrics exist in the community tier changes, so each metric is requested
on its own and a refusal is reported rather than failing the run.

Timing is NOT assumed here: whether a daily row stamped D describes the day
that starts at D or the one that ends there is established empirically by the
study (same-day vs next-day correlation with the price change), exactly as for
DVOL and DXY.

Usage:
    python scripts/fetch_onchain_coinmetrics.py
"""
import requests
from psycopg2.extras import execute_values

from db import get_connection

API = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
ASSETS = {"btc": ["CapMVRVCur", "AdrActCnt", "TxCnt", "NVTAdj", "HashRate", "PriceUSD", "CapRealUSD"],
          "eth": ["CapMVRVCur", "AdrActCnt", "TxCnt", "NVTAdj", "PriceUSD", "CapRealUSD"]}


def fetch_metric(asset: str, metric: str, start="2018-01-01"):
    rows, url, params = [], API, dict(assets=asset, metrics=metric, frequency="1d", start_time=start, page_size=10000)
    while url:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            return None, f"{r.status_code} {r.text[:120]}"
        body = r.json()
        for d in body.get("data", []):
            v = d.get(metric)
            if v is not None:
                rows.append((asset, d["time"], metric, float(v)))
        url, params = body.get("next_page_url"), None
    return rows, None


def main():
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS onchain_daily (
                asset text NOT NULL, ts timestamptz NOT NULL, metric text NOT NULL,
                value double precision, PRIMARY KEY (asset, ts, metric))""")
    conn.commit()
    for asset, metrics in ASSETS.items():
        for metric in metrics:
            rows, err = fetch_metric(asset, metric)
            if rows is None:
                print(f"{asset} {metric}: НЕДОСТУПНО ({err})")
                continue
            with conn.cursor() as cur:
                execute_values(cur, """INSERT INTO onchain_daily (asset, ts, metric, value) VALUES %s
                                       ON CONFLICT (asset, ts, metric) DO UPDATE SET value = EXCLUDED.value""", rows)
            conn.commit()
            print(f"{asset} {metric}: {len(rows)} дней, {rows[0][1][:10] if rows else '-'}..{rows[-1][1][:10] if rows else '-'}")
    conn.close()


if __name__ == "__main__":
    main()
