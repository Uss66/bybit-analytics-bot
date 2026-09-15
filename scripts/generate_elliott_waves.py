"""
Generate Elliott Wave candidates from detected pivots.

For each (symbol, threshold_pct) pair:
1. Generate 5-wave impulse candidates from every 6 consecutive pivots
2. Generate 3-wave corrective candidates from every 4 consecutive pivots
3. Check structural rules for each wave type (rule_valid TRUE/FALSE)
4. Compute volume metrics from hourly OHLCV data
5. Insert results into elliott_waves table

Volume metrics computed from daily aggregates of hourly volume:
- avg_volume: mean of daily volumes in wave's date range
- end_volume: daily volume sum for day containing end_ts
- volume_trend: 'rising'/'falling'/'flat' based on linear slope
- volume_vs_prior_wave: this wave's avg_volume / prior wave's avg_volume

Usage:
    cd scripts && ../../.venv/Scripts/python.exe generate_elliott_waves.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from psycopg2.extras import execute_values

from db import get_connection


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
THRESHOLDS = [0.05, 0.08, 0.12]


def load_daily_volumes(conn, symbol):
    """
    Load all daily volume sums for a symbol, indexed by date.

    Returns a pandas Series indexed by date (datetime at midnight UTC)
    with daily volume sums.
    """
    query = """
        SELECT date_trunc('day', ts) AS day, SUM(volume) AS daily_volume
        FROM ohlcv
        WHERE symbol = %s
        GROUP BY day
        ORDER BY day
    """
    df = pd.read_sql(query, conn, params=(symbol,))
    if df.empty:
        return pd.Series(dtype=float)

    df['day'] = pd.to_datetime(df['day'], utc=True)
    return df.set_index('day')['daily_volume']


def get_wave_volume_metrics(wave_start_ts, wave_end_ts, daily_volumes_series):
    """
    Compute volume metrics for a wave given its [start_ts, end_ts] range.

    Args:
        wave_start_ts: wave start timestamp (datetime with tz)
        wave_end_ts: wave end timestamp (datetime with tz)
        daily_volumes_series: pandas Series indexed by date with daily volume sums

    Returns:
        dict with keys: avg_volume, end_volume, volume_trend
    """
    # Convert to dates at midnight UTC for indexing
    start_date = wave_start_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = wave_end_ts.replace(hour=0, minute=0, second=0, microsecond=0)

    # Slice daily volumes within the wave range (inclusive)
    mask = (daily_volumes_series.index >= start_date) & (daily_volumes_series.index <= end_date)
    wave_volumes = daily_volumes_series[mask]

    if wave_volumes.empty:
        return {'avg_volume': None, 'end_volume': None, 'volume_trend': None}

    avg_volume = wave_volumes.mean()
    end_volume = wave_volumes.iloc[-1] if len(wave_volumes) > 0 else None

    # Compute volume trend via linear fit
    if len(wave_volumes) >= 2:
        day_indices = np.arange(len(wave_volumes))
        slope, _ = np.polyfit(day_indices, wave_volumes.values, 1)

        # Trend is 'rising' if slope > 1% of mean per day, 'falling' if < -1%, else 'flat'
        threshold = 0.01 * avg_volume
        if slope > threshold:
            volume_trend = 'rising'
        elif slope < -threshold:
            volume_trend = 'falling'
        else:
            volume_trend = 'flat'
    else:
        volume_trend = 'flat'

    return {
        'avg_volume': avg_volume,
        'end_volume': end_volume,
        'volume_trend': volume_trend
    }


def check_impulse_rules(p0, p1, p2, p3, p4, p5):
    """
    Check 5-wave impulse structural rules.

    All three rules must pass for rule_valid = TRUE.

    Args:
        p0, p1, p2, p3, p4, p5: tuples (ts, price, pivot_type) from consecutive pivots

    Returns:
        bool: TRUE if all three rules pass
    """
    # Determine direction from p0
    direction = 'up' if p0[2] == 'low' else 'down'

    # Rule 1: Wave 2 doesn't retrace more than 100% of wave 1
    if direction == 'up':
        rule1 = p2[1] > p0[1]  # p2.price > p0.price
    else:
        rule1 = p2[1] < p0[1]  # p2.price < p0.price

    # Rule 2: Wave 3 is not the shortest among waves 1, 3, 5
    wave1_size = abs(p1[1] - p0[1])
    wave3_size = abs(p3[1] - p2[1])
    wave5_size = abs(p5[1] - p4[1])

    rule2 = not (wave3_size < wave1_size and wave3_size < wave5_size)

    # Rule 3: Wave 4 doesn't overlap wave 1's price territory
    if direction == 'up':
        rule3 = p4[1] > p1[1]  # p4.price > p1.price
    else:
        rule3 = p4[1] < p1[1]  # p4.price < p1.price

    return rule1 and rule2 and rule3


def check_correction_rules(p0, p1, p2):
    """
    Check 3-wave correction structural rule.

    Wave B must not fully retrace wave A (with 2% tolerance).

    Args:
        p0, p1, p2: tuples (ts, price, pivot_type) from consecutive pivots

    Returns:
        bool: TRUE if rule passes
    """
    # Determine direction from p0
    direction = 'down' if p0[2] == 'high' else 'up'

    # Rule: Wave B doesn't retrace more than 100% + 2% tolerance of wave A
    if direction == 'down':
        # For down correction, p2.price < p0.price * 1.02
        return p2[1] < p0[1] * 1.02
    else:
        # For up correction, p2.price > p0.price * 0.98
        return p2[1] > p0[1] * 0.98


def generate_impulses(pivots, daily_volumes_series, seq_id_start=0):
    """
    Generate 5-wave impulse candidates from pivot sequence.

    Every 6 consecutive pivots (p0...p5) forms one impulse candidate.

    Args:
        pivots: list of (ts, price, pivot_type) tuples, chronologically ordered
        daily_volumes_series: pandas Series of daily volumes indexed by date
        seq_id_start: starting sequence_id (to avoid conflicts across symbol/
            threshold groups and with correction sequences - previously this
            always restarted at 0 per call, causing unrelated impulse
            candidates from different symbols/thresholds to collide on the
            same sequence_id)

    Returns:
        list of wave row dicts to be inserted into elliott_waves, and the
        final sequence_id counter value (for chaining into the next call)
    """
    waves = []
    sequence_id_counter = seq_id_start

    for i in range(len(pivots) - 5):
        p0, p1, p2, p3, p4, p5 = pivots[i:i+6]

        # Determine direction
        direction = 'up' if p0[2] == 'low' else 'down'

        # Check rules
        rule_valid = check_impulse_rules(p0, p1, p2, p3, p4, p5)

        # Generate sequence_id (increment for each 5-wave group)
        sequence_id_counter += 1
        seq_id = sequence_id_counter

        # Wave 1: p0 -> p1
        vol_metrics_w1 = get_wave_volume_metrics(p0[0], p1[0], daily_volumes_series)
        waves.append({
            'wave_label': '1',
            'sequence_id': seq_id,
            'start_ts': p0[0],
            'end_ts': p1[0],
            'start_price': p0[1],
            'end_price': p1[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_w1['avg_volume'],
            'end_volume': vol_metrics_w1['end_volume'],
            'volume_vs_prior_wave': None,  # No prior wave
            'volume_trend': vol_metrics_w1['volume_trend'],
        })

        # Wave 2: p1 -> p2
        vol_metrics_w2 = get_wave_volume_metrics(p1[0], p2[0], daily_volumes_series)
        waves.append({
            'wave_label': '2',
            'sequence_id': seq_id,
            'start_ts': p1[0],
            'end_ts': p2[0],
            'start_price': p1[1],
            'end_price': p2[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_w2['avg_volume'],
            'end_volume': vol_metrics_w2['end_volume'],
            'volume_vs_prior_wave': (vol_metrics_w2['avg_volume'] / vol_metrics_w1['avg_volume']
                                     if vol_metrics_w1['avg_volume'] and vol_metrics_w1['avg_volume'] > 0 else None),
            'volume_trend': vol_metrics_w2['volume_trend'],
        })

        # Wave 3: p2 -> p3
        vol_metrics_w3 = get_wave_volume_metrics(p2[0], p3[0], daily_volumes_series)
        waves.append({
            'wave_label': '3',
            'sequence_id': seq_id,
            'start_ts': p2[0],
            'end_ts': p3[0],
            'start_price': p2[1],
            'end_price': p3[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_w3['avg_volume'],
            'end_volume': vol_metrics_w3['end_volume'],
            'volume_vs_prior_wave': (vol_metrics_w3['avg_volume'] / vol_metrics_w2['avg_volume']
                                     if vol_metrics_w2['avg_volume'] and vol_metrics_w2['avg_volume'] > 0 else None),
            'volume_trend': vol_metrics_w3['volume_trend'],
        })

        # Wave 4: p3 -> p4
        vol_metrics_w4 = get_wave_volume_metrics(p3[0], p4[0], daily_volumes_series)
        waves.append({
            'wave_label': '4',
            'sequence_id': seq_id,
            'start_ts': p3[0],
            'end_ts': p4[0],
            'start_price': p3[1],
            'end_price': p4[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_w4['avg_volume'],
            'end_volume': vol_metrics_w4['end_volume'],
            'volume_vs_prior_wave': (vol_metrics_w4['avg_volume'] / vol_metrics_w3['avg_volume']
                                     if vol_metrics_w3['avg_volume'] and vol_metrics_w3['avg_volume'] > 0 else None),
            'volume_trend': vol_metrics_w4['volume_trend'],
        })

        # Wave 5: p4 -> p5
        vol_metrics_w5 = get_wave_volume_metrics(p4[0], p5[0], daily_volumes_series)
        waves.append({
            'wave_label': '5',
            'sequence_id': seq_id,
            'start_ts': p4[0],
            'end_ts': p5[0],
            'start_price': p4[1],
            'end_price': p5[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_w5['avg_volume'],
            'end_volume': vol_metrics_w5['end_volume'],
            'volume_vs_prior_wave': (vol_metrics_w5['avg_volume'] / vol_metrics_w4['avg_volume']
                                     if vol_metrics_w4['avg_volume'] and vol_metrics_w4['avg_volume'] > 0 else None),
            'volume_trend': vol_metrics_w5['volume_trend'],
        })

    return waves, sequence_id_counter


def generate_corrections(pivots, daily_volumes_series, seq_id_start):
    """
    Generate 3-wave correction candidates from pivot sequence.

    Every 4 consecutive pivots (p0...p3) forms one correction candidate.

    Args:
        pivots: list of (ts, price, pivot_type) tuples, chronologically ordered
        daily_volumes_series: pandas Series of daily volumes indexed by date
        seq_id_start: starting sequence_id (to avoid conflicts with impulses)

    Returns:
        list of wave row dicts, sequence_id counter for last generated
    """
    waves = []
    sequence_id_counter = seq_id_start

    for i in range(len(pivots) - 3):
        p0, p1, p2, p3 = pivots[i:i+4]

        # Determine direction
        direction = 'down' if p0[2] == 'high' else 'up'

        # Check rule
        rule_valid = check_correction_rules(p0, p1, p2)

        # Generate sequence_id
        sequence_id_counter += 1
        seq_id = sequence_id_counter

        # Wave A: p0 -> p1
        vol_metrics_a = get_wave_volume_metrics(p0[0], p1[0], daily_volumes_series)
        waves.append({
            'wave_label': 'A',
            'sequence_id': seq_id,
            'start_ts': p0[0],
            'end_ts': p1[0],
            'start_price': p0[1],
            'end_price': p1[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_a['avg_volume'],
            'end_volume': vol_metrics_a['end_volume'],
            'volume_vs_prior_wave': None,  # No prior wave
            'volume_trend': vol_metrics_a['volume_trend'],
        })

        # Wave B: p1 -> p2
        vol_metrics_b = get_wave_volume_metrics(p1[0], p2[0], daily_volumes_series)
        waves.append({
            'wave_label': 'B',
            'sequence_id': seq_id,
            'start_ts': p1[0],
            'end_ts': p2[0],
            'start_price': p1[1],
            'end_price': p2[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_b['avg_volume'],
            'end_volume': vol_metrics_b['end_volume'],
            'volume_vs_prior_wave': (vol_metrics_b['avg_volume'] / vol_metrics_a['avg_volume']
                                     if vol_metrics_a['avg_volume'] and vol_metrics_a['avg_volume'] > 0 else None),
            'volume_trend': vol_metrics_b['volume_trend'],
        })

        # Wave C: p2 -> p3
        vol_metrics_c = get_wave_volume_metrics(p2[0], p3[0], daily_volumes_series)
        waves.append({
            'wave_label': 'C',
            'sequence_id': seq_id,
            'start_ts': p2[0],
            'end_ts': p3[0],
            'start_price': p2[1],
            'end_price': p3[1],
            'direction': direction,
            'rule_valid': rule_valid,
            'avg_volume': vol_metrics_c['avg_volume'],
            'end_volume': vol_metrics_c['end_volume'],
            'volume_vs_prior_wave': (vol_metrics_c['avg_volume'] / vol_metrics_b['avg_volume']
                                     if vol_metrics_b['avg_volume'] and vol_metrics_b['avg_volume'] > 0 else None),
            'volume_trend': vol_metrics_c['volume_trend'],
        })

    return waves, sequence_id_counter


def main():
    conn = get_connection()

    # Summary statistics
    all_impulses = []
    all_corrections = []
    summary_stats = []

    global_seq_id = 0

    # Process each symbol and threshold combination
    for symbol in SYMBOLS:
        print(f"\n--- Processing {symbol} ---")

        # Load daily volumes for this symbol (once per symbol for efficiency)
        daily_volumes = load_daily_volumes(conn, symbol)
        print(f"  Loaded daily volumes for {len(daily_volumes)} days")

        for threshold_pct in THRESHOLDS:
            print(f"  Threshold {threshold_pct*100:.1f}%: ", end="", flush=True)

            # Load pivots for this (symbol, threshold_pct)
            query = """
                SELECT ts, price, pivot_type
                FROM elliott_pivots
                WHERE symbol = %s AND threshold_pct = %s
                ORDER BY ts
            """
            pivots_df = pd.read_sql(query, conn, params=(symbol, threshold_pct))

            if pivots_df.empty:
                print(f"No pivots found")
                continue

            # Convert to list of tuples for easier manipulation
            pivots = [
                (pd.to_datetime(row['ts'], utc=True), row['price'], row['pivot_type'])
                for _, row in pivots_df.iterrows()
            ]
            print(f"Found {len(pivots)} pivots, ", end="", flush=True)

            # Generate impulse candidates
            impulse_waves, global_seq_id = generate_impulses(pivots, daily_volumes, global_seq_id)
            print(f"generated {len(impulse_waves) // 5} impulses, ", end="", flush=True)

            # Generate correction candidates
            correction_waves, global_seq_id = generate_corrections(
                pivots, daily_volumes, global_seq_id
            )
            print(f"{len(correction_waves) // 3} corrections")

            # Combine all waves for this (symbol, threshold_pct)
            all_waves = impulse_waves + correction_waves

            if all_waves:
                # Insert into database
                values = []
                for wave in all_waves:
                    # Convert numpy types to Python native types
                    avg_vol = float(wave['avg_volume']) if wave['avg_volume'] is not None else None
                    end_vol = float(wave['end_volume']) if wave['end_volume'] is not None else None
                    vol_ratio = float(wave['volume_vs_prior_wave']) if wave['volume_vs_prior_wave'] is not None else None

                    values.append((
                        symbol,
                        float(threshold_pct),
                        wave['wave_label'],
                        int(wave['sequence_id']),
                        str(threshold_pct),  # degree = threshold_pct as string
                        wave['start_ts'],
                        wave['end_ts'],
                        float(wave['start_price']),
                        float(wave['end_price']),
                        wave['direction'],
                        bool(wave['rule_valid']),
                        avg_vol,
                        end_vol,
                        vol_ratio,
                        wave['volume_trend'],
                        None,  # notes
                    ))

                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO elliott_waves
                        (symbol, threshold_pct, wave_label, sequence_id, degree,
                         start_ts, end_ts, start_price, end_price, direction,
                         rule_valid, avg_volume, end_volume, volume_vs_prior_wave,
                         volume_trend, notes)
                        VALUES %s
                        """,
                        values,
                    )
                conn.commit()

            # Accumulate summary
            all_impulses.extend(impulse_waves)
            all_corrections.extend(correction_waves)

            summary_stats.append({
                'symbol': symbol,
                'threshold_pct': threshold_pct,
                'num_impulses': len(impulse_waves) // 5 if impulse_waves else 0,
                'num_corrections': len(correction_waves) // 3 if correction_waves else 0,
                'impulses_valid': sum(1 for w in impulse_waves if w['rule_valid']) // 5 if impulse_waves else 0,
                'corrections_valid': sum(1 for w in correction_waves if w['rule_valid']) // 3 if correction_waves else 0,
            })

    conn.close()

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY: Elliott Wave Candidates Generated")
    print("="*100)
    print(f"{'Symbol':<12} {'Threshold':<12} {'Impulses':<15} {'Valid':<10} {'Corrections':<15} {'Valid':<10}")
    print("-"*100)

    total_impulses = 0
    total_impulses_valid = 0
    total_corrections = 0
    total_corrections_valid = 0

    for row in summary_stats:
        threshold_str = f"{row['threshold_pct']*100:.1f}%"
        print(f"{row['symbol']:<12} {threshold_str:<12} {row['num_impulses']:<15} "
              f"{row['impulses_valid']:<10} {row['num_corrections']:<15} {row['corrections_valid']:<10}")
        total_impulses += row['num_impulses']
        total_impulses_valid += row['impulses_valid']
        total_corrections += row['num_corrections']
        total_corrections_valid += row['corrections_valid']

    print("-"*100)
    print(f"{'TOTAL':<12} {'':<12} {total_impulses:<15} {total_impulses_valid:<10} "
          f"{total_corrections:<15} {total_corrections_valid:<10}")
    print("="*100)

    # Print sample rows
    print("\nSample Elliott Waves (first 5 waves from first valid impulse sequence):")
    print("-"*120)

    # Find first valid impulse sequence
    valid_impulses = [w for w in all_impulses if w['rule_valid']]
    if valid_impulses:
        sample_seq_id = valid_impulses[0]['sequence_id']
        sample_waves = [w for w in valid_impulses if w['sequence_id'] == sample_seq_id]

        # Query back the inserted rows to show actual DB values
        conn = get_connection()
        query = """
            SELECT symbol, wave_label, start_price, end_price, direction, rule_valid,
                   avg_volume, end_volume, volume_vs_prior_wave, volume_trend
            FROM elliott_waves
            WHERE sequence_id = %s AND wave_label IN ('1', '2', '3', '4', '5')
            ORDER BY wave_label
        """
        sample_df = pd.read_sql(query, conn, params=(sample_seq_id,))
        conn.close()

        if not sample_df.empty:
            print(f"Sequence ID {sample_seq_id}:")
            print(sample_df.to_string(index=False))
        else:
            print("(No sample data available)")
    else:
        print("(No valid impulse sequences found)")

    print("="*120)
    print(f"\nDone. Total waves inserted: {len(all_impulses) + len(all_corrections)}")


if __name__ == "__main__":
    main()
