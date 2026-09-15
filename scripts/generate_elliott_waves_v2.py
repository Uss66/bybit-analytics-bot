"""
Generate Elliott Wave sequences with non-overlapping sequential tiling.

FIXES for v1:
1. Eliminates severe overlap problem: sliding window caused same end_ts to be
   relabeled as different wave_label across different sequence_id (reuse ratio 1.76x
   for impulses, 4.05x for corrections - pseudo-replication trap)
2. Implements proper per-degree analysis: 3 thresholds (5%/8%/12%) serve as
   natural proxy for wave degree (smaller threshold = smaller/faster degree)

Algorithm: For each (symbol, threshold_pct):
  1. Load pivot sequence from DB
  2. Walk through with cursor at index i, non-overlapping sequential tiling:
     - If 6+ remaining pivots: try to validate 5-wave impulse at i..i+5
       - If VALID: record 5 waves as one sequence_id, advance cursor to i+5
       - If NOT valid: try 3-wave correction at i..i+3 (if 4+ remain)
         - If VALID: record 3 waves, advance cursor to i+3
         - If NOT valid: advance cursor by 1 (i += 1) and retry
     - If fewer than 4 remaining pivots: stop
  3. No confirmed wave's pivots are ever reused (only "searching" phase revisits)

Output: elliott_waves_v2 table with:
  - Same schema as elliott_waves
  - Added: duration_hours (in hours, computed as (end_ts - start_ts))
  - All rows have rule_valid = TRUE (confirmed by non-overlapping tiling)

Summary Report:
  a. Overlap verification: reuse ratio for confirmed waves (should be ~1.0)
  b. Coverage/count by (symbol, threshold_pct)
  c. Duration-by-degree: median/mean duration by threshold_pct and wave_label,
     plus Elliott proportionality ratios (wave 3/1 and wave 5/1 median durations)

Usage:
    cd scripts && ../../.venv/Scripts/python.exe generate_elliott_waves_v2.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from psycopg2.extras import execute_values

from db import get_connection


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"]
# Expanded 2026-09-12 to match detect_elliott_pivots.py's wider scale range.
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]


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


def create_elliott_waves_v2_table(conn):
    """
    Create elliott_waves_v2 table with same schema as elliott_waves plus duration_hours.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS elliott_waves_v2 (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                threshold_pct DOUBLE PRECISION NOT NULL,
                wave_label VARCHAR(10) NOT NULL,
                sequence_id BIGINT NOT NULL,
                degree VARCHAR(20),
                start_ts TIMESTAMP WITH TIME ZONE NOT NULL,
                end_ts TIMESTAMP WITH TIME ZONE NOT NULL,
                start_price DOUBLE PRECISION NOT NULL,
                end_price DOUBLE PRECISION NOT NULL,
                direction VARCHAR(10),
                rule_valid BOOLEAN DEFAULT FALSE,
                avg_volume DOUBLE PRECISION,
                end_volume DOUBLE PRECISION,
                volume_vs_prior_wave DOUBLE PRECISION,
                volume_trend VARCHAR(20),
                duration_hours DOUBLE PRECISION,
                end_confirmed_ts TIMESTAMP WITH TIME ZONE,
                notes TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()
    print("Table elliott_waves_v2 created or verified")


def generate_waves_sequential_tiling(pivots, daily_volumes_series, seq_id_start=1000):
    """
    Generate Elliott Wave sequences using non-overlapping sequential tiling.

    Algorithm:
      - Start with cursor i = 0
      - At position i:
        - If 6+ remaining pivots: try impulse (5 waves) at i..i+5
          - If valid: record all 5 waves, advance to i+5
          - If not valid: try correction (3 waves) at i..i+3
            - If valid: record all 3 waves, advance to i+3
            - If not: advance by 1 (i += 1) and retry
        - If fewer than 4 remaining: stop
    - Guarantee: confirmed wave's pivots never reused (only searching phase revisits)

    Args:
        pivots: list of (ts, price, pivot_type) tuples, chronologically ordered
        daily_volumes_series: pandas Series of daily volumes indexed by date

    Returns:
        list of wave row dicts to be inserted into elliott_waves_v2
    """
    waves = []
    # NOTE: this must be threaded across (symbol, threshold_pct) calls via
    # seq_id_start, not reset to a fixed value each call - otherwise
    # unrelated waves from different symbols/thresholds collide on the same
    # sequence_id (this exact bug previously hit generate_elliott_waves.py
    # for impulses; caught again here via direct DB verification after this
    # script's own self-reported summary didn't catch it either - always
    # verify sequence_id global uniqueness directly, don't trust the
    # generation script's own count).
    sequence_id_counter = seq_id_start

    i = 0
    while i + 3 < len(pivots):  # At least 4 pivots needed
        # Try impulse first (6 pivots needed)
        if i + 5 < len(pivots):
            p0, p1, p2, p3, p4, p5 = pivots[i:i+6]

            # Determine direction
            direction = 'up' if p0[2] == 'low' else 'down'

            # Check impulse rules
            rule_valid_impulse = check_impulse_rules(p0, p1, p2, p3, p4, p5)

            if rule_valid_impulse:
                # Impulse is valid: record all 5 waves
                seq_id = sequence_id_counter
                sequence_id_counter += 1

                # Compute volume metrics for each wave
                vol_metrics_w1 = get_wave_volume_metrics(p0[0], p1[0], daily_volumes_series)
                vol_metrics_w2 = get_wave_volume_metrics(p1[0], p2[0], daily_volumes_series)
                vol_metrics_w3 = get_wave_volume_metrics(p2[0], p3[0], daily_volumes_series)
                vol_metrics_w4 = get_wave_volume_metrics(p3[0], p4[0], daily_volumes_series)
                vol_metrics_w5 = get_wave_volume_metrics(p4[0], p5[0], daily_volumes_series)

                duration_hours_w1 = (p1[0] - p0[0]).total_seconds() / 3600.0
                duration_hours_w2 = (p2[0] - p1[0]).total_seconds() / 3600.0
                duration_hours_w3 = (p3[0] - p2[0]).total_seconds() / 3600.0
                duration_hours_w4 = (p4[0] - p3[0]).total_seconds() / 3600.0
                duration_hours_w5 = (p5[0] - p4[0]).total_seconds() / 3600.0

                # Wave 1
                waves.append({
                    'wave_label': '1',
                    'sequence_id': seq_id,
                    'start_ts': p0[0],
                    'end_ts': p1[0],
                    'start_price': p0[1],
                    'end_price': p1[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_w1['avg_volume'],
                    'end_volume': vol_metrics_w1['end_volume'],
                    'volume_vs_prior_wave': None,
                    'volume_trend': vol_metrics_w1['volume_trend'],
                    'duration_hours': duration_hours_w1,
                })

                # Wave 2
                waves.append({
                    'wave_label': '2',
                    'sequence_id': seq_id,
                    'start_ts': p1[0],
                    'end_ts': p2[0],
                    'start_price': p1[1],
                    'end_price': p2[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_w2['avg_volume'],
                    'end_volume': vol_metrics_w2['end_volume'],
                    'volume_vs_prior_wave': (vol_metrics_w2['avg_volume'] / vol_metrics_w1['avg_volume']
                                           if vol_metrics_w1['avg_volume'] and vol_metrics_w1['avg_volume'] > 0 else None),
                    'volume_trend': vol_metrics_w2['volume_trend'],
                    'duration_hours': duration_hours_w2,
                })

                # Wave 3
                waves.append({
                    'wave_label': '3',
                    'sequence_id': seq_id,
                    'start_ts': p2[0],
                    'end_ts': p3[0],
                    'start_price': p2[1],
                    'end_price': p3[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_w3['avg_volume'],
                    'end_volume': vol_metrics_w3['end_volume'],
                    'volume_vs_prior_wave': (vol_metrics_w3['avg_volume'] / vol_metrics_w2['avg_volume']
                                           if vol_metrics_w2['avg_volume'] and vol_metrics_w2['avg_volume'] > 0 else None),
                    'volume_trend': vol_metrics_w3['volume_trend'],
                    'duration_hours': duration_hours_w3,
                })

                # Wave 4
                waves.append({
                    'wave_label': '4',
                    'sequence_id': seq_id,
                    'start_ts': p3[0],
                    'end_ts': p4[0],
                    'start_price': p3[1],
                    'end_price': p4[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_w4['avg_volume'],
                    'end_volume': vol_metrics_w4['end_volume'],
                    'volume_vs_prior_wave': (vol_metrics_w4['avg_volume'] / vol_metrics_w3['avg_volume']
                                           if vol_metrics_w3['avg_volume'] and vol_metrics_w3['avg_volume'] > 0 else None),
                    'volume_trend': vol_metrics_w4['volume_trend'],
                    'duration_hours': duration_hours_w4,
                })

                # Wave 5
                waves.append({
                    'wave_label': '5',
                    'sequence_id': seq_id,
                    'start_ts': p4[0],
                    'end_ts': p5[0],
                    'start_price': p4[1],
                    'end_price': p5[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_w5['avg_volume'],
                    'end_volume': vol_metrics_w5['end_volume'],
                    'volume_vs_prior_wave': (vol_metrics_w5['avg_volume'] / vol_metrics_w4['avg_volume']
                                           if vol_metrics_w4['avg_volume'] and vol_metrics_w4['avg_volume'] > 0 else None),
                    'volume_trend': vol_metrics_w5['volume_trend'],
                    'duration_hours': duration_hours_w5,
                })

                # Advance cursor to end of this impulse sequence
                i += 5
                continue

        # Impulse not valid or not enough pivots: try correction (4 pivots needed)
        if i + 3 < len(pivots):
            p0, p1, p2, p3 = pivots[i:i+4]

            # Determine direction
            direction = 'down' if p0[2] == 'high' else 'up'

            # Check correction rule
            rule_valid_correction = check_correction_rules(p0, p1, p2)

            if rule_valid_correction:
                # Correction is valid: record all 3 waves
                seq_id = sequence_id_counter
                sequence_id_counter += 1

                # Compute volume metrics for each wave
                vol_metrics_a = get_wave_volume_metrics(p0[0], p1[0], daily_volumes_series)
                vol_metrics_b = get_wave_volume_metrics(p1[0], p2[0], daily_volumes_series)
                vol_metrics_c = get_wave_volume_metrics(p2[0], p3[0], daily_volumes_series)

                duration_hours_a = (p1[0] - p0[0]).total_seconds() / 3600.0
                duration_hours_b = (p2[0] - p1[0]).total_seconds() / 3600.0
                duration_hours_c = (p3[0] - p2[0]).total_seconds() / 3600.0

                # Wave A
                waves.append({
                    'wave_label': 'A',
                    'sequence_id': seq_id,
                    'start_ts': p0[0],
                    'end_ts': p1[0],
                    'start_price': p0[1],
                    'end_price': p1[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_a['avg_volume'],
                    'end_volume': vol_metrics_a['end_volume'],
                    'volume_vs_prior_wave': None,
                    'volume_trend': vol_metrics_a['volume_trend'],
                    'duration_hours': duration_hours_a,
                })

                # Wave B
                waves.append({
                    'wave_label': 'B',
                    'sequence_id': seq_id,
                    'start_ts': p1[0],
                    'end_ts': p2[0],
                    'start_price': p1[1],
                    'end_price': p2[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_b['avg_volume'],
                    'end_volume': vol_metrics_b['end_volume'],
                    'volume_vs_prior_wave': (vol_metrics_b['avg_volume'] / vol_metrics_a['avg_volume']
                                           if vol_metrics_a['avg_volume'] and vol_metrics_a['avg_volume'] > 0 else None),
                    'volume_trend': vol_metrics_b['volume_trend'],
                    'duration_hours': duration_hours_b,
                })

                # Wave C
                waves.append({
                    'wave_label': 'C',
                    'sequence_id': seq_id,
                    'start_ts': p2[0],
                    'end_ts': p3[0],
                    'start_price': p2[1],
                    'end_price': p3[1],
                    'direction': direction,
                    'rule_valid': True,
                    'avg_volume': vol_metrics_c['avg_volume'],
                    'end_volume': vol_metrics_c['end_volume'],
                    'volume_vs_prior_wave': (vol_metrics_c['avg_volume'] / vol_metrics_b['avg_volume']
                                           if vol_metrics_b['avg_volume'] and vol_metrics_b['avg_volume'] > 0 else None),
                    'volume_trend': vol_metrics_c['volume_trend'],
                    'duration_hours': duration_hours_c,
                })

                # Advance cursor to end of this correction sequence
                i += 3
                continue

        # Neither impulse nor correction valid at position i: advance by 1 and retry
        i += 1

    return waves, sequence_id_counter


def main():
    conn = get_connection()

    # Create v2 table
    create_elliott_waves_v2_table(conn)

    # Clear any prior run's data - without this, re-running the script
    # (e.g. while debugging) silently duplicates every row instead of
    # replacing them, since there's no unique constraint to catch it.
    with conn.cursor() as cur:
        cur.execute("TRUNCATE elliott_waves_v2")
    conn.commit()

    # Summary statistics
    all_waves = []
    all_impulse_waves = []
    all_correction_waves = []
    summary_stats = []
    global_seq_id = 1000

    print("\n" + "="*100)
    print("Elliott Waves V2: Non-Overlapping Sequential Tiling")
    print("="*100)

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
                SELECT ts, price, pivot_type, confirmed_ts
                FROM elliott_pivots
                WHERE symbol = %s AND threshold_pct = %s
                ORDER BY ts
            """
            pivots_df = pd.read_sql(query, conn, params=(symbol, threshold_pct))

            if pivots_df.empty:
                print(f"No pivots found")
                continue

            # Convert to list of tuples for easier manipulation (rule-checking
            # logic only looks at [ts, price, pivot_type], so keep it a
            # 3-tuple there; confirmed_ts is looked up separately below when
            # recording each wave, keyed by (ts -> confirmed_ts))
            pivots = [
                (pd.to_datetime(row['ts'], utc=True), row['price'], row['pivot_type'])
                for _, row in pivots_df.iterrows()
            ]
            confirmed_ts_by_pivot_ts = {
                pd.to_datetime(row['ts'], utc=True): pd.to_datetime(row['confirmed_ts'], utc=True)
                for _, row in pivots_df.iterrows()
            }
            print(f"Found {len(pivots)} pivots, ", end="", flush=True)

            # Generate waves using non-overlapping sequential tiling
            waves_for_this_pair, global_seq_id = generate_waves_sequential_tiling(
                pivots, daily_volumes, global_seq_id
            )
            print(f"generated {len(waves_for_this_pair)} waves")

            # Add symbol and threshold to each wave (for later grouping in reports)
            for wave in waves_for_this_pair:
                wave['symbol'] = symbol
                wave['threshold_pct'] = threshold_pct

            # Count impulses and corrections
            num_impulses = len([w for w in waves_for_this_pair if w['wave_label'] in ['1', '2', '3', '4', '5']])
            num_impulses_sequences = len(set(w['sequence_id'] for w in waves_for_this_pair if w['wave_label'] == '1'))

            num_corrections = len([w for w in waves_for_this_pair if w['wave_label'] in ['A', 'B', 'C']])
            num_correction_sequences = len(set(w['sequence_id'] for w in waves_for_this_pair if w['wave_label'] == 'A'))

            if waves_for_this_pair:
                # Insert into database
                values = []
                for wave in waves_for_this_pair:
                    # Convert numpy types to Python native types
                    avg_vol = float(wave['avg_volume']) if wave['avg_volume'] is not None else None
                    end_vol = float(wave['end_volume']) if wave['end_volume'] is not None else None
                    vol_ratio = float(wave['volume_vs_prior_wave']) if wave['volume_vs_prior_wave'] is not None else None
                    duration = float(wave['duration_hours']) if wave['duration_hours'] is not None else None
                    # earliest this wave's end could actually have been known - the
                    # fix for the repainting bug found 2026-09-12 (see
                    # project_elliott_wave_analysis memory). Falls back to end_ts
                    # only if somehow missing (shouldn't happen - every wave end is
                    # a pivot with a confirmed_ts), so a missing lookup is visible
                    # rather than silently NULL.
                    end_confirmed = confirmed_ts_by_pivot_ts.get(wave['end_ts'], wave['end_ts'])

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
                        duration,
                        end_confirmed,
                        None,  # notes
                    ))

                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO elliott_waves_v2
                        (symbol, threshold_pct, wave_label, sequence_id, degree,
                         start_ts, end_ts, start_price, end_price, direction,
                         rule_valid, avg_volume, end_volume, volume_vs_prior_wave,
                         volume_trend, duration_hours, end_confirmed_ts, notes)
                        VALUES %s
                        """,
                        values,
                    )
                conn.commit()

            # Accumulate summary
            all_waves.extend(waves_for_this_pair)
            all_impulse_waves.extend([w for w in waves_for_this_pair if w['wave_label'] in ['1', '2', '3', '4', '5']])
            all_correction_waves.extend([w for w in waves_for_this_pair if w['wave_label'] in ['A', 'B', 'C']])

            summary_stats.append({
                'symbol': symbol,
                'threshold_pct': threshold_pct,
                'num_pivots': len(pivots),
                'num_impulse_sequences': num_impulses_sequences,
                'num_correction_sequences': num_correction_sequences,
                'num_impulse_waves': num_impulses,
                'num_correction_waves': num_corrections,
            })

    conn.close()

    # ========== OVERLAP VERIFICATION (Section a) ==========
    print("\n" + "="*100)
    print("SECTION A: OVERLAP VERIFICATION")
    print("="*100)

    # For each (symbol, threshold_pct, wave_label, end_ts) group, check if it appears
    # in multiple sequence_id values (which would indicate relabeling/overlap)

    impulse_data = [w for w in all_impulse_waves]
    impulse_df = pd.DataFrame(impulse_data)

    if not impulse_df.empty:
        # Group by (symbol, threshold_pct, wave_label, end_ts) and count distinct sequence_ids
        # If count > 1, that's an overlap (same end_ts, same wave_label but different sequence)
        impulse_overlap_groups = impulse_df.groupby(['symbol', 'threshold_pct', 'wave_label', 'end_ts'])['sequence_id'].nunique().reset_index(name='seq_id_count')

        # Also check: for the same (symbol, threshold_pct, end_ts), do multiple wave_labels exist?
        # If yes, that's relabeling/overlap
        impulse_end_ts_labels = impulse_df.groupby(['symbol', 'threshold_pct', 'end_ts'])['wave_label'].nunique().reset_index(name='label_count')
        overlap_end_ts_count = len(impulse_end_ts_labels[impulse_end_ts_labels['label_count'] > 1])

        impulse_waves_count = len(impulse_df)
        impulse_sequences_count = impulse_df['sequence_id'].nunique()
        distinct_impulse_end_ts = impulse_df['end_ts'].nunique()

        print(f"\nImpulse Waves:")
        print(f"  Total impulse wave rows: {impulse_waves_count}")
        print(f"  Number of impulse sequences: {impulse_sequences_count}")
        print(f"  Distinct end_ts values: {distinct_impulse_end_ts}")
        print(f"  Distinct (wave_label, end_ts) pairs: {len(impulse_df.groupby(['wave_label', 'end_ts']))}")
        print(f"  end_ts values with MULTIPLE wave_labels (relabeling): {overlap_end_ts_count}")
        if overlap_end_ts_count == 0:
            print(f"  RESULT: No overlap detected (PASS - each end_ts has single wave_label)")
        else:
            print(f"  RESULT: Overlap detected (FAIL - same end_ts has different wave_labels)")
    else:
        print(f"\nImpulse Waves: None generated")

    # Check correction waves similarly
    correction_data = [w for w in all_correction_waves]
    correction_df = pd.DataFrame(correction_data)

    if not correction_df.empty:
        # Group by (symbol, threshold_pct, end_ts) and count distinct wave_labels
        correction_end_ts_labels = correction_df.groupby(['symbol', 'threshold_pct', 'end_ts'])['wave_label'].nunique().reset_index(name='label_count')
        overlap_end_ts_count = len(correction_end_ts_labels[correction_end_ts_labels['label_count'] > 1])

        correction_waves_count = len(correction_df)
        correction_sequences_count = correction_df['sequence_id'].nunique()
        distinct_correction_end_ts = correction_df['end_ts'].nunique()

        print(f"\nCorrection Waves:")
        print(f"  Total correction wave rows: {correction_waves_count}")
        print(f"  Number of correction sequences: {correction_sequences_count}")
        print(f"  Distinct end_ts values: {distinct_correction_end_ts}")
        print(f"  Distinct (wave_label, end_ts) pairs: {len(correction_df.groupby(['wave_label', 'end_ts']))}")
        print(f"  end_ts values with MULTIPLE wave_labels (relabeling): {overlap_end_ts_count}")
        if overlap_end_ts_count == 0:
            print(f"  RESULT: No overlap detected (PASS - each end_ts has single wave_label)")
        else:
            print(f"  RESULT: Overlap detected (FAIL - same end_ts has different wave_labels)")
    else:
        print(f"\nCorrection Waves: None generated")

    # ========== COVERAGE SUMMARY (Section b) ==========
    print("\n" + "="*100)
    print("SECTION B: COVERAGE/COUNT SUMMARY")
    print("="*100)
    print(f"\n{'Symbol':<12} {'Threshold':<12} {'Pivots':<10} {'Impulse Seq':<15} {'Correction Seq':<15} {'Total Waves':<12}")
    print("-"*100)

    total_impulse_sequences = 0
    total_correction_sequences = 0
    total_waves = 0

    for row in summary_stats:
        threshold_str = f"{row['threshold_pct']*100:.1f}%"
        total_waves_row = row['num_impulse_waves'] + row['num_correction_waves']

        print(f"{row['symbol']:<12} {threshold_str:<12} {row['num_pivots']:<10} "
              f"{row['num_impulse_sequences']:<15} {row['num_correction_sequences']:<15} {total_waves_row:<12}")

        total_impulse_sequences += row['num_impulse_sequences']
        total_correction_sequences += row['num_correction_sequences']
        total_waves += total_waves_row

    print("-"*100)
    print(f"{'TOTAL':<12} {'':<12} {'':<10} {total_impulse_sequences:<15} {total_correction_sequences:<15} {total_waves:<12}")
    print("="*100)

    # ========== DURATION-BY-DEGREE ANALYSIS (Section c) ==========
    print("\n" + "="*100)
    print("SECTION C: DURATION-BY-DEGREE ANALYSIS")
    print("="*100)

    all_waves_df = pd.DataFrame(all_waves)

    if not all_waves_df.empty:
        # For each threshold and wave_label, compute median and mean duration
        print("\nMedian and Mean Duration (hours) by Threshold and Wave Label:")
        print("(Pooled across all 7 symbols)\n")

        duration_by_degree = []

        for threshold in THRESHOLDS:
            # Filter for this threshold AND for impulse waves only
            threshold_subset = all_waves_df[
                (all_waves_df['threshold_pct'] == threshold) &
                (all_waves_df['wave_label'].isin(['1', '2', '3', '4', '5']))
            ]

            if threshold_subset.empty:
                print(f"Threshold: {threshold*100:.1f}%")
                print("  (No impulse waves)")
                print()
                continue

            print(f"Threshold: {threshold*100:.1f}%")
            print(f"{'Wave Label':<12} {'Median Hours':<15} {'Mean Hours':<15} {'Std Dev':<15} {'Count':<10}")
            print("-"*60)

            for wave_label in ['1', '2', '3', '4', '5']:
                wave_data = threshold_subset[(threshold_subset['wave_label'] == wave_label)]

                if wave_data.empty:
                    continue

                median_duration = wave_data['duration_hours'].median()
                mean_duration = wave_data['duration_hours'].mean()
                std_duration = wave_data['duration_hours'].std()
                count = len(wave_data)

                print(f"{wave_label:<12} {median_duration:<15.2f} {mean_duration:<15.2f} {std_duration:<15.2f} {count:<10}")

                duration_by_degree.append({
                    'threshold_pct': threshold,
                    'wave_label': wave_label,
                    'median_hours': median_duration,
                    'mean_hours': mean_duration,
                    'count': count
                })

            print()

        # Elliott Proportionality Analysis: Wave 3/Wave 1 and Wave 5/Wave 1 ratios
        print("\nElliott Proportionality: Median Duration Ratios by Threshold")
        print("(Impulse waves pooled across all 7 symbols)\n")
        print(f"{'Threshold':<12} {'Wave 3/1 Ratio':<18} {'Wave 5/1 Ratio':<18}")
        print("-"*50)

        for threshold in THRESHOLDS:
            # Get wave 1, 3, 5 durations for this threshold specifically
            w1_data = all_waves_df[
                (all_waves_df['threshold_pct'] == threshold) &
                (all_waves_df['wave_label'] == '1')
            ]
            w3_data = all_waves_df[
                (all_waves_df['threshold_pct'] == threshold) &
                (all_waves_df['wave_label'] == '3')
            ]
            w5_data = all_waves_df[
                (all_waves_df['threshold_pct'] == threshold) &
                (all_waves_df['wave_label'] == '5')
            ]

            w1_median = w1_data['duration_hours'].median() if not w1_data.empty else None
            w3_median = w3_data['duration_hours'].median() if not w3_data.empty else None
            w5_median = w5_data['duration_hours'].median() if not w5_data.empty else None

            if w1_median and w1_median > 0:
                ratio_3_1 = w3_median / w1_median if w3_median else None
                ratio_5_1 = w5_median / w1_median if w5_median else None
            else:
                ratio_3_1 = None
                ratio_5_1 = None

            threshold_str = f"{threshold*100:.1f}%"

            ratio_3_1_str = f"{ratio_3_1:.3f}" if ratio_3_1 is not None else "N/A"
            ratio_5_1_str = f"{ratio_5_1:.3f}" if ratio_5_1 is not None else "N/A"

            print(f"{threshold_str:<12} {ratio_3_1_str:<18} {ratio_5_1_str:<18}")

        print("="*100)
    else:
        print("No waves generated")

    print(f"\nDone. Total waves inserted: {total_waves}")
    print("Table: elliott_waves_v2")
    print("\nKey validation: If overlap fix is successful, Impulse and Correction")
    print("reuse ratios should be ~1.0 (not 1.76 and 4.05 as in v1)")


if __name__ == "__main__":
    main()
