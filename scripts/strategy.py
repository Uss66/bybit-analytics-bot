"""
Reusable long-only (spot) signal strategy derived from event_study.py's
findings, plus a simple non-overlapping backtest against buy-and-hold that
accounts for real Bybit spot fees.

v3 (2026-08-28): a much bigger bug than fees was hiding here. The
score-based backtest (added to fix an earlier "AVOID blocks everything"
bug) turned discrete events into hold-window "is this rule active"
indicators - but it never checked whether a rule is EVER inactive. Any
category with weekly hold windows (168h) and more than roughly one
qualifying event every few days has overlapping windows essentially all the
time, so the rule is active ~97-100% of the timeline and contributes a
constant bias, not a signal. Measured directly on data (see
project_funding_oi_findings / event_study memory): `exchange_outflow_large`,
`exchange_inflow_large`, `whale_wallet_move_large`, `stablecoin_mint_large`
(all $50M+ whale-transfer categories) and even `macro_negative` were active
97-100% of the time on their target symbols. Raising the whale magnitude
threshold doesn't fix this cleanly either - re-running event_study.py at
$500M+ showed some of these categories' effects flip sign entirely
(different population of events, not a cleaner version of the same signal),
so it needs fresh validation, not just a bigger number, before going back in.

All of the above are parked in PARKED_HIGH_DUTY_CYCLE_RULES below, pending a
redesign (shorter windows, and/or a genuinely rare trigger, and/or an
aggregate-flow measure instead of "did one qualifying event happen").
The only two rules still active are the two with naturally low, genuinely
selective duty cycles: `fgi_greed` (Fear & Greed Index, ~4% of the time) and
`technical_hype` (4h window, ~2% of the time). Full period + holdout numbers
were re-run with the corrected scorer before trusting this state - see
README.md "Результаты".

This is still in-sample and exploratory - see README.md "Результаты" for the
full caveats. Treat this as a documented, testable starting point, not a
finished edge. Always backtest with --fee set to the real fee before trusting
a number, and read event_study_summary.csv yourself before extending RULES.

Usage:
    python scripts/strategy.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --capital 10000
"""
import argparse

import numpy as np
import pandas as pd

from db import get_connection

SPOT_TAKER_FEE = 0.001  # Bybit non-VIP spot taker fee, one side (0.1%)
WHALE_MIN_USD = 50_000_000  # magnitude filter matching the event_study run these rules were derived from

# category, sentiment_sign filter (None = any), hold horizon, which symbols
# this was actually validated on (event_study_summary.csv), min USD value for
# whale-alert-derived categories (ignored for non-whale categories), rationale.
RULES = [
    dict(
        name="fgi_greed",
        category="fgi_greed",
        sentiment_sign=None,
        horizon_hours=72,
        symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"},
        min_usd_value=0,
        weight=3,
        rationale="Fear & Greed Index in 55-80 band: the strongest validated signal so far - positive "
        "and consistent in BOTH train and holdout, on BTC/ETH/SOL, at both 24h (t=5.4-7.8 holdout) "
        "and 72h (t=9.7-14.1 holdout). extreme_fear/extreme_greed zones were tested too but flipped "
        "sign between train/holdout - not included. Extended 2026-08-28 to XRP/DOGE/BNB/LINK after "
        "confirming the same signal on train+holdout for each (72h holdout: XRP t=4.10, DOGE t=2.18, "
        "BNB t=10.35(!), LINK t=3.24). ADA tested too but excluded - train was strong (t=4.12) but "
        "holdout collapsed to t=0.87, not confirmed. NOTE on portfolio backtest (not the event_study "
        "above): XRP and LINK show real train-period losses despite the valid signal - XRP because "
        "the strategy can't ride a +271% moonshot regardless of stop-loss width (expected cost of "
        "risk management), LINK because its 6 discrete train trades were badly timed even with the "
        "stop removed entirely (likely small-sample noise, not a broken signal - holdout t=3.24 "
        "backtest +8.4% vs buy&hold -4.3% was clean). Kept active per user decision 2026-08-28; "
        "LINK is the least-confident of the four additions, revisit if more history disagrees.",
    ),
    dict(
        name="dxy_up_extreme",
        category="dxy_up_extreme",
        sentiment_sign=None,
        horizon_hours=168,
        symbols={"ETHUSDT", "SOLUSDT"},
        min_usd_value=0,
        weight=2,
        rationale="DXY (dollar index) top quintile of trailing-7d change: +5.7%/wk SOL (t=5.55 train, "
        "3.39 holdout - both significant), +4.5%/wk ETH (t=4.69 train, 1.85 holdout - weaker but same "
        "direction). BTC excluded: train ~flat, holdout not significant. Counter to the popular "
        "'strong dollar bad for crypto' narrative - validated on this data regardless. Duty cycle "
        "~38%, see classify_traditional_markets.py.",
    ),
    dict(
        name="dvol_drop_extreme",
        category="dvol_drop_extreme",
        sentiment_sign=None,
        horizon_hours=24,
        symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"},
        min_usd_value=0,
        weight=2,
        rationale="Deribit BTC DVOL (options-implied vol, crypto's VIX) bottom quintile of 1-day "
        "change (<=-2.61%, 'vol crush') at 24h: strongly positive forward returns, same sign in "
        "train AND holdout, across ALL 7 active symbols (holdout t=3.1-4.7, train t=1.8-3.5, "
        "holdout consistently stronger than train, not weaker). Same underlying vol-shock mechanism "
        "as VIX-crush effects in traditional markets. Duty cycle ~20% (quintile-based, non-"
        "overlapping daily events at 24h hold - see classify_deribit_dvol.py). Added 2026-08-28.",
    ),
]

# Tested 2026-08-28 and REJECTED, for one of two reasons: (a) failed an
# out-of-sample train/holdout split (train: <2025-12-01, holdout:
# >=2025-12-01) despite looking promising on the full period, or (b) passed
# its own standalone train/holdout + duty-cycle checks but still made the
# *combined* portfolio backtest worse once added to RULES/AVOID_RULES -
# reason (b) is a distinct trap from (a): a real, validated standalone
# effect can still hurt if it fires often enough to rack up fee drag or
# crowd out cleaner, higher-conviction signals. Always re-run the full
# strategy.py backtest (not just event_study.py) before trusting that a
# validated candidate is actually worth adding. Kept here, not deleted, so
# these specific experiments don't get silently re-run and re-"discovered".
REJECTED_RULES = [
    dict(
        name="key_person_statement_positive", category="key_person_statement", sentiment_sign="positive",
        horizon_hours=168, symbols={"BTCUSDT"}, min_usd_value=0,
        rationale="Full-period: +0.32%/wk BTC (t=2.10, n=1705). Contributed to BTC's holdout failure.",
    ),
    dict(
        name="key_person_statement_charged_positive_4h", category="key_person_statement", sentiment_sign="positive",
        horizon_hours=4, symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT"}, min_usd_value=0,
        rationale="Reason (b): at 168h looked like the strongest signal of the whole project (t up to "
        "5.97, n>1000) but duty cycle was 100% (constant, not a signal - same trap as the parked whale "
        "rules). At 4h, duty cycle drops to a usable 35-47% and the standalone effect still validates "
        "(train t=1.1-1.7 weak, holdout t=3.3-4.3 strong, same direction, all 3 symbols, neutral-bucket "
        "weaker so not just drift) - but adding it to the live RULES/AVOID_RULES mix made every symbol's "
        "holdout result worse (BTC flipped from +12.0% to -13.9% vs buy&hold, trade count exploded from "
        "2 to 104). It fires too often (thousands of person-mentions/2yr) - fee drag plus crowding out "
        "fgi_greed/dxy_up_extreme's cleaner signal outweighs its own modest 0.06-0.14%/trade edge.",
    ),
    dict(
        name="key_person_statement_charged_negative_4h", category="key_person_statement", sentiment_sign="negative",
        horizon_hours=4, symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT"}, min_usd_value=0,
        rationale="Mirror of key_person_statement_charged_positive_4h, rejected for the same reason (b) - "
        "see that entry.",
    ),
    dict(
        name="etf_positive", category="etf", sentiment_sign="positive", horizon_hours=72,
        symbols={"BTCUSDT"}, min_usd_value=0,
        rationale="Full-period: +0.49% at 72h BTC (t=2.71, n=584). Contributed to BTC's holdout failure.",
    ),
    dict(
        name="regulation_negative", category="regulation", sentiment_sign="negative", horizon_hours=72,
        symbols={"BTCUSDT"}, min_usd_value=0,
        rationale="Full-period: +0.30% at 72h BTC (t=2.09, n=727). Contributed to BTC's holdout failure.",
    ),
    dict(
        name="institutional_adoption_reversal", category="institutional_adoption", sentiment_sign="positive",
        horizon_hours=72, symbols={"SOLUSDT"}, min_usd_value=0,
        rationale="Full-period: -0.56% at 72h SOL (t=-2.18, n=744). Never fired in the holdout window "
        "either way - inert, not harmful, but unproven.",
    ),
    dict(
        name="exchange_security_positive", category="exchange_security", sentiment_sign="positive",
        horizon_hours=72, symbols={"BTCUSDT", "ETHUSDT"}, min_usd_value=0,
        rationale="Full-period: -0.80% at 72h BTC (t=-2.6, n=178). Small n, didn't move the holdout "
        "result meaningfully either way - parked pending more data, not confirmed useful.",
    ),
]

AVOID_RULES = [
    dict(
        name="technical_hype",
        category="technical",
        sentiment_sign="positive",
        horizon_hours=4,
        symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT"},
        min_usd_value=0,
        weight=1,
        rationale="Negative fwd return at 4h across BTC/ETH/SOL (t=-2.0 to -2.2, n=117) - "
        "'buy the upgrade announcement, sell the actual thing' pattern.",
    ),
    dict(
        name="dxy_down_extreme",
        category="dxy_down_extreme",
        sentiment_sign=None,
        horizon_hours=168,
        symbols={"ETHUSDT", "SOLUSDT"},
        min_usd_value=0,
        weight=2,
        rationale="DXY bottom quintile of trailing-7d change: -4.0%/wk SOL (t=-2.76 holdout), "
        "-3.4%/wk ETH (t=-2.36 holdout) - both significant in holdout, train weaker/mixed but same "
        "sign. Mirror of dxy_up_extreme. Duty cycle ~33-38%.",
    ),
    dict(
        name="dvol_spike_extreme",
        category="dvol_spike_extreme",
        sentiment_sign=None,
        horizon_hours=24,
        symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT"},
        min_usd_value=0,
        weight=2,
        rationale="Mirror of dvol_drop_extreme: DVOL top quintile of 1-day change (>=+2.35%, 'vol "
        "shock') at 24h predicts negative forward returns, same sign in train and holdout across all "
        "7 symbols (holdout t=-2.0 to -4.4, train t=-0.8 to -3.1 - XRP/DOGE weak in train, t=-1.0/"
        "-0.8, but confirm strongly in holdout). Duty cycle ~20%. Added 2026-08-28.",
    ),
]

# Parked 2026-08-28: each of these measured a real, statistically solid
# event-study effect (t-stats 2-11), but at their 168h hold window they're
# active 97-100% of the backtest timeline (measured directly - see the
# strategy.py module docstring), so under correct scoring they contribute a
# constant bias, not a discriminating signal. Not moved to REJECTED_RULES
# because the underlying event-study finding may still be real - they need a
# redesign (shorter window / rarer trigger / aggregate-flow measure) and
# fresh train+holdout validation before being reactivated, not just a
# parameter tweak. `min_usd_value` shown is what they were last validated
# at - do NOT assume a bigger number is a clean fix: re-running event_study
# at $500M+ flipped the sign on wallet_to_wallet and exchange_inflow
# entirely (different population of events), so a new threshold needs full
# re-validation, not substitution.
PARKED_HIGH_DUTY_CYCLE_RULES = [
    dict(
        name="exchange_outflow_large", category="exchange_outflow", sentiment_sign=None, horizon_hours=168,
        symbols={"BTCUSDT", "ETHUSDT"}, min_usd_value=WHALE_MIN_USD,
        rationale="$50M+ withdrawals from exchanges: +0.25%/wk BTC (t=2.6), +0.48%/wk ETH (t=3.3), n=3741. "
        "Duty cycle ~99.8% on BTC in the holdout window - essentially always on.",
    ),
    dict(
        name="macro_negative", category="macro", sentiment_sign="negative", horizon_hours=168,
        symbols={"ETHUSDT"}, min_usd_value=0,
        rationale="+1.53%/wk on ETH (t=2.66, n=369). Duty cycle ~97% on ETH in the holdout window - "
        "the earlier '+9.1% holdout' result for ETH was computed with the buggy summing scorer and "
        "needs to be treated as unconfirmed, not as this rule's validation.",
    ),
    dict(
        name="macro_negative_short", category="macro", sentiment_sign="negative", horizon_hours=24,
        symbols={"SOLUSDT"}, min_usd_value=0,
        rationale="+0.59% at 24h on SOL (t=2.82, n=373). Shorter window than macro_negative but not "
        "duty-cycle-checked yet - park alongside it pending recheck.",
    ),
    dict(
        name="exchange_inflow_large", category="exchange_inflow", sentiment_sign="negative", horizon_hours=168,
        symbols={"BTCUSDT", "SOLUSDT"}, min_usd_value=WHALE_MIN_USD,
        rationale="$50M+ deposits to exchanges: -0.34%/wk BTC (t=-4.0), -0.80%/wk SOL (t=-5.5). "
        "Duty cycle ~99.9% on BTC in the holdout window. UPDATE 2026-08-28: coin-matching (raw.coin "
        "== symbol's own coin, see event_study.py's COIN_SPECIFIC_CATEGORIES) helps unevenly - BTC "
        "stays saturated (73-92% duty cycle at any horizon, BTC transfers are just too frequent) but "
        "SOL's duty cycle drops to 48% at 168h with the signal intact (train t=-2.46, n=167; holdout "
        "same direction but n=15, below the n>=30 confidence bar) - closest thing to reactivatable so "
        "far, needs more holdout data before going back into RULES. ETH's low-duty-cycle horizon (24h, "
        "42%) has too weak a signal (t=-0.58 to -1.26); its strong horizons (72h/168h) are still 71-86%.",
    ),
    dict(
        name="whale_wallet_move_large", category="wallet_to_wallet", sentiment_sign=None, horizon_hours=168,
        symbols={"BTCUSDT", "ETHUSDT", "SOLUSDT"}, min_usd_value=WHALE_MIN_USD,
        rationale="$50M+ non-exchange whale-to-whale transfers: -0.65%/wk BTC (t=-6.4), -0.76%/wk ETH "
        "(t=-4.7), -1.81%/wk SOL (t=-10.8). Duty cycle ~99.98% on BTC in the holdout window - the "
        "worst offender. At $500M+ this flips POSITIVE (t=2-4) - a different signal, not validated.",
    ),
    dict(
        name="stablecoin_mint_large", category="stablecoin_mint", sentiment_sign="positive", horizon_hours=168,
        symbols={"BTCUSDT", "SOLUSDT"}, min_usd_value=WHALE_MIN_USD,
        rationale="$50M+ stablecoin mints: -0.50%/wk BTC (t=-3.4), -1.22%/wk SOL (t=-4.7). Duty cycle "
        "~99.8% on BTC in the holdout window - not yet rechecked at a rarer threshold.",
    ),
]


def load_events_with_returns(conn) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT e.id, e.ts, e.category, e.sentiment, e.raw
        FROM events e
        WHERE e.category IS NOT NULL
        """,
        conn,
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["sentiment_sign"] = np.sign(df["sentiment"].fillna(0)).map({-1.0: "negative", 0.0: "neutral", 1.0: "positive"})
    df["usd_value"] = df["raw"].apply(lambda r: (r or {}).get("usd_value") if isinstance(r, dict) else None)
    return df


def _matches(events: pd.DataFrame, rule: dict) -> pd.Series:
    mask = events["category"] == rule["category"]
    if rule["sentiment_sign"] is not None:
        mask &= events["sentiment_sign"] == rule["sentiment_sign"]
    if rule.get("min_usd_value"):
        mask &= events["usd_value"].fillna(0) >= rule["min_usd_value"]
    return mask


def generate_signals(events: pd.DataFrame, symbol: str) -> pd.DataFrame:
    signals = []
    for rule in RULES:
        if symbol not in rule["symbols"]:
            continue
        matched = events[_matches(events, rule)].copy()
        matched["rule"] = rule["name"]
        matched["hold_hours"] = rule["horizon_hours"]
        matched["action"] = "buy"
        matched["weight"] = rule["weight"]
        signals.append(matched[["ts", "rule", "hold_hours", "action", "weight"]])
    for rule in AVOID_RULES:
        if symbol not in rule["symbols"]:
            continue
        matched = events[_matches(events, rule)].copy()
        matched["rule"] = rule["name"]
        matched["hold_hours"] = rule["horizon_hours"]
        matched["action"] = "avoid"
        matched["weight"] = rule["weight"]
        signals.append(matched[["ts", "rule", "hold_hours", "action", "weight"]])
    return pd.concat(signals, ignore_index=True).sort_values("ts") if signals else pd.DataFrame()


def compute_score_series(signals: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.Series:
    """
    Turns discrete BUY/AVOID events into a continuous conviction score aligned
    to the ohlcv timeline: each *rule* contributes +/-weight for as long as
    ANY of its events has an active window, once - not once per overlapping
    event. Rules from different events/rules just add up.

    v2 (2026-08-28): the first version summed every individual event's weight
    while its window was open. For whale-alert-derived rules ($50M+ transfers,
    168h windows) that fire multiple times a day, dozens of windows overlap
    at any moment, so the summed score for a single rule alone could reach
    -300+ (measured on BTCUSDT's holdout: mean score -139, negative 99.95% of
    the time) - no BUY rule's weight of 2-3 could ever compete. The
    event-study each rule's weight is based on measured the effect of ONE
    event over its horizon, not N overlapping ones, so "is this rule
    currently active at all" is also the methodologically correct read, not
    just the practical fix.
    """
    if signals.empty:
        return pd.Series(0.0, index=ohlcv.index)
    total = pd.Series(0.0, index=ohlcv.index)
    for rule_name, group in signals.groupby("rule"):
        sign = 1 if (group["action"] == "buy").iloc[0] else -1
        weight = group["weight"].iloc[0]
        starts = pd.DataFrame({"ts": group["ts"], "delta": 1})
        ends = pd.DataFrame({"ts": group["ts"] + pd.to_timedelta(group["hold_hours"], unit="h"), "delta": -1})
        changes = pd.concat([starts, ends], ignore_index=True).groupby("ts", as_index=False)["delta"].sum()
        changes = changes.sort_values("ts")
        changes["active_count"] = changes["delta"].cumsum()
        merged = pd.merge_asof(ohlcv[["ts"]], changes[["ts", "active_count"]], on="ts", direction="backward")
        is_active = (merged["active_count"].fillna(0) > 0).astype(float)
        total = total + sign * weight * is_active.values
    return total


def compute_buy_rule_count_series(signals: pd.DataFrame, ohlcv: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Shared helper (2026-09-15): returns (buy_active_count, avoid_active)
    per-hour, aligned to ohlcv - how many distinct BUY rules are currently
    active, and whether any AVOID rule is active. Factored out of
    compute_confluence_score_series so a second, DIFFERENT use of the same
    per-rule interval counting (entry-only confluence filtering, see
    project_confluence_gate_study memory) doesn't duplicate this logic."""
    if signals.empty:
        return pd.Series(0, index=ohlcv.index), pd.Series(False, index=ohlcv.index)
    buy_active_count = pd.Series(0, index=ohlcv.index)
    avoid_active = pd.Series(False, index=ohlcv.index)
    for rule_name, group in signals.groupby("rule"):
        is_buy = (group["action"] == "buy").iloc[0]
        starts = pd.DataFrame({"ts": group["ts"], "delta": 1})
        ends = pd.DataFrame({"ts": group["ts"] + pd.to_timedelta(group["hold_hours"], unit="h"), "delta": -1})
        changes = pd.concat([starts, ends], ignore_index=True).groupby("ts", as_index=False)["delta"].sum()
        changes = changes.sort_values("ts")
        changes["active_count"] = changes["delta"].cumsum()
        merged = pd.merge_asof(ohlcv[["ts"]], changes[["ts", "active_count"]], on="ts", direction="backward")
        is_active = (merged["active_count"].fillna(0) > 0).values
        if is_buy:
            buy_active_count = buy_active_count + is_active.astype(int)
        else:
            avoid_active = avoid_active | is_active
    return buy_active_count, avoid_active


def compute_confluence_score_series(signals: pd.DataFrame, ohlcv: pd.DataFrame, min_buy_rules: int = 2) -> pd.Series:
    """
    Alternative to compute_score_series's additive scoring (2026-09-15,
    see project_confluence_gate_study memory). compute_score_series is an
    OR-gate: any single active BUY rule makes score>0 - Opus flagged this
    as structurally weak once more than one rule can fire (adding rules
    just grows time-in-market rather than creating real confluence). This
    instead requires at least `min_buy_rules` distinct BUY rules
    simultaneously active; any active AVOID rule still vetoes entirely
    regardless of buy count, matching the existing AVOID-rule veto
    behavior in compute_score_series.

    Returns a score-like series (>0 = confluent long, <=0 = flat/exit) -
    drop-in compatible with backtest()/compute_candidate_segments() via
    their `score_series` override parameter, so all the same verified
    segment/stop-loss/trend-filter/cooldown logic is reused unchanged.

    NOTE: this requires confluence to hold for the ENTIRE segment, which
    was found (2026-09-15) to truncate trade duration to the shortest
    overlapping rule's window and was REJECTED - see
    compute_candidate_segments' `min_buy_rules_at_entry` parameter for the
    entry-only variant that was tried next instead."""
    if signals.empty:
        return pd.Series(-1.0, index=ohlcv.index)
    buy_active_count, avoid_active = compute_buy_rule_count_series(signals, ohlcv)
    confluent = (buy_active_count >= min_buy_rules) & (~avoid_active)
    return pd.Series(np.where(confluent, buy_active_count.astype(float), -1.0), index=ohlcv.index)


def backtest(symbol: str, ohlcv: pd.DataFrame, signals: pd.DataFrame, capital: float, fee: float = SPOT_TAKER_FEE,
             stop_loss: float | None = None, trend_filter_sma: int | None = None, cooldown_after_losses: int | None = None,
             score_series: pd.Series | None = None):
    """
    stop_loss: if set (e.g. -0.10 for -10%), force-exit a position the first
    time its close drops that far below entry, instead of waiting for the
    score to turn non-positive naturally. None = no stop-loss (original
    behaviour) - trades can run for weeks and draw down 30-50% before the
    score-based exit fires, see project_strategy_scoring_bug memory.

    trend_filter_sma: if set (hours, e.g. 720 for a 30-day SMA), skip a BUY
    entry entirely when price is below its trailing SMA of that length -
    caps per-trade loss doesn't help against buying into a confirmed
    downtrend; measured on SOL: below-SMA trades averaged -5.2% (33% win
    rate) vs +2.8% (42% win rate) above it.

    cooldown_after_losses: if set (an int N), skip new entries for the next
    `cooldown_hours` after N consecutive losing trades close - a per-trade
    stop-loss caps single-trade risk but not a losing *streak*, which is
    what actually produced SOL's -40.9% cumulative drawdown even with the
    stop-loss active (see project_practical_usability_verdict memory).
    """
    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)
    ohlcv["score"] = score_series.reset_index(drop=True) if score_series is not None else compute_score_series(signals, ohlcv)
    ohlcv["is_long"] = ohlcv["score"] > 0
    ohlcv["seg_id"] = ohlcv["is_long"].ne(ohlcv["is_long"].shift(fill_value=False)).cumsum()
    if trend_filter_sma:
        ohlcv["sma"] = ohlcv["close"].rolling(trend_filter_sma, min_periods=trend_filter_sma // 3).mean()

    trades = []
    consecutive_losses = 0
    cooldown_until_ts = None
    for seg_id, seg in ohlcv[ohlcv["is_long"]].groupby("seg_id"):
        entry_idx = seg.index[0]
        natural_exit_idx = min(seg.index[-1] + 1, len(ohlcv) - 1)  # first candle after the score turns non-positive
        entry_price = ohlcv["close"].iloc[entry_idx]
        entry_ts_candidate = ohlcv["ts"].iloc[entry_idx]
        if cooldown_until_ts is not None and entry_ts_candidate < cooldown_until_ts:
            continue  # still cooling down after a losing streak
        if trend_filter_sma:
            sma = ohlcv["sma"].iloc[entry_idx]
            if pd.notna(sma) and entry_price < sma:
                continue  # price below its own trend - skip this entry entirely
        exit_idx = natural_exit_idx
        if stop_loss is not None:
            path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
            hit = path[path / entry_price - 1 <= stop_loss]
            if not hit.empty:
                exit_idx = hit.index[0]
        exit_price = ohlcv["close"].iloc[exit_idx]
        entry_ts, exit_ts = ohlcv["ts"].iloc[entry_idx], ohlcv["ts"].iloc[exit_idx]
        gross_ret = (exit_price - entry_price) / entry_price
        # buy pays fee on the way in, sell pays fee on the way out (round-trip = ~2x fee)
        net_ret = (exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1
        # worst unrealized mark-to-market drawdown DURING the hold, not just at exit - a trade
        # can dip much lower than its close-to-close return before eventually recovering.
        path = ohlcv["close"].iloc[entry_idx:exit_idx + 1]
        worst_intratrade_dd = (path / entry_price - 1).min()
        trades.append(dict(
            entry_ts=entry_ts, exit_ts=exit_ts, entry_price=entry_price, exit_price=exit_price,
            hours_held=(exit_ts - entry_ts).total_seconds() / 3600, gross_ret=gross_ret, ret=net_ret,
            worst_intratrade_dd=worst_intratrade_dd,
        ))
        if cooldown_after_losses:
            if net_ret < 0:
                consecutive_losses += 1
                if consecutive_losses >= cooldown_after_losses:
                    cooldown_until_ts = exit_ts + pd.Timedelta(hours=168)
                    consecutive_losses = 0
            else:
                consecutive_losses = 0

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {}

    equity_curve = [capital]
    for r in trades_df["ret"]:
        equity_curve.append(equity_curve[-1] * (1 + r))
    equity = equity_curve[-1]
    total_return = equity / capital - 1
    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    trade_level_max_drawdown = ((equity_series - running_peak) / running_peak).min()

    start_price = ohlcv["close"].iloc[0]
    end_price = ohlcv["close"].iloc[-1]
    buy_hold_return = end_price / start_price - 1

    stats = dict(
        symbol=symbol,
        n_trades=len(trades_df),
        pct_time_in_market=ohlcv["is_long"].mean(),
        win_rate=(trades_df["ret"] > 0).mean(),
        avg_hours_held=trades_df["hours_held"].mean(),
        max_hours_held=trades_df["hours_held"].max(),
        avg_trade_return_gross=trades_df["gross_ret"].mean(),
        avg_trade_return_net=trades_df["ret"].mean(),
        worst_single_trade_ret=trades_df["ret"].min(),
        worst_intratrade_drawdown=trades_df["worst_intratrade_dd"].min(),
        trade_level_max_drawdown=trade_level_max_drawdown,
        fee_per_trade=fee * 2,
        strategy_total_return_net=total_return,
        buy_hold_return=buy_hold_return,
        final_equity=equity,
    )
    return trades_df, stats


def compute_candidate_segments(symbol: str, ohlcv: pd.DataFrame, signals: pd.DataFrame, fee: float = SPOT_TAKER_FEE,
                                stop_loss: float | None = None, trend_filter_sma: int | None = None,
                                score_series: pd.Series | None = None,
                                min_buy_rules_at_entry: int | None = None,
                                buy_count_series: pd.Series | None = None) -> list[dict]:
    """Per-symbol segment precomputation for simulate_portfolio(): mirrors
    backtest() exactly, EXCEPT it does not apply cooldown here (cooldown
    depends on cross-symbol chronological ordering once a shared capacity
    cap is involved, so it's applied afterward in simulate_portfolio) -
    returns each segment's trend-filter eligibility and predetermined
    entry/exit/ret, unconditional on cooldown/capacity.

    score_series: optional precomputed score (e.g. from
    compute_confluence_score_series) to use INSTEAD of the default
    additive compute_score_series - lets alternative scoring schemes reuse
    all of this verified segment logic unchanged (added 2026-09-15, see
    project_confluence_gate_study memory).

    min_buy_rules_at_entry / buy_count_series: optional ENTRY-ONLY
    confluence filter (added 2026-09-15, the follow-up to
    compute_confluence_score_series after that full-segment version was
    REJECTED - it truncated trade duration to the shortest overlapping
    rule's window). This keeps segment boundaries from the BASELINE score
    (so long "lucky regime" trades stay possible) but additionally
    requires buy_count_series to be >= min_buy_rules_at_entry AT THE
    SEGMENT'S OWN START - reuses the same permanent-skip-this-segment
    semantics as the trend filter (trend_ok=False), not a new mechanism.

    Verified (2026-09-13) to reproduce backtest()'s own trades exactly when
    fed into simulate_portfolio() with max_positions=None - see
    project_four_hypotheses_2026-09-13 memory for the two bugs this caught
    in an earlier, less careful version of this same logic."""
    ohlcv = ohlcv.sort_values("ts").reset_index(drop=True)
    ohlcv["score"] = score_series.reset_index(drop=True) if score_series is not None else compute_score_series(signals, ohlcv)
    ohlcv["is_long"] = ohlcv["score"] > 0
    ohlcv["seg_id"] = ohlcv["is_long"].ne(ohlcv["is_long"].shift(fill_value=False)).cumsum()
    if trend_filter_sma:
        ohlcv["sma"] = ohlcv["close"].rolling(trend_filter_sma, min_periods=trend_filter_sma // 3).mean()
    if min_buy_rules_at_entry is not None:
        ohlcv["buy_count"] = buy_count_series.reset_index(drop=True).values

    segments = []
    for seg_id, seg in ohlcv[ohlcv["is_long"]].groupby("seg_id"):
        entry_idx = seg.index[0]
        natural_exit_idx = min(seg.index[-1] + 1, len(ohlcv) - 1)
        entry_price = ohlcv["close"].iloc[entry_idx]
        entry_ts = ohlcv["ts"].iloc[entry_idx]
        entry_score = ohlcv["score"].iloc[entry_idx]
        trend_ok = True
        if trend_filter_sma:
            sma = ohlcv["sma"].iloc[entry_idx]
            trend_ok = not (pd.notna(sma) and entry_price < sma)
        if min_buy_rules_at_entry is not None:
            trend_ok = trend_ok and (ohlcv["buy_count"].iloc[entry_idx] >= min_buy_rules_at_entry)

        exit_idx = natural_exit_idx
        if stop_loss is not None:
            path = ohlcv["close"].iloc[entry_idx:natural_exit_idx + 1]
            hit = path[path / entry_price - 1 <= stop_loss]
            if not hit.empty:
                exit_idx = hit.index[0]
        exit_price = ohlcv["close"].iloc[exit_idx]
        exit_ts = ohlcv["ts"].iloc[exit_idx]
        net_ret = (exit_price * (1 - fee)) / (entry_price * (1 + fee)) - 1

        sma_at_entry = ohlcv["sma"].iloc[entry_idx] if trend_filter_sma and pd.notna(ohlcv["sma"].iloc[entry_idx]) else None
        trend_strength = (entry_price / sma_at_entry - 1) if sma_at_entry else None

        segments.append(dict(symbol=symbol, entry_ts=entry_ts, entry_score=entry_score, exit_ts=exit_ts,
                              trend_ok=trend_ok, ret=net_ret, entry_price=entry_price,
                              trend_strength=trend_strength))
    return segments


def simulate_portfolio(all_segments: list[dict], max_positions: int | None,
                        cooldown_after_losses: int | None = None, cooldown_hours: int = 168,
                        tiebreak_key: str = "entry_score"):
    """Merges every symbol's candidate segments (from compute_candidate_segments)
    in GLOBAL chronological order, applying each symbol's OWN independent
    cooldown state plus a shared capacity cap across ALL symbols - the ONLY
    new permanent-skip condition a cap introduces beyond what backtest()
    already does per symbol. Reproduces backtest() exactly per-symbol when
    max_positions=None.

    Ties at the same entry_ts (routine here, since fgi_greed/DVOL are
    market-wide and often fire many symbols at once - see the "6 of 7
    entered simultaneously" first live run) are broken by the HIGHEST
    value of `tiebreak_key` (default "entry_score", matching what's live
    in testnet_trader.py). This is a deliberate design choice for fairness,
    NOT the exact tie-break the original portfolio_exposure_study.py
    measured (there, ties fell out of symbol-list processing order and
    were flagged in project_four_hypotheses_2026-09-13 memory as an
    artifact - BTC winning most ties was NOT proof BTC is objectively the
    better trade). `tiebreak_key` is pluggable (2026-09-15, see
    project_cap_tiebreak_study memory) specifically because "entry_score"
    is itself a weak tiebreak - the 3 active rules are all global/
    market-wide, so score differences between simultaneously-firing
    symbols mostly reflect which subset of rules happens to apply to that
    symbol, not a real per-symbol opportunity signal. A segment missing
    the given key (None) sorts last, never wins a contested slot."""
    def _tiebreak_value(seg):
        v = seg.get(tiebreak_key)
        return v if v is not None else float("-inf")
    all_segments = sorted(all_segments, key=lambda s: (s["entry_ts"], -_tiebreak_value(s)))
    cooldown_until = {}
    consecutive_losses = {}
    open_trades = []  # list of (exit_ts, symbol) currently occupying a slot
    realized = []
    rejection_log = []

    for seg in all_segments:
        symbol, entry_ts = seg["symbol"], seg["entry_ts"]
        open_trades = [(ex, sym) for ex, sym in open_trades if ex > entry_ts]

        if not seg["trend_ok"]:
            continue
        cd = cooldown_until.get(symbol)
        if cd is not None and entry_ts < cd:
            continue
        if max_positions is not None and len(open_trades) >= max_positions:
            rejection_log.append(dict(symbol=symbol, entry_ts=entry_ts))
            continue

        realized.append(seg)
        open_trades.append((seg["exit_ts"], symbol))
        if cooldown_after_losses:
            if seg["ret"] < 0:
                consecutive_losses[symbol] = consecutive_losses.get(symbol, 0) + 1
                if consecutive_losses[symbol] >= cooldown_after_losses:
                    cooldown_until[symbol] = seg["exit_ts"] + pd.Timedelta(hours=cooldown_hours)
                    consecutive_losses[symbol] = 0
            else:
                consecutive_losses[symbol] = 0

    return pd.DataFrame(realized), rejection_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--fee", type=float, default=SPOT_TAKER_FEE, help="One-side spot fee as a fraction, e.g. 0.001 = 0.1%%")
    parser.add_argument("--start-date", type=str, default=None, help="Only use data from this date on (YYYY-MM-DD) - for out-of-sample checks")
    parser.add_argument("--end-date", type=str, default=None, help="Only use data up to this date (YYYY-MM-DD) - for out-of-sample checks")
    parser.add_argument("--stop-loss", type=str, default="-0.08", help="Force-exit if price drops this fraction below entry, e.g. -0.08 for -8%% "
                         "(default -0.08 - tames trade_level_max_drawdown from 37-49%% down to <12%% in the holdout window, see README.md). "
                         "Pass 'none' to disable and let trades run until the score-based exit fires naturally.")
    parser.add_argument("--trend-filter-sma", type=int, default=720, help="Skip BUY entries below their trailing SMA of this many hours "
                         "(default 720 = 30 days - improves win rate/drawdown on SOL and ETH, no effect on BTC in the holdout window, see "
                         "README.md). Pass 0 to disable.")
    parser.add_argument("--cooldown-after-losses", type=int, default=3, help="Skip new entries for 168h after this many consecutive losing "
                         "trades (default 3 - caps losing-streak risk that a per-trade stop-loss alone doesn't catch). Pass 0 to disable.")
    parser.add_argument("--max-positions", type=int, default=None, help="Cap simultaneous open positions across ALL requested symbols "
                         "(shared resource, highest-score entry candidate wins when more want in than there's room for). Validated "
                         "2026-09-13: cap=3 or cap=4 gives the best risk-adjusted (return/drawdown) ratio on both train and holdout, "
                         "see README.md 'Портфельный лимит экспозиции'. Only meaningful with 2+ symbols. Omit (None) for the original "
                         "per-symbol-independent behaviour (no cap).")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    stop_loss = None if args.stop_loss.lower() == "none" else float(args.stop_loss)

    conn = get_connection()
    events = load_events_with_returns(conn)
    if args.start_date:
        events = events[events["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        events = events[events["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]

    print("Active BUY rules:")
    for rule in RULES:
        print(f"  {rule['name']} [{','.join(sorted(rule['symbols']))}]: hold {rule['horizon_hours']}h - {rule['rationale']}")
    print("Active AVOID rules:")
    for rule in AVOID_RULES:
        print(f"  {rule['name']} [{','.join(sorted(rule['symbols']))}]: window {rule['horizon_hours']}h - {rule['rationale']}")

    all_trades = []
    for symbol in symbols:
        signals = generate_signals(events, symbol)
        ohlcv = pd.read_sql(
            "SELECT ts, close FROM ohlcv WHERE symbol = %(symbol)s ORDER BY ts",
            conn,
            params={"symbol": symbol},
        )
        ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
        if args.start_date:
            ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
        if args.end_date:
            ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
        trades_df, stats = backtest(symbol, ohlcv, signals, args.capital, fee=args.fee, stop_loss=stop_loss,
                                     trend_filter_sma=args.trend_filter_sma or None,
                                     cooldown_after_losses=args.cooldown_after_losses or None)
        if not stats:
            print(f"\n{symbol}: no trades generated ({len(signals)} signals)")
            continue
        print(f"\n=== {symbol} backtest ({len(signals)} signals) ===")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        all_trades.append(trades_df.assign(symbol=symbol))

    if args.max_positions and len(symbols) > 1:
        all_segments = []
        for symbol in symbols:
            signals = generate_signals(events, symbol)
            ohlcv = pd.read_sql("SELECT ts, close FROM ohlcv WHERE symbol = %(symbol)s ORDER BY ts",
                                 conn, params={"symbol": symbol})
            ohlcv["ts"] = pd.to_datetime(ohlcv["ts"], utc=True)
            if args.start_date:
                ohlcv = ohlcv[ohlcv["ts"] >= pd.Timestamp(args.start_date, tz="UTC")]
            if args.end_date:
                ohlcv = ohlcv[ohlcv["ts"] <= pd.Timestamp(args.end_date, tz="UTC")]
            all_segments.extend(compute_candidate_segments(symbol, ohlcv, signals, fee=args.fee,
                                                             stop_loss=stop_loss, trend_filter_sma=args.trend_filter_sma or None))
        trades_df, rejection_log = simulate_portfolio(all_segments, args.max_positions,
                                                        cooldown_after_losses=args.cooldown_after_losses or None)
        print(f"\n=== Portfolio simulation (max_positions={args.max_positions}, {len(symbols)} symbols) ===")
        if trades_df.empty:
            print("  no trades generated")
        else:
            per_slot_capital = args.capital / len(symbols)
            trades_df = trades_df.sort_values("exit_ts")
            equity_curve = [per_slot_capital * args.max_positions]
            for r in trades_df["ret"]:
                equity_curve.append(equity_curve[-1] + per_slot_capital * r)
            equity_series = pd.Series(equity_curve)
            total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
            running_peak = equity_series.cummax()
            max_dd = ((equity_series - running_peak) / running_peak).min()
            print(f"  n_trades: {len(trades_df)}")
            print(f"  win_rate: {(trades_df['ret'] > 0).mean():.4f}")
            print(f"  total_return: {total_return:.4f}")
            print(f"  max_drawdown: {max_dd:.4f}")
            print(f"  rejected_entries (crowded out): {len(rejection_log)}")
    conn.close()

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("strategy_trades.csv", index=False)
        print("\nTrade log written to strategy_trades.csv")


if __name__ == "__main__":
    main()
