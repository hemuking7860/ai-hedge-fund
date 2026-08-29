"""Swing risk layer — sizing and gating that upstream's two limits cannot do.

`risk/limits.py` enforces exactly two things: a per-ticker weight cap and a
gross cap. That is a long-only allocator's risk model. A swing desk needs
four more, and all four depend on facts the weight vector alone does not
carry — where the stop is, what sector the name is in, and whether a binary
event lands inside the holding period:

    size_from_risk       weight implied by a stop distance and a risk budget
    portfolio_heat       summed open risk, capped
    sector_caps          concentration, because swing setups cluster
    event_gate           refuse entries in front of a known binary

The ordering is deliberate and each stage only ever SHRINKS exposure, which
preserves the upstream invariant that risk never increases a position
("conviction requests, risk disposes"). Freed exposure stays in cash rather
than being redistributed.

Everything here is pure arithmetic over explicit inputs. No I/O, no clock.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hedge_fund.config.swing_trading_config import SwingProfile


class SwingClamp(BaseModel):
    """One swing limit firing — the audit trail, mirroring ClampEvent."""

    limit: str
    ticker: str | None = None
    before: float
    after: float
    detail: str | None = None


class SwingRiskResult(BaseModel):
    """Post-swing-risk weights plus every clamp that fired and the heat used."""

    weights: dict[str, float]
    clamps: list[SwingClamp] = Field(default_factory=list)
    heat_used: float = 0.0
    stops: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def stop_distance(entry: float, atr_value: float, profile: SwingProfile) -> float:
    """Initial stop distance in price units: `atr_stop_multiple` ATRs.

    Returns 0.0 when ATR is unusable, which callers must treat as "cannot
    size this trade" rather than "zero risk".
    """
    if entry <= 0 or atr_value <= 0:
        return 0.0
    return atr_value * profile.atr_stop_multiple


def size_from_risk(
    entry: float,
    atr_value: float,
    profile: SwingProfile,
    conviction: float = 1.0,
) -> float:
    """Position weight that risks `risk_per_trade` of equity to the stop.

    The identity a swing desk actually sizes on:

        weight = risk_budget / (stop_distance / entry)

    i.e. a tight stop earns a big position and a wide stop earns a small
    one, so every trade contributes the SAME dollar risk regardless of how
    volatile the name is. That is the whole point of volatility-aware
    sizing, and it is what makes portfolio heat additive.

    Conviction scales the risk budget, not the stop — a half-conviction
    trade risks half as much, it does not move its stop closer.

    Returns 0.0 if the stop is uncomputable (no ATR) or conviction is
    non-positive: refusing to size is the safe failure.
    """
    if conviction <= 0:
        return 0.0
    distance = stop_distance(entry, atr_value, profile)
    if distance <= 0:
        return 0.0

    stop_fraction = distance / entry
    budget = profile.risk_per_trade * min(conviction, 1.0)
    weight = budget / stop_fraction

    return min(weight, profile.max_position_size)


def trade_risk(weight: float, entry: float, stop: float) -> float:
    """Equity fraction at risk for an open position — its contribution to heat."""
    if entry <= 0 or stop <= 0 or stop >= entry:
        return 0.0
    return abs(weight) * (entry - stop) / entry


# ---------------------------------------------------------------------------
# Portfolio-level gates
# ---------------------------------------------------------------------------

def apply_portfolio_heat(
    weights: dict[str, float],
    stops: dict[str, float],
    entries: dict[str, float],
    profile: SwingProfile,
) -> tuple[dict[str, float], list[SwingClamp], float]:
    """Scale the book down until summed open risk fits the heat budget.

    Heat is the sum of each position's distance-to-stop risk. If the book
    is over budget, every position is scaled by the same factor — this
    preserves the relative shape the alpha models asked for while bringing
    total risk inside the cap.
    """
    risks = {
        t: trade_risk(w, entries.get(t, 0.0), stops.get(t, 0.0))
        for t, w in weights.items()
    }
    heat = sum(risks.values())
    if heat <= profile.max_portfolio_heat or heat <= 0:
        return dict(weights), [], round(heat, 6)

    scale = profile.max_portfolio_heat / heat
    scaled = {t: w * scale for t, w in weights.items()}
    clamp = SwingClamp(
        limit="max_portfolio_heat",
        before=round(heat, 6),
        after=profile.max_portfolio_heat,
        detail=f"scaled every position by {scale:.3f}",
    )
    return scaled, [clamp], profile.max_portfolio_heat


def apply_sector_caps(
    weights: dict[str, float],
    sectors: dict[str, str],
    profile: SwingProfile,
) -> tuple[dict[str, float], list[SwingClamp]]:
    """Keep at most `max_positions_per_sector` names per sector.

    When a sector is over-subscribed the weakest positions are dropped, not
    trimmed: a swing book wants full-size expressions of its best setups,
    not a smeared half-position across everything in the sector. Ranking is
    by |weight| with the ticker as a deterministic tiebreak.

    Tickers with no known sector are grouped under "UNKNOWN" and capped the
    same way — an unknown sector is a concentration risk we cannot see, and
    silently exempting it would be the wrong default.
    """
    by_sector: dict[str, list[str]] = {}
    for ticker, weight in weights.items():
        if weight == 0:
            continue
        by_sector.setdefault(sectors.get(ticker, "UNKNOWN"), []).append(ticker)

    out = dict(weights)
    clamps: list[SwingClamp] = []
    for sector, tickers in sorted(by_sector.items()):
        if len(tickers) <= profile.max_positions_per_sector:
            continue
        ranked = sorted(tickers, key=lambda t: (-abs(weights[t]), t))
        for ticker in ranked[profile.max_positions_per_sector:]:
            clamps.append(SwingClamp(
                limit="max_positions_per_sector",
                ticker=ticker,
                before=weights[ticker],
                after=0.0,
                detail=(
                    f"{sector} already holds {profile.max_positions_per_sector} "
                    "higher-conviction names"
                ),
            ))
            out[ticker] = 0.0

    return out, clamps


def apply_event_gate(
    weights: dict[str, float],
    blocked: dict[str, bool],
    held: dict[str, float],
    profile: SwingProfile,
) -> tuple[dict[str, float], list[SwingClamp]]:
    """Refuse NEW exposure into a name with an imminent binary event.

    Deliberately asymmetric: this blocks *opening* or *increasing* a
    position, it does not force-close one. Closing in front of earnings is
    the exit layer's job (`exit_before_earnings_days`), and conflating the
    two would make a position that is already being wound down look like a
    fresh entry veto.

    A no-op when the profile does not avoid binary events.
    """
    if not profile.avoid_binary_events:
        return dict(weights), []

    out = dict(weights)
    clamps: list[SwingClamp] = []
    for ticker, weight in weights.items():
        if not blocked.get(ticker):
            continue
        current = held.get(ticker, 0.0)
        # Allow holding or reducing; forbid adding.
        capped = current if abs(weight) > abs(current) else weight
        if capped != weight:
            clamps.append(SwingClamp(
                limit="event_gate",
                ticker=ticker,
                before=weight,
                after=capped,
                detail="binary event inside the swing horizon — no new exposure",
            ))
            out[ticker] = capped

    return out, clamps


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def apply_swing_risk(
    weights: dict[str, float],
    profile: SwingProfile,
    *,
    entries: dict[str, float],
    atrs: dict[str, float],
    sectors: dict[str, str] | None = None,
    blocked_by_event: dict[str, bool] | None = None,
    held_weights: dict[str, float] | None = None,
) -> SwingRiskResult:
    """Run the full swing risk stack over target *weights*.

    Order matters:
      1. event gate    — a blocked name should never consume heat budget
      2. sector caps   — drop over-concentration before measuring risk
      3. heat cap      — scale what survives into the risk budget
    Each stage only shrinks, so later stages cannot re-violate earlier ones.

    Stops are derived per ticker from ATR and returned alongside, because
    the exit layer needs the SAME stop the sizing assumed — recomputing it
    later from different data is how a backtest quietly stops being
    self-consistent.
    """
    sectors = sectors or {}
    blocked_by_event = blocked_by_event or {}
    held_weights = held_weights or {}

    clamps: list[SwingClamp] = []

    gated, event_clamps = apply_event_gate(
        weights, blocked_by_event, held_weights, profile
    )
    clamps.extend(event_clamps)

    capped, sector_clamps = apply_sector_caps(gated, sectors, profile)
    clamps.extend(sector_clamps)

    stops = {
        t: round(entries[t] - stop_distance(entries[t], atrs.get(t, 0.0), profile), 6)
        for t in capped
        if entries.get(t, 0.0) > 0 and stop_distance(entries[t], atrs.get(t, 0.0), profile) > 0
    }

    final, heat_clamps, heat = apply_portfolio_heat(capped, stops, entries, profile)
    clamps.extend(heat_clamps)

    return SwingRiskResult(
        weights={t: round(w, 8) for t, w in final.items()},
        clamps=clamps,
        heat_used=heat,
        stops=stops,
    )
