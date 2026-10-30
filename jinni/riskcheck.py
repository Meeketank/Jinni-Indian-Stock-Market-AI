"""Pre-trade sizing and risk gate (after day-trader-skill's trader-budget / trader-risk).

Sizing finds the whole shares that fit inside every limit and says which
limit decided. A stop too far away for the max loss is a refusal, not a
smaller order. The gate returns PASS / WARN / BLOCK per rule with the reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_RULES = {
    "max_loss_per_trade": 10_000.0,   # ₹ lost if the stop fills
    "max_position_pct": 0.10,         # of total capital (SEBI MF single-stock cap as reference)
    "max_sector_pct": 0.35,           # sector weight after the trade
    "min_reward_risk": 1.5,
    "cash_buffer_pct": 0.05,          # keep this share of capital in cash
    "max_open_trades": 8,
    "min_probability": 0.0,           # block below this blended P(target first)
    "require_stop": True,
}


@dataclass
class Sizing:
    qty: int
    deciding_limit: str
    limits: dict
    refused: str | None = None


def size_position(entry: float, stop: float | None, capital: float, cash: float, rules: dict) -> Sizing:
    lim = {}
    if stop:
        risk = entry - stop
        lim["Max loss per trade"] = int(rules["max_loss_per_trade"] // risk) if risk > 0 else 0
    lim["Position cap"] = int(rules["max_position_pct"] * capital // entry)
    lim["Cash after buffer"] = int(max(cash - rules["cash_buffer_pct"] * capital, 0) // entry)
    decider = min(lim, key=lim.get)
    qty = lim[decider]
    refused = None
    if qty < 1:
        refused = {
            "Max loss per trade": f"Even 1 share risks ₹{entry - stop:,.2f}, above your ₹{rules['max_loss_per_trade']:,.0f} "
                                  "max loss. Use a closer stop or skip the trade.",
            "Position cap": "One share is larger than your position cap.",
            "Cash after buffer": "Not enough free cash after your buffer.",
        }[decider]
    return Sizing(max(qty, 0), decider, lim, refused)


@dataclass
class GateResult:
    checks: list = field(default_factory=list)  # (status, rule, detail)

    @property
    def status(self) -> str:
        s = [c[0] for c in self.checks]
        return "BLOCK" if "BLOCK" in s else "WARN" if "WARN" in s else "PASS"


def gate(entry: float, stop: float | None, target: float, qty: int, sector: str, prob: float,
         capital: float, cash: float, sector_values: dict, open_trades: int, rules: dict,
         events: list[str] | None = None) -> GateResult:
    g = GateResult()
    add = g.checks.append
    if qty < 1:
        add(("BLOCK", "Size", "Position size is zero."))
        return g
    if rules["require_stop"] and not stop:
        add(("BLOCK", "Stop", "A stop is required by your rules."))
    if stop:
        loss = qty * (entry - stop)
        ok = loss <= rules["max_loss_per_trade"] + 1e-6
        add(("PASS" if ok else "BLOCK", "Max loss", f"₹{loss:,.0f} at the stop (limit ₹{rules['max_loss_per_trade']:,.0f})"))
        rr = (target - entry) / (entry - stop)
        add(("PASS" if rr >= rules["min_reward_risk"] else "BLOCK", "Reward:risk",
             f"{rr:.2f} (min {rules['min_reward_risk']:.2f})"))
    value = qty * entry
    pos = value / capital
    add(("PASS" if pos <= rules["max_position_pct"] + 1e-6 else "BLOCK", "Position size",
         f"{pos:.1%} of capital (cap {rules['max_position_pct']:.0%})"))
    after = cash - value
    buf = rules["cash_buffer_pct"] * capital
    add(("PASS" if after >= buf else "BLOCK", "Cash", f"₹{after:,.0f} left (buffer ₹{buf:,.0f})"))
    sec_after = (sector_values.get(sector, 0) + value) / capital
    add(("PASS" if sec_after <= rules["max_sector_pct"] + 1e-6 else "WARN", "Sector",
         f"{sector} becomes {sec_after:.1%} of capital (limit {rules['max_sector_pct']:.0%})"))
    add(("PASS" if open_trades < rules["max_open_trades"] else "BLOCK", "Open trades",
         f"{open_trades} open (max {rules['max_open_trades']})"))
    if rules.get("min_probability", 0) > 0:
        add(("PASS" if prob >= rules["min_probability"] else "BLOCK", "Probability",
             f"{prob:.0%} (min {rules['min_probability']:.0%})"))
    for e in events or []:
        add(("WARN", "Event", e))
    return g


def portfolio_impact(rets: pd.DataFrame, weights_value: pd.Series, ticker: str, add_value: float,
                     conf: float = 0.95) -> dict:
    """1-day historical VaR and volatility of the book before and after the trade."""
    from .risk import historical_var, portfolio_returns

    before = weights_value[weights_value > 0]
    after = before.copy()
    after[ticker] = after.get(ticker, 0.0) + add_value
    out = {}
    for name, v in (("Before", before), ("After", after)):
        cols = [c for c in v.index if c in rets.columns]
        if not cols:
            out[name] = {"Value": float(v.sum()), "VaR %": np.nan, "VaR ₹": np.nan, "Volatility": np.nan}
            continue
        w = v[cols] / v[cols].sum()
        pr = portfolio_returns(rets[cols], w)
        var, _ = historical_var(pr, conf)
        out[name] = {"Value": float(v.sum()), "VaR %": var, "VaR ₹": var * float(v.sum()),
                     "Volatility": float(pr.std() * np.sqrt(252))}
    return out
