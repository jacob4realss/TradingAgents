"""Carry out a TradingAgents decision on an Alpaca paper (demo) account.

The agents decide; this module turns the Portfolio Manager's rating into a
whole-share market order and sends it to Alpaca's paper trading API. Paper only
by construction: the trading host is fixed to paper-api.alpaca.markets and any
other host is refused, so no setting can send an order to a live account.

Sizing is deliberately simple and long-only (the rating carries no size of its
own, so the rules below supply one):

- Buy:         raise the position to ``max_position_pct`` of account equity
- Overweight:  raise it to ``overweight_fraction`` of that target
- Hold:        no order
- Underweight: sell ``underweight_sell_fraction`` of the shares held
- Sell:        sell every share held; nothing is sold short
- REVIEW:      no order (the decision had no readable rating)

Buys are capped by the cash on hand (never margin) and by ``max_order_value``.

Keys come from APCA_API_KEY_ID and APCA_API_SECRET_KEY, Alpaca's own names for
them. With neither set, requests go out unsigned, for a network proxy that adds
the credentials itself.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from tradingagents.agents.rating import RATING_REVIEW
from tradingagents.portfolio import PortfolioContext, Position

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
MARKET_DATA_URL = "https://data.alpaca.markets"

KEY_ID_ENV = "APCA_API_KEY_ID"
SECRET_KEY_ENV = "APCA_API_SECRET_KEY"
ORDER_LOG_ENV = "TRADINGAGENTS_ALPACA_ORDER_LOG"


class AlpacaError(RuntimeError):
    """Alpaca refused a request, or answered with something unusable."""


class AlpacaClient:
    """The few Alpaca Trading and Market Data calls a paper trade needs."""

    def __init__(
        self,
        key_id: str | None = None,
        secret_key: str | None = None,
        *,
        base_url: str = PAPER_TRADING_URL,
        data_url: str = MARKET_DATA_URL,
        session: requests.Session | None = None,
        timeout: float = 20.0,
    ):
        if base_url.rstrip("/") != PAPER_TRADING_URL:
            raise AlpacaError(
                f"only Alpaca paper trading is supported ({PAPER_TRADING_URL}); refusing {base_url}"
            )
        self.base_url = PAPER_TRADING_URL
        self.data_url = data_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        key_id = key_id if key_id is not None else os.getenv(KEY_ID_ENV, "").strip()
        secret_key = secret_key if secret_key is not None else os.getenv(SECRET_KEY_ENV, "").strip()
        if bool(key_id) != bool(secret_key):
            raise AlpacaError(f"set both {KEY_ID_ENV} and {SECRET_KEY_ENV}, or neither")
        if key_id:
            self.session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key})

    # -- transport ---------------------------------------------------------

    def _request(self, method: str, url: str, **kwargs):
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise AlpacaError(f"could not reach Alpaca: {exc}") from exc
        if response.status_code in (401, 403):
            raise AlpacaError(
                "Alpaca refused the credentials. Use the API keys of your *paper* account "
                f"(they start with PK) in {KEY_ID_ENV} and {SECRET_KEY_ENV}."
            )
        if response.status_code >= 400:
            try:
                message = response.json().get("message") or response.text
            except ValueError:
                message = response.text
            raise AlpacaError(f"Alpaca answered {response.status_code}: {message.strip()}")
        try:
            return response.json()
        except ValueError as exc:
            raise AlpacaError(f"Alpaca sent a response that is not JSON: {response.text[:200]}") from exc

    def _trading(self, method: str, path: str, **kwargs):
        return self._request(method, f"{self.base_url}{path}", **kwargs)

    # -- reads ---------------------------------------------------------------

    def account(self) -> dict:
        return self._trading("GET", "/v2/account")

    def positions(self) -> list[dict]:
        return self._trading("GET", "/v2/positions")

    def clock(self) -> dict:
        return self._trading("GET", "/v2/clock")

    def asset(self, symbol: str) -> dict:
        return self._trading("GET", f"/v2/assets/{symbol}")

    def open_orders(self, symbol: str) -> list[dict]:
        return self._trading("GET", "/v2/orders", params={"status": "open", "symbols": symbol})

    def latest_price(self, symbol: str) -> float:
        """Last trade on the IEX feed, which every Alpaca account may read."""
        data = self._request("GET", f"{self.data_url}/v2/stocks/{symbol}/trades/latest", params={"feed": "iex"})
        try:
            price = float(data["trade"]["p"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AlpacaError(f"Alpaca has no recent trade price for {symbol}") from exc
        if not math.isfinite(price) or price <= 0:
            raise AlpacaError(f"Alpaca gave {symbol} an unusable price: {price}")
        return price

    # -- writes --------------------------------------------------------------

    def submit_market_order(self, symbol: str, qty: int, side: str, client_order_id: str) -> dict:
        if side not in ("buy", "sell"):
            raise AlpacaError(f"unknown order side {side!r}")
        if qty < 1:
            raise AlpacaError("an order needs at least one share")
        return self._trading("POST", "/v2/orders", json={
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id,
        })


# ---------------------------------------------------------------------------
# Account -> the agents' view of the book
# ---------------------------------------------------------------------------


def _position_qty(raw: dict) -> float:
    qty = float(raw["qty"])
    # Alpaca reports shorts with side "short"; keep the sign either way.
    return -abs(qty) if raw.get("side") == "short" else qty


def portfolio_from_alpaca(account: dict, positions: list[dict]) -> PortfolioContext:
    """The paper account as the trader, risk team and portfolio manager read it."""
    return PortfolioContext(
        cash=float(account["cash"]),
        currency=account.get("currency") or "USD",
        positions=[
            Position(
                ticker=p["symbol"],
                quantity=_position_qty(p),
                average_price=float(p["avg_entry_price"]) if p.get("avg_entry_price") else None,
            )
            for p in positions
        ],
    )


def held_quantity(positions: list[dict], symbol: str) -> float:
    for p in positions:
        if p["symbol"].upper() == symbol.upper():
            return _position_qty(p)
    return 0.0


# ---------------------------------------------------------------------------
# Rating -> order
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizingRules:
    max_position_pct: float = 0.10
    overweight_fraction: float = 0.5
    underweight_sell_fraction: float = 0.5
    max_order_value: float = 25_000.0

    def __post_init__(self):
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be above 0 and at most 1 (a fraction of equity)")
        if not 0 < self.overweight_fraction <= 1 or not 0 < self.underweight_sell_fraction <= 1:
            raise ValueError("overweight and underweight fractions must be above 0 and at most 1")
        if self.max_order_value <= 0:
            raise ValueError("max_order_value must be positive")


@dataclass
class OrderPlan:
    symbol: str
    rating: str
    price: float
    current_qty: float
    side: str | None = None
    qty: int = 0
    reason: str = ""

    @property
    def is_trade(self) -> bool:
        return self.side is not None and self.qty >= 1

    @property
    def value(self) -> float:
        return self.qty * self.price

    def describe(self) -> str:
        if not self.is_trade:
            return f"{self.symbol}: {self.rating}. No order: {self.reason}"
        verb = "BUY" if self.side == "buy" else "SELL"
        return (f"{self.symbol}: {self.rating}. {verb} {self.qty} shares at about "
                f"${self.price:,.2f} (${self.value:,.2f}). {self.reason}")


def plan_order(
    symbol: str,
    rating: str,
    *,
    price: float,
    equity: float,
    cash: float,
    current_qty: float,
    rules: SizingRules = SizingRules(),
) -> OrderPlan:
    """The order a rating calls for, given the account as it stands."""
    plan = OrderPlan(symbol=symbol, rating=rating, price=price, current_qty=current_qty)

    if rating == RATING_REVIEW:
        plan.reason = "the decision had no readable rating, so it needs a person to read it"
        return plan
    if current_qty < 0:
        plan.reason = f"you are short {abs(current_qty):g} shares; this bridge only trades long, so it leaves them alone"
        return plan

    if rating in ("Buy", "Overweight"):
        target = equity * rules.max_position_pct
        if rating == "Overweight":
            target *= rules.overweight_fraction
        held_value = current_qty * price
        gap = target - held_value
        spend = min(gap, cash, rules.max_order_value)
        qty = math.floor(spend / price) if spend > 0 else 0
        if gap < price:
            plan.reason = (f"already holds ${held_value:,.0f}, within one share of or above "
                           f"the ${target:,.0f} target for {rating}")
        elif qty < 1:
            plan.reason = f"not enough cash (${cash:,.2f}) for one share"
        else:
            plan.side, plan.qty = "buy", qty
            plan.reason = (f"Target for {rating}: ${target:,.0f} "
                           f"({target / equity:.0%} of ${equity:,.0f} equity).")
        return plan

    if rating == "Hold":
        plan.reason = "Hold keeps the position as it is"
        return plan

    whole = math.floor(current_qty)
    if whole < 1:
        plan.reason = "no whole shares held to sell, and this bridge never sells short"
        return plan
    if rating == "Underweight":
        qty = max(1, math.floor(whole * rules.underweight_sell_fraction))
        plan.side, plan.qty = "sell", qty
        plan.reason = f"Underweight trims {qty} of the {whole} shares held."
        return plan
    if rating == "Sell":
        plan.side, plan.qty = "sell", whole
        plan.reason = f"Sell closes the {whole}-share position."
        return plan

    plan.reason = f"unknown rating {rating!r}"
    return plan


# ---------------------------------------------------------------------------
# One ticker, end to end
# ---------------------------------------------------------------------------


def default_order_log() -> Path:
    override = os.getenv(ORDER_LOG_ENV)
    if override:
        return Path(override)
    return Path.home() / ".tradingagents" / "alpaca" / "orders.jsonl"


@dataclass
class TradeOutcome:
    symbol: str
    trade_date: str
    rating: str | None = None
    plan: OrderPlan | None = None
    order: dict | None = None
    report_path: str | None = None
    skipped: str | None = None
    errors: list[str] = field(default_factory=list)

    def to_log(self) -> dict:
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "rating": self.rating,
            "plan": asdict(self.plan) if self.plan else None,
            "order": {k: self.order.get(k) for k in
                      ("id", "client_order_id", "status", "side", "qty", "symbol", "submitted_at")}
            if self.order else None,
            "report_path": self.report_path,
            "skipped": self.skipped,
            "errors": self.errors,
        }
        return record


def _append_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def trade_ticker(
    client: AlpacaClient,
    graph,
    symbol: str,
    trade_date: str,
    *,
    rules: SizingRules = SizingRules(),
    confirm: Callable[[OrderPlan], bool],
    dry_run: bool = False,
    save_reports: bool = True,
    order_log: Path | None = None,
    say: Callable[[str], None] = print,
) -> TradeOutcome:
    """Analyse one ticker with the agents and place the order its rating calls for.

    ``graph`` is a TradingAgentsGraph (anything with ``propagate`` and
    ``save_reports``). ``confirm`` is asked before an order is sent; returning
    False leaves the account untouched.
    """
    symbol = symbol.strip().upper()
    outcome = TradeOutcome(symbol=symbol, trade_date=trade_date)
    log_path = order_log or default_order_log()

    def finish() -> TradeOutcome:
        _append_log(log_path, outcome.to_log())
        return outcome

    # Check Alpaca can trade it before spending an analysis on it.
    asset = client.asset(symbol)
    if not asset.get("tradable"):
        outcome.skipped = f"Alpaca does not trade {symbol}"
        say(outcome.skipped)
        return finish()

    account = client.account()
    positions = client.positions()
    book = portfolio_from_alpaca(account, positions)
    say(f"Analysing {symbol} for {trade_date} with your paper account's "
        f"${float(account['cash']):,.2f} cash and {len(positions)} open position(s)...")

    final_state, rating = graph.propagate(symbol, trade_date, portfolio=book)
    outcome.rating = rating
    if save_reports:
        try:
            outcome.report_path = str(graph.save_reports(final_state, symbol))
        except Exception as exc:  # a report that fails to save must not block the trade
            outcome.errors.append(f"report not saved: {exc}")

    # The analysis takes minutes; read the account again before sizing.
    account = client.account()
    positions = client.positions()
    plan = plan_order(
        symbol, rating,
        price=client.latest_price(symbol),
        equity=float(account["equity"]),
        cash=float(account["cash"]),
        current_qty=held_quantity(positions, symbol),
        rules=rules,
    )
    outcome.plan = plan
    say(plan.describe())

    if not plan.is_trade:
        return finish()
    if dry_run:
        outcome.skipped = "dry run: no order sent"
        say(outcome.skipped)
        return finish()
    if client.open_orders(symbol):
        outcome.skipped = f"an order for {symbol} is already open on the account; not adding another"
        say(outcome.skipped)
        return finish()
    if not confirm(plan):
        outcome.skipped = "not approved: no order sent"
        say(outcome.skipped)
        return finish()

    client_order_id = f"ta-{symbol[:12]}-{trade_date}-{uuid.uuid4().hex[:8]}"
    outcome.order = client.submit_market_order(symbol, plan.qty, plan.side, client_order_id)
    say(f"Order sent: {plan.side} {plan.qty} {symbol}, status {outcome.order.get('status')} "
        f"(id {outcome.order.get('id')}).")
    clock = client.clock()
    if not clock.get("is_open"):
        say(f"The market is closed; Alpaca fills it after the next open ({clock.get('next_open')}).")
    return finish()
