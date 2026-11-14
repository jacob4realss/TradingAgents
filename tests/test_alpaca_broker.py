"""The Alpaca paper bridge: rating -> order sizing, the account read as a book,
paper-only transport, and one ticker end to end against a fake Alpaca."""

import json

import pytest
from typer.testing import CliRunner

from tradingagents.brokers import alpaca
from tradingagents.brokers.alpaca import (
    AlpacaClient,
    AlpacaError,
    SizingRules,
    plan_order,
    portfolio_from_alpaca,
    trade_ticker,
)

pytestmark = pytest.mark.unit

ACCOUNT = {"cash": "100000", "equity": "100000", "buying_power": "200000", "currency": "USD",
           "account_number": "PA123", "status": "ACTIVE"}


def plan(rating, *, qty=0.0, price=100.0, cash=100_000.0, equity=100_000.0, **rules):
    return plan_order("NVDA", rating, price=price, equity=equity, cash=cash, current_qty=qty,
                      rules=SizingRules(**rules))


class TestSizing:
    def test_buy_fills_the_position_to_its_share_of_equity(self):
        p = plan("Buy")
        assert (p.side, p.qty) == ("buy", 100)  # 10% of 100k at $100

    def test_buy_tops_up_only_the_gap(self):
        assert plan("Buy", qty=60).qty == 40
        assert not plan("Buy", qty=100).is_trade

    def test_overweight_targets_half_a_buy(self):
        p = plan("Overweight")
        assert (p.side, p.qty) == ("buy", 50)

    def test_buys_never_spend_more_than_cash_or_the_order_cap(self):
        assert plan("Buy", cash=2_550).qty == 25
        assert plan("Buy", max_order_value=1_000).qty == 10
        assert not plan("Buy", cash=50).is_trade

    def test_hold_and_review_send_nothing(self):
        assert not plan("Hold", qty=10).is_trade
        assert not plan("REVIEW", qty=10).is_trade

    def test_underweight_trims_and_sell_closes(self):
        assert (plan("Underweight", qty=10).side, plan("Underweight", qty=10).qty) == ("sell", 5)
        assert plan("Underweight", qty=1).qty == 1
        assert (plan("Sell", qty=7.5).side, plan("Sell", qty=7.5).qty) == ("sell", 7)

    def test_nothing_is_ever_sold_short(self):
        assert not plan("Sell", qty=0).is_trade
        assert not plan("Underweight", qty=0).is_trade
        assert not plan("Buy", qty=-5).is_trade  # an existing short is left alone

    def test_rules_reject_nonsense(self):
        with pytest.raises(ValueError):
            SizingRules(max_position_pct=10)  # a percent where a fraction belongs
        with pytest.raises(ValueError):
            SizingRules(max_order_value=0)


def test_the_account_becomes_the_agents_book():
    book = portfolio_from_alpaca(ACCOUNT, [
        {"symbol": "AAPL", "qty": "3", "avg_entry_price": "190.5", "side": "long"},
        {"symbol": "TSLA", "qty": "2", "avg_entry_price": "250", "side": "short"},
    ])
    assert book.cash == 100_000 and book.currency == "USD"
    assert book.position_in("AAPL").quantity == 3
    assert book.position_in("TSLA").quantity == -2
    assert "No current position in NVDA" in book.render("NVDA")


class TestTransport:
    def test_only_the_paper_host_is_accepted(self):
        with pytest.raises(AlpacaError, match="paper"):
            AlpacaClient("k", "s", base_url="https://api.alpaca.markets")

    def test_keys_come_in_pairs(self, monkeypatch):
        monkeypatch.delenv(alpaca.KEY_ID_ENV, raising=False)
        monkeypatch.delenv(alpaca.SECRET_KEY_ENV, raising=False)
        with pytest.raises(AlpacaError, match="both"):
            AlpacaClient("PKabc", "")
        client = AlpacaClient()  # neither: requests go out unsigned, for a signing proxy
        assert "APCA-API-KEY-ID" not in client.session.headers

    def test_keys_are_sent_as_alpaca_headers(self):
        client = AlpacaClient("PKabc", "secret")
        assert client.session.headers["APCA-API-KEY-ID"] == "PKabc"
        assert client.session.headers["APCA-API-SECRET-KEY"] == "secret"

    def test_refused_keys_say_which_keys_to_use(self):
        client = AlpacaClient("PKabc", "secret", session=FakeSession(FakeAlpaca(status=401)))
        with pytest.raises(AlpacaError, match="paper"):
            client.account()


# ---------------------------------------------------------------------------
# A fake Alpaca, served through a requests-shaped session
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeAlpaca:
    def __init__(self, *, positions=(), price=100.0, tradable=True, open_orders=(), status=200):
        self.positions = list(positions)
        self.price, self.tradable, self.open = price, tradable, list(open_orders)
        self.status = status
        self.orders = []

    def handle(self, method, url, params=None, json=None):
        if self.status != 200:
            return FakeResponse(self.status, {"message": "unauthorized."})
        path = url.split(".markets", 1)[1]
        if path == "/v2/account":
            return FakeResponse(200, ACCOUNT)
        if path == "/v2/positions":
            return FakeResponse(200, self.positions)
        if path == "/v2/clock":
            return FakeResponse(200, {"is_open": False, "next_open": "2026-11-16T09:30:00-05:00"})
        if path.startswith("/v2/assets/"):
            return FakeResponse(200, {"symbol": path.rsplit("/", 1)[1], "tradable": self.tradable})
        if path.endswith("/trades/latest"):
            return FakeResponse(200, {"trade": {"p": self.price}})
        if path == "/v2/orders" and method == "GET":
            return FakeResponse(200, self.open)
        if path == "/v2/orders" and method == "POST":
            self.orders.append(json)
            return FakeResponse(200, {"id": "ord-1", "status": "accepted", **json})
        return FakeResponse(404, {"message": "not found"})


class FakeSession:
    def __init__(self, server):
        self.server, self.headers = server, {}

    def request(self, method, url, timeout=None, **kwargs):
        return self.server.handle(method, url, **kwargs)


class FakeGraph:
    def __init__(self, rating):
        self.rating, self.books = rating, []

    def propagate(self, ticker, date, portfolio=None):
        self.books.append(portfolio)
        return {"final_trade_decision": f"Rating: {self.rating}"}, self.rating

    def save_reports(self, state, ticker):
        return "/tmp/report"


def run(server, rating, tmp_path, **kwargs):
    client = AlpacaClient("PKabc", "secret", session=FakeSession(server))
    graph = FakeGraph(rating)
    kwargs.setdefault("confirm", lambda plan: True)
    outcome = trade_ticker(client, graph, "nvda", "2026-11-13", order_log=tmp_path / "orders.jsonl",
                           say=lambda line: None, **kwargs)
    return outcome, graph


class TestEndToEnd:
    def test_a_buy_is_sized_sent_and_logged(self, tmp_path):
        server = FakeAlpaca()
        outcome, graph = run(server, "Buy", tmp_path)
        assert graph.books[0].cash == 100_000  # the agents saw the account
        assert server.orders == [{"symbol": "NVDA", "qty": "100", "side": "buy", "type": "market",
                                  "time_in_force": "day", "client_order_id": server.orders[0]["client_order_id"]}]
        assert outcome.order["status"] == "accepted"
        logged = json.loads((tmp_path / "orders.jsonl").read_text())
        assert logged["rating"] == "Buy" and logged["order"]["id"] == "ord-1"

    def test_a_sell_closes_a_held_position(self, tmp_path):
        server = FakeAlpaca(positions=[{"symbol": "NVDA", "qty": "12", "avg_entry_price": "90", "side": "long"}])
        run(server, "Sell", tmp_path)
        assert [(o["side"], o["qty"]) for o in server.orders] == [("sell", "12")]

    def test_nothing_is_sent_without_approval_on_a_dry_run_or_over_an_open_order(self, tmp_path):
        for kwargs, server in (
            ({"confirm": lambda plan: False}, FakeAlpaca()),
            ({"dry_run": True}, FakeAlpaca()),
            ({}, FakeAlpaca(open_orders=[{"id": "x"}])),
        ):
            outcome, _ = run(server, "Buy", tmp_path, **kwargs)
            assert server.orders == [] and outcome.skipped

    def test_an_untradable_ticker_is_skipped_before_any_analysis(self, tmp_path):
        server = FakeAlpaca(tradable=False)
        outcome, graph = run(server, "Buy", tmp_path)
        assert graph.books == [] and server.orders == [] and "does not trade" in outcome.skipped


def test_the_cli_lists_the_alpaca_commands():
    from cli.main import app

    result = CliRunner().invoke(app, ["alpaca", "--help"])
    assert result.exit_code == 0
    assert "account" in result.output and "trade" in result.output
