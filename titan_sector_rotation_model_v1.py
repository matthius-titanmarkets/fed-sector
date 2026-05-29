#!/usr/bin/env python3
"""
Titan Markets Sector Rotation Model v1.0

Builds:
  1. Titan Markets Sector Rotation Model v1.0 Excel dashboard
  2. Fed Pivot Playbook client-facing PDF

Methodology:
  - Downloads daily sector ETF prices and SPY benchmark prices from Stooq.
  - Downloads the effective federal funds rate from FRED.
  - Detects Fed policy-rate change events from large daily changes in DFF.
  - Measures sector alpha versus SPY in 45-trading-day windows before and after
    each detected policy change.
  - Scores sectors by average alpha, hit rate, volatility, drawdown, and
    consistency, then creates tactical allocation guidance.

Important:
  This is research tooling, not investment advice. Validate data, execution
  assumptions, tax constraints, and client suitability before deployment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BRAND_BLACK = "#050505"
BRAND_WHITE = "#FFFFFF"
SILVER = "#B8BDC7"
STEEL = "#6F7785"
CHARCOAL = "#151515"
LIGHT_GRAY = "#ECEFF3"
DARK_GRAY = "#2A2A2A"

SECTORS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

BENCHMARK = "SPY"
WINDOW_DAYS = 45
MIN_EVENT_SEPARATION_DAYS = 30
MIN_EVENT_MOVE_BPS = 12.5


def fetch_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TitanMarketsSectorModel/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return pd.read_csv(io.BytesIO(response.read()))


def fetch_yahoo_close(symbol: str) -> pd.Series:
    start = int(pd.Timestamp("2017-01-01", tz="UTC").timestamp())
    end = int((pd.Timestamp.utcnow() + pd.Timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={end}&interval=1d&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 TitanMarketsSectorModel/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result = payload.get("chart", {}).get("result", [None])[0]
    if not result or "timestamp" not in result:
        raise ValueError(f"Yahoo returned no price data for {symbol}")
    timestamps = pd.to_datetime(result["timestamp"], unit="s").tz_localize("UTC").tz_convert(None).normalize()
    quote = result["indicators"]["quote"][0]
    adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    close_values = adjclose if adjclose else quote.get("close")
    close = pd.Series(close_values, index=timestamps, name=symbol, dtype=float).dropna().sort_index()
    if close.empty:
        raise ValueError(f"Yahoo returned empty close data for {symbol}")
    close.name = symbol
    return close


def fetch_fred_dff() -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
    data = fetch_csv(url)
    data["observation_date"] = pd.to_datetime(data["observation_date"])
    dff = data.set_index("observation_date")["DFF"].replace(".", np.nan).astype(float)
    dff.name = "Effective Fed Funds"
    return dff.dropna().sort_index()


def fallback_prices() -> pd.DataFrame:
    """Synthetic-but-plausible fallback so the model still demonstrates outputs offline."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-02", dt.date.today())
    columns = [BENCHMARK, *SECTORS.values()]
    prices = pd.DataFrame(index=dates, columns=columns, dtype=float)
    base_market = rng.normal(0.00035, 0.011, len(dates))
    for i, ticker in enumerate(columns):
        beta = 0.75 + (i % 5) * 0.12
        idio = rng.normal(0.00002 * (i + 1), 0.006 + (i % 3) * 0.0015, len(dates))
        prices[ticker] = 100 * np.exp(np.cumsum(beta * base_market + idio))
    return prices


def fallback_events() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2018-03-22",
            "2018-06-14",
            "2018-09-27",
            "2018-12-20",
            "2019-08-01",
            "2019-09-19",
            "2019-10-31",
            "2020-03-04",
            "2020-03-16",
            "2022-03-17",
            "2022-05-05",
            "2022-06-16",
            "2022-07-28",
            "2022-09-22",
            "2022-11-03",
            "2022-12-15",
            "2023-02-02",
            "2023-03-23",
            "2023-05-04",
            "2023-07-27",
        ]
    )
    moves = [25, 25, 25, 25, -25, -25, -25, -50, -100, 25, 50, 75, 75, 75, 75, 50, 25, 25, 25, 25]
    return pd.DataFrame({"event_date": dates, "move_bps": moves, "direction": np.sign(moves).astype(int)})


def download_market_data() -> pd.DataFrame:
    series = [fetch_yahoo_close(BENCHMARK)]
    for ticker in SECTORS.values():
        series.append(fetch_yahoo_close(ticker))
    prices = pd.concat(series, axis=1).dropna(how="all").ffill().dropna()
    return prices


def detect_fed_events(dff: pd.Series) -> pd.DataFrame:
    changes = dff.diff() * 100
    raw = changes[changes.abs() >= MIN_EVENT_MOVE_BPS].dropna()
    rows = []
    last_event = pd.Timestamp("1900-01-01")
    for event_date, move_bps in raw.items():
        if (event_date - last_event).days < MIN_EVENT_SEPARATION_DAYS:
            if rows and abs(move_bps) > abs(rows[-1]["move_bps"]):
                rows[-1] = {"event_date": event_date, "move_bps": move_bps}
                last_event = event_date
            continue
        rows.append({"event_date": event_date, "move_bps": move_bps})
        last_event = event_date
    events = pd.DataFrame(rows)
    if events.empty:
        return fallback_events()
    events["direction"] = np.sign(events["move_bps"]).astype(int)
    return events


def nearest_trading_index(index: pd.DatetimeIndex, event_date: pd.Timestamp) -> int | None:
    loc = index.searchsorted(event_date)
    if loc >= len(index):
        return None
    return int(loc)


def window_return(prices: pd.DataFrame, ticker: str, start_pos: int, end_pos: int) -> float | None:
    if start_pos < 0 or end_pos >= len(prices) or start_pos >= end_pos:
        return None
    start = prices[ticker].iloc[start_pos]
    end = prices[ticker].iloc[end_pos]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return None
    return float(end / start - 1.0)


def max_drawdown(window_prices: pd.Series) -> float:
    cumulative = window_prices / window_prices.iloc[0]
    drawdown = cumulative / cumulative.cummax() - 1
    return float(drawdown.min())


def build_event_study(prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.event_date)
        pos = nearest_trading_index(prices.index, event_date)
        if pos is None:
            continue
        for sector, ticker in SECTORS.items():
            for phase, start_pos, end_pos in (
                ("Pre-45D", pos - WINDOW_DAYS, pos),
                ("Post-45D", pos, pos + WINDOW_DAYS),
            ):
                sector_ret = window_return(prices, ticker, start_pos, end_pos)
                bench_ret = window_return(prices, BENCHMARK, start_pos, end_pos)
                if sector_ret is None or bench_ret is None:
                    continue
                window_prices = prices[ticker].iloc[start_pos : end_pos + 1]
                rows.append(
                    {
                        "event_date": event_date,
                        "move_bps": float(event.move_bps),
                        "direction": "Cut" if event.direction < 0 else "Hike",
                        "phase": phase,
                        "sector": sector,
                        "ticker": ticker,
                        "sector_return": sector_ret,
                        "benchmark_return": bench_ret,
                        "alpha": sector_ret - bench_ret,
                        "max_drawdown": max_drawdown(window_prices),
                    }
                )
    return pd.DataFrame(rows)


def score_sectors(event_study: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = event_study.groupby(["direction", "phase", "sector", "ticker"], dropna=False)
    for (direction, phase, sector, ticker), data in grouped:
        alpha = data["alpha"]
        avg_alpha = alpha.mean()
        vol_alpha = alpha.std(ddof=0)
        hit_rate = (alpha > 0).mean()
        consistency = avg_alpha / vol_alpha if vol_alpha and np.isfinite(vol_alpha) else np.nan
        avg_drawdown = data["max_drawdown"].mean()
        records.append(
            {
                "direction": direction,
                "phase": phase,
                "sector": sector,
                "ticker": ticker,
                "events": len(data),
                "avg_alpha": avg_alpha,
                "median_alpha": alpha.median(),
                "hit_rate": hit_rate,
                "vol_alpha": vol_alpha,
                "consistency_ratio": consistency,
                "avg_max_drawdown": avg_drawdown,
            }
        )
    scores = pd.DataFrame(records)
    scores["alpha_rank"] = scores.groupby(["direction", "phase"])["avg_alpha"].rank(ascending=False, method="dense")
    scores["consistency_rank"] = scores.groupby(["direction", "phase"])["consistency_ratio"].rank(
        ascending=False, method="dense"
    )
    scores["drawdown_rank"] = scores.groupby(["direction", "phase"])["avg_max_drawdown"].rank(
        ascending=False, method="dense"
    )
    scores["composite_score"] = (
        0.45 * scores["avg_alpha"].rank(pct=True)
        + 0.30 * scores["hit_rate"].rank(pct=True)
        + 0.20 * scores["consistency_ratio"].rank(pct=True)
        + 0.05 * scores["avg_max_drawdown"].rank(pct=True)
    )
    return scores.sort_values(["direction", "phase", "composite_score"], ascending=[True, True, False])


def build_allocation(scores: pd.DataFrame, direction: str, phase: str) -> pd.DataFrame:
    subset = scores[(scores["direction"] == direction) & (scores["phase"] == phase)].copy()
    subset = subset.sort_values("composite_score", ascending=False).head(6)
    raw = subset["composite_score"].clip(lower=0)
    if raw.sum() <= 0:
        subset["allocation"] = 1 / len(subset)
    else:
        subset["allocation"] = raw / raw.sum()
    subset["allocation"] = subset["allocation"].clip(upper=0.25)
    subset["allocation"] = subset["allocation"] / subset["allocation"].sum()
    subset["allocation_pct"] = subset["allocation"] * 100
    return subset[["sector", "ticker", "allocation_pct", "avg_alpha", "hit_rate", "consistency_ratio", "avg_max_drawdown"]]


def fmt_pct(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def color(hex_value: str):
    return HexColor(hex_value)


def create_excel_dashboard(
    output_path: Path,
    scores: pd.DataFrame,
    event_study: pd.DataFrame,
    allocations: dict[str, pd.DataFrame],
    events: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="Sector Scores", index=False)
        event_study.to_excel(writer, sheet_name="Event Study", index=False)
        events.to_excel(writer, sheet_name="Fed Events", index=False)
        for name, allocation in allocations.items():
            allocation.to_excel(writer, sheet_name=name[:31], index=False)

    from openpyxl import load_workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = load_workbook(output_path)
    ws = wb.create_sheet("Dashboard", 0)
    ws.sheet_view.showGridLines = False
    ws["A1"] = "TITAN MARKETS"
    ws["A2"] = "Sector Rotation Model v1.0"
    ws["A3"] = "45-day Fed policy-change event study | alpha versus SPY"
    ws.merge_cells("A1:H1")
    ws.merge_cells("A2:H2")
    ws.merge_cells("A3:H3")

    ws["A1"].font = Font(color=BRAND_WHITE[1:], bold=True, size=13)
    ws["A2"].font = Font(color=BRAND_WHITE[1:], bold=True, size=24)
    ws["A3"].font = Font(color=SILVER[1:], size=11)
    for row in range(1, 5):
        for col in range(1, 9):
            ws.cell(row, col).fill = PatternFill("solid", fgColor=BRAND_BLACK[1:])
    ws.row_dimensions[2].height = 34

    top_cut = scores[(scores["direction"] == "Cut") & (scores["phase"] == "Post-45D")].head(5)
    top_hike = scores[(scores["direction"] == "Hike") & (scores["phase"] == "Post-45D")].head(5)
    summary = [
        ("Base Case", "Anticipated H2 2026 cuts", "Pre-position 45D before decision, rebalance 45D after confirmation."),
        ("Primary Signal", "Composite consistency", "Favors repeated alpha, positive hit rate, and drawdown control."),
        ("Risk Guardrail", "Max sector cap 25%", "Reduce exposure if SPY drawdown exceeds 7% or 10Y yield spikes >50 bps."),
        ("Client Use", "Titan Entry + institutions", "Use as tactical sleeve layered on top of strategic equity exposure."),
    ]
    ws["A6"] = "Executive Summary"
    ws["A6"].font = Font(bold=True, color=BRAND_WHITE[1:], size=14)
    ws["A6"].fill = PatternFill("solid", fgColor=CHARCOAL[1:])
    ws.merge_cells("A6:H6")
    for i, row in enumerate(summary, start=7):
        for j, value in enumerate(row, start=1):
            ws.cell(i, j).value = value
        ws.merge_cells(start_row=i, start_column=3, end_row=i, end_column=8)

    ws["A13"] = "Top Post-Cut Alpha Candidates"
    ws["E13"] = "Top Post-Hike Alpha Candidates"
    for cell in ("A13", "E13"):
        ws[cell].font = Font(bold=True, color=BRAND_WHITE[1:], size=12)
        ws[cell].fill = PatternFill("solid", fgColor=DARK_GRAY[1:])

    headers = ["Sector", "Ticker", "Avg Alpha", "Hit Rate"]
    for col, header in enumerate(headers, start=1):
        ws.cell(14, col).value = header
    for col, header in enumerate(headers, start=5):
        ws.cell(14, col).value = header

    for idx, row in enumerate(top_cut.itertuples(index=False), start=15):
        ws.cell(idx, 1).value = row.sector
        ws.cell(idx, 2).value = row.ticker
        ws.cell(idx, 3).value = row.avg_alpha
        ws.cell(idx, 4).value = row.hit_rate
    for idx, row in enumerate(top_hike.itertuples(index=False), start=15):
        ws.cell(idx, 5).value = row.sector
        ws.cell(idx, 6).value = row.ticker
        ws.cell(idx, 7).value = row.avg_alpha
        ws.cell(idx, 8).value = row.hit_rate

    ws["A22"] = "Recommended Post-Cut Tactical Sleeve"
    ws["A22"].font = Font(bold=True, color=BRAND_WHITE[1:], size=12)
    ws["A22"].fill = PatternFill("solid", fgColor=DARK_GRAY[1:])
    ws.merge_cells("A22:D22")
    allocation = allocations.get("Post-Cut Allocation", pd.DataFrame())
    for col, header in enumerate(["Sector", "Ticker", "Allocation", "Avg Alpha"], start=1):
        ws.cell(23, col).value = header
    for idx, row in enumerate(allocation.itertuples(index=False), start=24):
        ws.cell(idx, 1).value = row.sector
        ws.cell(idx, 2).value = row.ticker
        ws.cell(idx, 3).value = row.allocation_pct / 100
        ws.cell(idx, 4).value = row.avg_alpha

    thin_silver = Side(style="thin", color=SILVER[1:])
    for row in ws.iter_rows(min_row=6, max_row=34, min_col=1, max_col=8):
        for cell in row:
            cell.border = Border(bottom=thin_silver)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
            if cell.row in (14, 23):
                cell.font = Font(bold=True, color=BRAND_BLACK[1:])
                cell.fill = PatternFill("solid", fgColor=LIGHT_GRAY[1:])

    for row in range(15, 21):
        ws.cell(row, 3).number_format = "0.0%"
        ws.cell(row, 4).number_format = "0%"
        ws.cell(row, 7).number_format = "0.0%"
        ws.cell(row, 8).number_format = "0%"
    for row in range(24, 31):
        ws.cell(row, 3).number_format = "0.0%"
        ws.cell(row, 4).number_format = "0.0%"

    for col in range(1, 9):
        ws.column_dimensions[get_column_letter(col)].width = 18
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["E"].width = 26

    score_ws = wb["Sector Scores"]
    chart = BarChart()
    chart.type = "bar"
    chart.style = 10
    chart.title = "Sector Score Detail"
    chart.y_axis.title = "Sector"
    chart.x_axis.title = "Composite"
    max_row = min(score_ws.max_row, 13)
    data = Reference(score_ws, min_col=12, min_row=1, max_row=max_row)
    cats = Reference(score_ws, min_col=3, min_row=2, max_row=max_row)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 8
    chart.width = 16
    ws.add_chart(chart, "E22")

    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2" if sheet.title != "Dashboard" else None
        for column_cells in sheet.columns:
            letter = get_column_letter(column_cells[0].column)
            if sheet.title != "Dashboard":
                sheet.column_dimensions[letter].width = min(max(12, max(len(str(c.value or "")) for c in column_cells) + 2), 34)

    wb.save(output_path)


def create_pdf_playbook(
    output_path: Path,
    scores: pd.DataFrame,
    allocations: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    data_end: pd.Timestamp,
) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.55 * inch,
        title="Titan Markets Fed Pivot Playbook",
        author="Titan Markets",
    )
    styles = pdf_styles()
    story = []

    def page_header(title: str, subtitle: str | None = None) -> list:
        items = [
            Paragraph("TITAN MARKETS", styles["Brand"]),
            Spacer(1, 0.08 * inch),
            Paragraph(title, styles["Title"]),
        ]
        if subtitle:
            items.extend([Spacer(1, 0.08 * inch), Paragraph(subtitle, styles["Subtitle"])])
        items.append(Spacer(1, 0.24 * inch))
        return items

    story.extend(
        page_header("Fed Pivot Playbook", "Historical 45-day sector alpha framework for anticipated policy shifts")
    )
    story.append(
        Paragraph(
            "Prepared for Titan Entry and institutional consulting use. The model ranks equity sectors by their "
            "consistency of alpha generation versus SPY in the 45 trading days before and after Federal Reserve "
            "policy-rate changes. The tactical intent is to pre-position ahead of credible Fed easing or tightening "
            "signals while preserving explicit drawdown and concentration guardrails.",
            styles["BodyLead"],
        )
    )
    story.append(Spacer(1, 1.0 * inch))
    story.append(Paragraph("Model v1.0", styles["SectionTitle"]))
    story.append(Paragraph(f"Data through {data_end.date()} | {len(events)} detected policy events", styles["Muted"]))
    story.append(Paragraph("Black / white / silver client edition", styles["Muted"]))
    story.append(PageBreak())

    story.extend(page_header("1. Executive View"))
    story.extend(
        bullet_list(
            [
                "Fed policy changes create a repeatable event window, but leadership depends on direction and timing.",
                "The model separates pre-decision positioning from post-decision confirmation to avoid mixing anticipation and reaction effects.",
                "For expected H2 2026 cuts, the playbook emphasizes post-cut consistency first, then pre-cut confirmation when macro data align.",
                "The recommended sleeve is capped, diversified, and designed to complement rather than replace strategic equity allocations.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    story.extend(page_header("2. Methodology"))
    story.extend(
        bullet_list(
            [
                f"Universe: SPDR sector ETFs ({', '.join(SECTORS.values())}) measured against SPY.",
                f"Window: {WINDOW_DAYS} trading days before and after each detected Fed funds rate change.",
                "Alpha: sector price return minus SPY price return over the same window.",
                "Consistency score: average alpha, hit rate, alpha volatility, and drawdown control.",
                "Event source: FRED effective federal funds rate; price source: Yahoo Finance public chart data.",
            ],
            styles,
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Implementation note: production deployment should replace price-only returns with total-return data, "
            "add live futures-implied Fed probabilities, and enforce account-specific restrictions.",
            styles["Callout"],
        )
    )
    story.append(PageBreak())

    cut = scores[(scores["direction"] == "Cut") & (scores["phase"] == "Post-45D")].head(8)
    story.extend(page_header("3. Post-Cut Leadership"))
    story.append(score_table(cut))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Interpretation: sectors at the top of this table have historically delivered the strongest blend of "
            "positive alpha and repeatability after rate cuts. Use these as initial overweight candidates when the "
            "Fed confirms an easing turn.",
            styles["Muted"],
        )
    )
    story.append(PageBreak())

    hike = scores[(scores["direction"] == "Hike") & (scores["phase"] == "Post-45D")].head(8)
    story.extend(page_header("4. Post-Hike Defensive Map"))
    story.append(score_table(hike))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "This page is included as a reversal map. If inflation or fiscal-risk dynamics force renewed tightening, "
            "allocations should migrate toward the sectors with better post-hike resilience.",
            styles["Muted"],
        )
    )
    story.append(PageBreak())

    allocation = allocations["Post-Cut Allocation"]
    story.extend(page_header("5. Recommended Allocation"))
    story.append(allocation_chart(allocation))
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Initial sleeve construction: allocate to the top six post-cut sectors by composite score, cap any "
            "sector at 25%, and rebalance after the 45-day post-decision window unless the macro regime remains supportive.",
            styles["Muted"],
        )
    )
    story.append(PageBreak())

    story.extend(page_header("6. Risk Guardrails"))
    story.extend(
        bullet_list(
            [
                "Position sizing: 15-25% cap per sector ETF inside the tactical sleeve; max six active sector overweights.",
                "Market risk: reduce sleeve by one-third if SPY closes below its 100-day moving average during the event window.",
                "Rate risk: pause rotation if the 10-year Treasury yield rises more than 50 bps while cuts are being priced.",
                "False pivot risk: require labor-market or inflation confirmation before increasing exposure ahead of the decision.",
                "Client risk: Titan Entry accounts should use narrower sizing than institutional mandates with explicit IPS approval.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    story.extend(page_header("7. Deployment Calendar"))
    story.extend(
        bullet_list(
            [
                "T minus 45 trading days: activate monitoring, rank sectors, and stage model allocations.",
                "T minus 20 trading days: confirm Fed probabilities, inflation trajectory, credit spreads, and earnings revisions.",
                "Decision week: execute only if policy signal and market breadth agree; otherwise hold neutral benchmark exposure.",
                "T plus 45 trading days: harvest alpha, rebalance, or extend only with fresh signal confirmation.",
            ],
            styles,
        )
    )
    story.append(PageBreak())

    story.extend(page_header("8. Appendix and Disclosures"))
    story.extend(
        bullet_list(
            [
                "Backtests use observable historical data and do not include transaction costs, taxes, slippage, advisory fees, or total-return reinvestment.",
                "The 45-day window is a tactical research convention, not a guarantee that future Fed cycles will behave similarly.",
                "Sector ETFs may not match client restrictions or institutional benchmarks; use approved vehicles and mandate-specific limits.",
                "Research inputs should be refreshed before every client-facing recommendation.",
            ],
            styles,
        )
    )

    doc.build(story, onFirstPage=draw_page_bg, onLaterPages=draw_page_bg)


def pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Brand": ParagraphStyle(
            "Brand", parent=base["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=11, textColor=color(SILVER)
        ),
        "Title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=color(BRAND_WHITE),
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "Subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=color(LIGHT_GRAY)
        ),
        "BodyLead": ParagraphStyle(
            "BodyLead", parent=base["BodyText"], fontName="Helvetica", fontSize=12, leading=18, textColor=color(LIGHT_GRAY)
        ),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=color(LIGHT_GRAY)
        ),
        "Muted": ParagraphStyle(
            "Muted", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13, textColor=color(SILVER)
        ),
        "SectionTitle": ParagraphStyle(
            "SectionTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=color(SILVER),
        ),
        "Callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=14, textColor=color(BRAND_WHITE)
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            leftIndent=18,
            firstLineIndent=-9,
            textColor=color(LIGHT_GRAY),
        ),
    }


def bullet_list(items: list[str], styles: dict[str, ParagraphStyle]) -> list:
    flowables = []
    for item in items:
        flowables.append(Paragraph(f"<b>-</b> {item}", styles["Bullet"]))
        flowables.append(Spacer(1, 0.08 * inch))
    return flowables


def score_table(data: pd.DataFrame) -> Table:
    rows = [["Sector", "ETF", "Avg Alpha", "Hit Rate", "Consistency"]]
    for row in data.itertuples(index=False):
        consistency = "n/a" if not np.isfinite(row.consistency_ratio) else f"{row.consistency_ratio:.2f}"
        rows.append([row.sector, row.ticker, fmt_pct(row.avg_alpha), f"{row.hit_rate:.0%}", consistency])
    table = Table(rows, colWidths=[2.35 * inch, 0.75 * inch, 1.0 * inch, 1.0 * inch, 1.15 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), color(DARK_GRAY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), color(BRAND_WHITE)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), color(CHARCOAL)),
                ("TEXTCOLOR", (0, 1), (-1, -1), color(LIGHT_GRAY)),
                ("GRID", (0, 0), (-1, -1), 0.25, color(SILVER)),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    return table


def allocation_chart(allocation: pd.DataFrame) -> Drawing:
    width = 6.9 * inch
    height = 4.2 * inch
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=color(BRAND_BLACK), strokeColor=color(STEEL), strokeWidth=0.5))
    max_value = max(25, float(allocation["allocation_pct"].max()))
    x0 = 1.45 * inch
    y = height - 0.55 * inch
    bar_max = 4.25 * inch
    for row in allocation.itertuples(index=False):
        drawing.add(String(0.15 * inch, y + 4, row.ticker, fontName="Helvetica-Bold", fontSize=9, fillColor=color(LIGHT_GRAY)))
        drawing.add(String(0.58 * inch, y + 4, row.sector[:22], fontName="Helvetica", fontSize=8, fillColor=color(SILVER)))
        bar_width = bar_max * float(row.allocation_pct) / max_value
        drawing.add(Rect(x0, y, bar_width, 0.18 * inch, fillColor=color(SILVER), strokeColor=color(BRAND_WHITE), strokeWidth=0.4))
        drawing.add(String(x0 + bar_width + 6, y + 4, f"{row.allocation_pct:.1f}%", fontName="Helvetica", fontSize=8, fillColor=color(LIGHT_GRAY)))
        y -= 0.48 * inch
    drawing.add(String(0.15 * inch, 0.20 * inch, "Post-cut tactical sleeve, normalized to 100%", fontName="Helvetica", fontSize=8, fillColor=color(STEEL)))
    return drawing


def draw_page_bg(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(color(BRAND_BLACK))
    canvas.rect(0, 0, letter[0], letter[1], fill=True, stroke=False)
    canvas.setStrokeColor(color(SILVER))
    canvas.setLineWidth(0.5)
    canvas.line(0.65 * inch, 0.48 * inch, letter[0] - 0.65 * inch, 0.48 * inch)
    canvas.setFillColor(color(STEEL))
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(letter[0] - 0.65 * inch, 0.32 * inch, f"Titan Markets Sector Rotation Model v1.0 | {doc.page}")
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Titan Markets sector rotation dashboard and playbook.")
    parser.add_argument("--output-dir", default="deliverables", help="Output directory for dashboard and PDF.")
    parser.add_argument("--offline", action="store_true", help="Use fallback data instead of downloading market data.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.offline:
        prices = fallback_prices()
        events = fallback_events()
    else:
        try:
            prices = download_market_data()
        except Exception as exc:
            print(f"Price download failed ({exc}). Falling back to embedded demonstration data.")
            prices = fallback_prices()
        try:
            events = detect_fed_events(fetch_fred_dff())
        except Exception as exc:
            print(f"FRED download failed ({exc}). Falling back to embedded Fed policy event table.")
            events = fallback_events()

    prices = prices.loc[prices.index >= "2018-01-01"].copy()
    events = events[(events["event_date"] >= prices.index.min()) & (events["event_date"] <= prices.index.max())].copy()

    event_study = build_event_study(prices, events)
    if event_study.empty:
        raise SystemExit("No complete event windows available. Try --offline or extend data history.")

    scores = score_sectors(event_study)
    allocations = {
        "Pre-Cut Allocation": build_allocation(scores, "Cut", "Pre-45D"),
        "Post-Cut Allocation": build_allocation(scores, "Cut", "Post-45D"),
        "Post-Hike Allocation": build_allocation(scores, "Hike", "Post-45D"),
    }

    scores_path = output_dir / "titan_sector_rotation_scores.csv"
    event_path = output_dir / "titan_fed_event_study.csv"
    workbook_path = output_dir / "Titan_Markets_Sector_Rotation_Model_v1.xlsx"
    pdf_path = output_dir / "Titan_Markets_Fed_Pivot_Playbook.pdf"

    scores.to_csv(scores_path, index=False)
    event_study.to_csv(event_path, index=False)
    create_excel_dashboard(workbook_path, scores, event_study, allocations, events)
    create_pdf_playbook(pdf_path, scores, allocations, events, prices.index.max())

    print("Created:")
    print(f"  {workbook_path}")
    print(f"  {pdf_path}")
    print(f"  {scores_path}")
    print(f"  {event_path}")


if __name__ == "__main__":
    main()
