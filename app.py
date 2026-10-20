"""
Forex Scaling Trading for an ADHD Mind
50-Trade Journaling & Analytics Dashboard

Run with:   streamlit run app.py
Requires:   pip install streamlit plotly pandas numpy openpyxl
            (openpyxl is only needed if you upload Excel files)

Optional (recommended) - force Streamlit's native dark theme by creating
.streamlit/config.toml next to this file:

    [theme]
    base = "dark"
    primaryColor = "#00e676"
    backgroundColor = "#0b0f19"
    secondaryBackgroundColor = "#131a2b"
    textColor = "#e8eaf0"

The script also injects its own dark CSS, so it looks right without it.
"""

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Scaling Journal | ADHD Mind",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS & STRATEGY RULES
# ──────────────────────────────────────────────────────────────────────────────
START_BALANCE = 100.0          # starting account capital ($)
RISK_THRESHOLD = 1000.0        # equity level where risk scales down
RISK_HIGH, RISK_LOW = 5.0, 2.0  # % risk: below / above the threshold
TARGET_TRADES = 50             # length of the journaling challenge
PKT = timezone(timedelta(hours=5), "PKT")  # Pakistan Standard Time (UTC+5)

SESSION_START_H, SESSION_END_H = 17, 21    # 5:00 PM -> 9:00 PM PKT

# Palette
GREEN, RED, AMBER, BLUE = "#00e676", "#ff5252", "#ffb300", "#4fc3f7"
BG, PANEL, TEXT, GRID = "#0b0f19", "#131a2b", "#e8eaf0", "rgba(255,255,255,0.07)"

# Canonical outcome labels used in the data schema
OUT_WIN = "Win (+2R to +3R)"
OUT_LOSS = "Loss (-1R)"
OUT_BE = "Trailed/BE"
OUTCOMES = [OUT_WIN, OUT_LOSS, OUT_BE]

# Short internal labels used by the charts
RESULT_ORDER = ["Win", "Trailed/BE", "Loss"]
RESULT_COLORS = {"Win": GREEN, "Trailed/BE": AMBER, "Loss": RED}
RESULT_LABELS = {
    "Win": "Full Wins (1:2 to 1:3)",
    "Trailed/BE": "Trailed / Break-even",
    "Loss": "Full Losses (-1R)",
}

SENTIMENTS = ["Bullish (Above SMA 50)", "Bearish (Below SMA 50)"]

# The journal schema - every column the dashboard tracks
COLUMNS = [
    "Trade_Num", "Date_Time", "Asset", "Sentiment_1D", "Zone_15M",
    "Entry_Price", "Stop_Loss", "Take_Profit", "Position_Risk_Pct",
    "Realized_RR", "Outcome", "PnL_Dollar", "Ending_Balance",
    "Checklist_Completed", "Notes_FOMO_Mindset",
]
NUMERIC_COLS = [
    "Trade_Num", "Entry_Price", "Stop_Loss", "Take_Profit",
    "Position_Risk_Pct", "Realized_RR", "PnL_Dollar", "Ending_Balance",
]
TEXT_COLS = ["Asset", "Sentiment_1D", "Zone_15M", "Outcome", "Notes_FOMO_Mindset"]


def _streamlit_version() -> tuple:
    """Parse st.__version__ into a (major, minor) tuple."""
    parts = []
    for p in st.__version__.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


# Streamlit >= 1.50 replaced `use_container_width=True` with `width="stretch"`.
STRETCH = (
    {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL STYLING (dark theme CSS)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    .stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #14203a 0%, {BG} 55%); color: {TEXT}; }}
    [data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid rgba(255,255,255,0.06); }}
    [data-testid="stHeader"] {{ background: transparent; }}
    h1, h2, h3, h4, label, p, span, div {{ color: {TEXT}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }}

    /* KPI cards */
    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, #172037 0%, #101728 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    }}
    div[data-testid="stMetricLabel"] p {{ color: #9aa4bf !important; font-size: 0.85rem; letter-spacing: .04em; text-transform: uppercase; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.9rem; font-weight: 700; }}

    .app-title {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 0;
        background: linear-gradient(90deg, {GREEN}, {BLUE}); -webkit-background-clip: text;
        -webkit-text-fill-color: transparent; }}
    .app-sub {{ color: #9aa4bf !important; margin-top: 2px; margin-bottom: 1.2rem; }}
    .section-title {{ font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .3rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA LAYER: sample generator, normaliser, finance rebuilder
# ──────────────────────────────────────────────────────────────────────────────
def auto_risk_pct(balance: float) -> float:
    """Tiered risk rule: 5% until the account reaches $1,000, then 2%."""
    return RISK_HIGH if balance < RISK_THRESHOLD else RISK_LOW


def classify_result(outcome, rr) -> str:
    """Map a free-text Outcome (or the realized RR as a fallback) to Win / Trailed/BE / Loss."""
    s = str(outcome).strip().lower()
    if "loss" in s:
        return "Loss"
    if "win" in s:
        return "Win"
    if "trail" in s or "break" in s or s.endswith("be"):
        return "Trailed/BE"
    # Fallback: derive from the realized RR
    if pd.notna(rr):
        if rr >= 1.5:
            return "Win"
        if rr <= -0.9:
            return "Loss"
    return "Trailed/BE"


def generate_sample_data(n: int = TARGET_TRADES, seed: int = 16) -> pd.DataFrame:
    """Build a synthetic, rule-consistent 50-trade journal that compounds from $100.

    Rules reflected: weekdays only, max 2 trades/day, 5-9 PM PKT window (a few
    deliberate rule-breaks), 5% risk until $1,000 then 2%, wins realise 2R-3R.
    """
    rng = np.random.default_rng(seed)

    # asset: (base price, min stop distance, max stop distance, decimals)
    assets = {
        "EURUSD": (1.0850, 0.0016, 0.0030, 5),
        "GBPUSD": (1.2900, 0.0020, 0.0038, 5),
        "XAUUSD": (2650.0, 8.0, 15.0, 2),
        "USDJPY": (150.20, 0.25, 0.45, 3),
        "AUDUSD": (0.6600, 0.0012, 0.0024, 5),
    }
    names = list(assets)
    weights = [0.40, 0.20, 0.20, 0.12, 0.08]

    notes_good_win = [
        "Waited for full checklist. Calm, trailed per structure.",
        "Respected the 1-open-trade rule. No FOMO, let it run.",
        "Patient on the retest. Did not touch the stop.",
        "Followed the plan, closed the laptop after TP.",
    ]
    notes_good_loss = [
        "Clean stop-out. Plan followed, no revenge trade.",
        "Valid setup, market said no. Moved on.",
        "Took the -1R calmly. Journaled before next setup.",
    ]
    notes_good_be = [
        "Trailed under the 15M pivot, got tapped at BE. Fine.",
        "Structure trail did its job. Small win, no stress.",
    ]
    notes_bad = [
        "FOMO - entered before the retest candle closed.",
        "Chased the breakout, skipped the 5M rejection.",
        "Revenge entry right after the last loss.",
        "Bored and impatient, forced a setup.",
    ]

    rows, balance, day = [], START_BALANCE, date(2026, 8, 31)
    while len(rows) < n:
        if day.weekday() < 5:  # Monday-Friday
            k = int(rng.choice([0, 1, 2], p=[0.2, 0.4, 0.4]))  # max 2 trades/day
            minutes = []
            for _ in range(k):
                if rng.random() < 0.08:  # occasional trade outside the session window
                    m = int(rng.choice([15 * 60 + int(rng.integers(30, 59)), 21 * 60 + int(rng.integers(10, 50))]))
                else:
                    m = int(rng.integers(SESSION_START_H * 60, SESSION_END_H * 60))
                minutes.append(m)
            for m in sorted(minutes):
                if len(rows) >= n:
                    break
                outside = not (SESSION_START_H * 60 <= m < SESSION_END_H * 60)
                checklist = (rng.random() > 0.12) and not outside

                asset = str(rng.choice(names, p=weights))
                base, lo, hi, dec = assets[asset]
                bullish = bool(rng.random() < 0.5)
                direction = 1 if bullish else -1
                entry = base + rng.normal(0, base * 0.01)
                dist = rng.uniform(lo, hi)
                sl = entry - direction * dist
                tp = entry + direction * dist * 3  # planned 1:3, wins realise 2R-3R via trail

                probs = [0.48, 0.34, 0.18] if checklist else [0.25, 0.55, 0.20]
                outcome = str(rng.choice(OUTCOMES, p=probs))
                if outcome == OUT_WIN:
                    rr = round(float(rng.uniform(2.0, 3.0)), 2)
                elif outcome == OUT_LOSS:
                    rr = -1.0
                else:
                    rr = 0.0 if rng.random() < 0.4 else round(float(rng.uniform(0.05, 0.6)), 2)

                risk = auto_risk_pct(balance)
                pnl = round(balance * risk / 100 * rr, 2)
                balance = round(balance + pnl, 2)

                if not checklist:
                    note = ("Outside the 5-9 PM window - impatience. " if outside else "") + str(rng.choice(notes_bad))
                elif outcome == OUT_WIN:
                    note = str(rng.choice(notes_good_win))
                elif outcome == OUT_LOSS:
                    note = str(rng.choice(notes_good_loss))
                else:
                    note = str(rng.choice(notes_good_be))

                zone = (
                    f"15M breakout above {round(entry - direction * dist * 0.1, dec)} -> retest hold, 5M rejection"
                    if bullish
                    else f"15M breakdown below {round(entry + dist * 0.1, dec)} -> retest fail, 5M rejection"
                )
                ts = datetime(day.year, day.month, day.day, m // 60, m % 60)
                rows.append(
                    {
                        "Trade_Num": len(rows) + 1,
                        "Date_Time": ts,
                        "Asset": asset,
                        "Sentiment_1D": SENTIMENTS[0] if bullish else SENTIMENTS[1],
                        "Zone_15M": zone,
                        "Entry_Price": round(entry, dec),
                        "Stop_Loss": round(sl, dec),
                        "Take_Profit": round(tp, dec),
                        "Position_Risk_Pct": risk,
                        "Realized_RR": rr,
                        "Outcome": outcome,
                        "PnL_Dollar": pnl,
                        "Ending_Balance": balance,
                        "Checklist_Completed": bool(checklist),
                        "Notes_FOMO_Mindset": note,
                    }
                )
        day += timedelta(days=1)

    return normalize_journal(pd.DataFrame(rows))


def _to_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "y", "1", "t", "✅"}
    if pd.isna(v):
        return False
    return bool(v)


def normalize_journal(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce any uploaded / generated frame into the journal schema.

    - matches column names case-insensitively
    - adds missing columns, coerces types, parses timestamps (-> naive PKT)
    - fills missing PnL / balance / risk using the tiered-risk rules
    - adds an internal `Result` column (Win / Trailed/BE / Loss) for the charts
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    df = df.rename(columns={lookup[c.lower()]: c for c in COLUMNS if c.lower() in lookup})

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    # Types
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TEXT_COLS:
        df[c] = df[c].fillna("").astype(str)
    df["Checklist_Completed"] = df["Checklist_Completed"].apply(_to_bool)

    # Timestamps -> naive datetimes expressed in PKT
    try:
        dt = pd.to_datetime(df["Date_Time"], errors="coerce")
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("Asia/Karachi").dt.tz_localize(None)
    except (AttributeError, TypeError, ValueError):
        dt = pd.to_datetime(df["Date_Time"], errors="coerce", utc=True)
        dt = dt.dt.tz_convert("Asia/Karachi").dt.tz_localize(None)
    df["Date_Time"] = dt

    # Trade numbering / ordering
    positional = pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float)
    df["Trade_Num"] = df["Trade_Num"].fillna(positional).astype(int)
    df = df.sort_values("Trade_Num", kind="stable").reset_index(drop=True)

    # Outcome / Result
    df["Realized_RR"] = df["Realized_RR"].fillna(0.0)
    df["Result"] = [classify_result(o, r) for o, r in zip(df["Outcome"], df["Realized_RR"])]
    canon = {"Win": OUT_WIN, "Loss": OUT_LOSS, "Trailed/BE": OUT_BE}
    df["Outcome"] = df["Result"].map(canon)

    # Finances: fill gaps sequentially so balances always compound correctly
    balance, risks, pnls, ends = START_BALANCE, [], [], []
    for _, r in df.iterrows():
        risk = r["Position_Risk_Pct"] if pd.notna(r["Position_Risk_Pct"]) else auto_risk_pct(balance)
        pnl = r["PnL_Dollar"] if pd.notna(r["PnL_Dollar"]) else round(balance * risk / 100 * r["Realized_RR"], 2)
        end = r["Ending_Balance"] if pd.notna(r["Ending_Balance"]) else round(balance + pnl, 2)
        risks.append(float(risk)); pnls.append(float(pnl)); ends.append(float(end))
        balance = end
    df["Position_Risk_Pct"], df["PnL_Dollar"], df["Ending_Balance"] = risks, pnls, ends

    return df[COLUMNS + ["Result"]]


def read_uploaded(file) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file into a normalised journal."""
    if file.name.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(file)  # needs `openpyxl` for .xlsx
    else:
        raw = pd.read_csv(file)
    return normalize_journal(raw)


# ──────────────────────────────────────────────────────────────────────────────
# 4. ANALYTICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def equity_series(df: pd.DataFrame):
    """Return (equity, running_peak, drawdown_pct), each starting at trade 0 ($100)."""
    eq = np.concatenate([[START_BALANCE], df["Ending_Balance"].to_numpy(dtype=float)])
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak - 1.0) * 100.0
    return eq, peak, dd


def time_slot(ts) -> str | None:
    """30-minute slot label inside the 5-9 PM window, else 'Outside window'."""
    if pd.isna(ts):
        return None
    m = ts.hour * 60 + ts.minute
    if SESSION_START_H * 60 <= m < SESSION_END_H * 60:
        start = (m // 30) * 30
        return f"{start // 60:02d}:{start % 60:02d}"
    return "Outside window"


SLOTS = [f"{h:02d}:{m:02d}" for h in range(SESSION_START_H, SESSION_END_H) for m in (0, 30)] + ["Outside window"]


def fmt_price(v) -> str:
    """Price formatter that adapts to the asset (1.08512 / 2650.12 / 150.231)."""
    if pd.isna(v):
        return ""
    return f"{v:.5f}".rstrip("0").rstrip(".")


def fmt_r(v) -> str:
    return "-" if pd.isna(v) else f"{v:+.2f}R"


def style_figure(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    """Apply the shared dark look to every chart."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.01, font=dict(size=17, color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19,26,43,0.55)",
        font=dict(color=TEXT),
        height=height,
        margin=dict(l=20, r=20, t=60, b=30),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig: go.Figure, key: str):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **STRETCH)


# ──────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR: data source + manual trade entry
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Journal Controls")

source = st.sidebar.radio(
    "Data source",
    ["Built-in 50-trade sample", "Upload CSV / Excel log"],
    help="The sample is synthetic demo data. Upload your own log to replace it.",
)

uploaded = None
if source == "Upload CSV / Excel log":
    uploaded = st.sidebar.file_uploader(
        "Trade log file", type=["csv", "xlsx", "xls"],
        help="Columns are matched by name (case-insensitive). Missing columns are filled in.",
    )

# Work out which dataset should currently be loaded; reload only when it changes
# so manual entries survive Streamlit reruns.
if source == "Built-in 50-trade sample":
    wanted_id = "sample"
elif uploaded is not None:
    wanted_id = f"upload:{uploaded.name}:{uploaded.size}"
else:
    wanted_id = "empty"

if st.sidebar.button("↺ Reset journal to source", **STRETCH):
    st.session_state.pop("source_id", None)

if st.session_state.get("source_id") != wanted_id:
    try:
        if wanted_id == "sample":
            st.session_state.journal = generate_sample_data()
        elif wanted_id == "empty":
            st.session_state.journal = normalize_journal(pd.DataFrame(columns=COLUMNS))
        else:
            st.session_state.journal = read_uploaded(uploaded)
        st.session_state.source_id = wanted_id
    except Exception as exc:  # bad file, missing openpyxl, etc.
        st.sidebar.error(f"Could not read that file: {exc}")
        st.session_state.journal = normalize_journal(pd.DataFrame(columns=COLUMNS))
        st.session_state.source_id = wanted_id

journal: pd.DataFrame = st.session_state.journal
current_balance = float(journal["Ending_Balance"].iloc[-1]) if len(journal) else START_BALANCE

st.sidebar.markdown("---")
with st.sidebar.expander("➕ Log a new trade", expanded=False):
    st.caption(f"Account now **${current_balance:,.2f}** → default risk **{auto_risk_pct(current_balance):.0f}%**")
    now_pkt = datetime.now(PKT)
    with st.form("trade_form", clear_on_submit=True):
        f_date = st.date_input("Date (PKT)", value=now_pkt.date())
        f_time = st.time_input("Time (PKT)", value=time(max(SESSION_START_H, min(now_pkt.hour, 20)), now_pkt.minute))
        f_asset = st.text_input("Asset", value="EURUSD")
        f_sent = st.selectbox("1D Sentiment", SENTIMENTS)
        f_zone = st.text_input("15M Zone / POI", placeholder="15M breakout above 1.0850 → retest hold")
        c1, c2, c3 = st.columns(3)
        f_entry = c1.number_input("Entry", value=1.08500, format="%.5f", step=0.0001)
        f_sl = c2.number_input("Stop", value=1.08300, format="%.5f", step=0.0001)
        f_tp = c3.number_input("Target", value=1.09100, format="%.5f", step=0.0001)
        f_risk = st.number_input(
            "Position risk %", min_value=0.1, max_value=10.0,
            value=auto_risk_pct(current_balance), step=0.5,
            help="5% below $1,000 equity, 2% above.",
        )
        f_outcome = st.selectbox("Outcome", OUTCOMES)
        f_rr = st.number_input(
            "Realized RR (0 = auto)", value=0.0, step=0.1, format="%.2f",
            help="Auto: Win = planned RR from your prices, Loss = -1R, Trailed/BE = 0R.",
        )
        f_check = st.checkbox("Checklist completed", value=True)
        f_notes = st.text_area("Notes / FOMO / mindset", placeholder="How did you feel? Which rules did you respect?")
        submitted = st.form_submit_button("Add trade to journal", **STRETCH)

    if submitted:
        risk_dist = abs(f_entry - f_sl)
        if risk_dist == 0:
            st.error("Entry and Stop can't be equal.")
        else:
            planned_rr = abs(f_tp - f_entry) / risk_dist
            rr = f_rr
            if rr == 0 and f_outcome == OUT_WIN:
                rr = round(planned_rr, 2)
            elif rr == 0 and f_outcome == OUT_LOSS:
                rr = -1.0

            if f_outcome == OUT_WIN and rr <= 0:
                st.error("A win needs a positive RR.")
            elif f_outcome == OUT_LOSS and rr > 0:
                st.error("A loss can't have a positive RR.")
            else:
                pnl = round(current_balance * f_risk / 100 * rr, 2)
                new_row = {
                    "Trade_Num": int(journal["Trade_Num"].max() + 1) if len(journal) else 1,
                    "Date_Time": datetime.combine(f_date, f_time),
                    "Asset": f_asset.strip().upper() or "N/A",
                    "Sentiment_1D": f_sent,
                    "Zone_15M": f_zone,
                    "Entry_Price": f_entry, "Stop_Loss": f_sl, "Take_Profit": f_tp,
                    "Position_Risk_Pct": f_risk,
                    "Realized_RR": rr,
                    "Outcome": f_outcome,
                    "PnL_Dollar": pnl,
                    "Ending_Balance": round(current_balance + pnl, 2),
                    "Checklist_Completed": f_check,
                    "Notes_FOMO_Mindset": f_notes,
                }
                st.session_state.journal = normalize_journal(
                    pd.concat([journal.drop(columns=["Result"]), pd.DataFrame([new_row])], ignore_index=True)
                )
                journal = st.session_state.journal
                current_balance = float(journal["Ending_Balance"].iloc[-1])
                st.success(f"Trade #{int(new_row['Trade_Num'])} logged ({fmt_r(rr)}).")

# ──────────────────────────────────────────────────────────────────────────────
# 6. HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">Forex Scaling Trading for an ADHD Mind</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-sub">50-Trade Journaling & Analytics Dashboard · Daily SMA 50 bias → 15M breakout-retest · '
    "5–9 PM PKT · 5% risk until $1,000, then 2%</p>",
    unsafe_allow_html=True,
)

if source == "Built-in 50-trade sample":
    st.caption("📌 Showing synthetic sample data - switch the data source in the sidebar to use your own log.")

if journal.empty:
    st.info("No trades yet. Upload a log or add your first trade from the sidebar (➕ Log a new trade).")
    st.stop()

# Core numbers used across the dashboard
df = journal.copy()
n_trades = len(df)
eq, peak, dd = equity_series(df)
final_balance = float(eq[-1])
net_dollar = final_balance - START_BALANCE
net_pct = net_dollar / START_BALANCE * 100

wins = df[df["Result"] == "Win"]
losses = df[df["Result"] == "Loss"]
trails = df[df["Result"] == "Trailed/BE"]
win_rate = len(wins) / n_trades * 100
avg_win_rr = wins["Realized_RR"].mean() if len(wins) else float("nan")
avg_loss_rr = losses["Realized_RR"].mean() if len(losses) else float("nan")
expectancy = df["Realized_RR"].mean()
max_dd = abs(dd.min())

st.progress(min(n_trades / TARGET_TRADES, 1.0), text=f"Journal progress: {n_trades} / {TARGET_TRADES} trades")

# ──────────────────────────────────────────────────────────────────────────────
# 7. KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Net Return",
    f"${net_dollar:+,.2f}",
    f"{net_pct:+,.1f}%  ·  balance ${final_balance:,.2f}",
    help=f"Starting capital ${START_BALANCE:,.0f}.",
)
k2.metric(
    "Win Rate",
    f"{win_rate:.1f}%",
    f"{len(wins)}W · {len(trails)}BE · {len(losses)}L of {n_trades}",
    delta_color="off",
    help="Full wins ÷ all executed trades. Trailed/BE trades count as non-wins.",
)
k3.metric(
    "Avg RR (Win / Loss)",
    f"{fmt_r(avg_win_rr)} / {fmt_r(avg_loss_rr)}",
    f"Expectancy {expectancy:+.2f}R per trade",
    help="Average realized RR on winning trades vs. losing trades. Trailed/BE trades are excluded from both "
    "averages but included in expectancy.",
)
k4.metric(
    "Max Drawdown",
    f"-{max_dd:.1f}%",
    f"Now {dd[-1]:.1f}% from peak",
    delta_color="off",
    help="Largest peak-to-trough decline of account equity.",
)

# ──────────────────────────────────────────────────────────────────────────────
# 8. CHART 1 - COMPOUNDING EQUITY CURVE vs TARGET BASELINE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Compounding Equity Curve</div>', unsafe_allow_html=True)
log_scale = st.toggle("Log scale (best for compounding)", value=False, key="log_toggle")

x_eq = np.arange(len(eq))  # 0 = starting balance, 1..n = after each trade
x_max = max(TARGET_TRADES, n_trades)
x_target = np.arange(0, x_max + 1)
y_target = START_BALANCE * (RISK_THRESHOLD / START_BALANCE) ** (x_target / TARGET_TRADES)

y_top = max(float(eq.max()), RISK_THRESHOLD) * 1.10
y_bottom = min(float(eq.min()), START_BALANCE) * 0.80

fig1 = go.Figure()

# Shaded risk zones (5% zone below $1,000, 2% zone above)
fig1.add_hrect(y0=y_bottom, y1=RISK_THRESHOLD, fillcolor="rgba(255,179,0,0.05)", line_width=0, layer="below")
fig1.add_hrect(y0=RISK_THRESHOLD, y1=y_top, fillcolor="rgba(0,230,118,0.05)", line_width=0, layer="below")

# Target baseline: ideal geometric growth $100 -> $1,000 over 50 trades
fig1.add_trace(go.Scatter(
    x=x_target, y=y_target, mode="lines", name="Target path ($100 → $1,000)",
    line=dict(color=BLUE, width=2, dash="dot"), hovertemplate="Target: $%{y:,.2f}<extra></extra>",
))

# Invisible running-peak trace; the equity trace fills down to it in red -> drawdown "dips"
fig1.add_trace(go.Scatter(x=x_eq, y=peak, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
fig1.add_trace(go.Scatter(
    x=x_eq, y=eq, mode="lines", name="Equity",
    line=dict(color=GREEN, width=3, shape="spline", smoothing=0.3),
    fill="tonexty", fillcolor="rgba(255,82,82,0.32)", hoverinfo="skip",
))
# Legend key for the red fill
fig1.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers", name="Drawdown from peak",
    marker=dict(size=11, color="rgba(255,82,82,0.6)", symbol="square"),
))

# Trade markers (carry the hover details)
hover = (
    "<b>Trade #%{customdata[0]}</b> · %{customdata[1]}<br>%{customdata[2]}<br>"
    "RR: %{customdata[3]:+.2f}R · P&L: $%{customdata[4]:+,.2f}<br>"
    "Balance: $%{y:,.2f}<extra></extra>"
)
fig1.add_trace(go.Scatter(
    x=np.arange(1, n_trades + 1),  # position in the journal (matches the equity series index)
    y=df["Ending_Balance"],
    mode="markers", name="Trades",
    marker=dict(size=8, color=[RESULT_COLORS[r] for r in df["Result"]], line=dict(width=1, color=BG)),
    customdata=np.stack([df["Trade_Num"], df["Asset"], df["Outcome"], df["Realized_RR"], df["PnL_Dollar"]], axis=-1),
    hovertemplate=hover,
))

# $1,000 threshold: risk steps down 5% -> 2%
fig1.add_hline(
    y=RISK_THRESHOLD, line=dict(color=AMBER, width=2, dash="dash"),
    annotation_text="$1,000 threshold · risk scales 5% → 2%", annotation_position="top left",
    annotation_font=dict(color=AMBER),
)
crossed = np.where(eq >= RISK_THRESHOLD)[0]
if len(crossed):
    fig1.add_vline(
        x=int(crossed[0]), line=dict(color=AMBER, width=1, dash="dot"),
        annotation_text=f"Crossed at trade {int(crossed[0])}", annotation_position="bottom right",
        annotation_font=dict(color=AMBER),
    )

style_figure(fig1, "Account Equity vs. Target Baseline", height=480)
fig1.update_xaxes(title="Trade number", range=[0, x_max + 1])
if log_scale:
    fig1.update_yaxes(title="Equity ($, log)", type="log", tickprefix="$")
else:
    fig1.update_yaxes(title="Equity ($)", range=[y_bottom, y_top], tickprefix="$")
show(fig1, "chart_equity")

# ──────────────────────────────────────────────────────────────────────────────
# 9. CHART 2 (underwater drawdown) + CHART 3 (result & RR distribution)
# ──────────────────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x_eq, y=dd, mode="lines", name="Drawdown",
        line=dict(color=RED, width=2), fill="tozeroy", fillcolor="rgba(255,82,82,0.35)",
        hovertemplate="After trade %{x}: %{y:.2f}% below peak<extra></extra>",
    ))
    if max_dd > 0:
        worst = int(np.argmin(dd))
        fig2.add_trace(go.Scatter(
            x=[worst], y=[dd[worst]], mode="markers+text", name="Max drawdown",
            marker=dict(size=10, color=AMBER, line=dict(width=1, color=BG)),
            text=[f"Max DD -{max_dd:.1f}%"], textposition="bottom right", textfont=dict(color=AMBER),
            hoverinfo="skip",
        ))
    style_figure(fig2, "Underwater Chart - Drawdown from Equity Peak", height=400)
    fig2.update_xaxes(title="Trade number", range=[0, x_max + 1])
    fig2.update_yaxes(title="Drawdown", ticksuffix="%", range=[min(dd.min() * 1.3, -1.0), 1.0])
    fig2.update_layout(showlegend=False)
    show(fig2, "chart_underwater")

with col_b:
    counts = df["Result"].value_counts().reindex(RESULT_ORDER, fill_value=0)
    fig3 = make_subplots(
        rows=1, cols=2, column_widths=[0.42, 0.58],
        subplot_titles=("Outcome mix", "Realized RR distribution"),
    )
    fig3.add_trace(go.Bar(
        x=[RESULT_LABELS[r] for r in RESULT_ORDER], y=counts.values,
        marker_color=[RESULT_COLORS[r] for r in RESULT_ORDER],
        text=[f"{v}<br>{v / n_trades * 100:.0f}%" for v in counts.values], textposition="outside",
        showlegend=False, hovertemplate="%{x}: %{y} trades<extra></extra>",
    ), row=1, col=1)
    for r in RESULT_ORDER:
        sub = df[df["Result"] == r]
        if len(sub):
            fig3.add_trace(go.Histogram(
                x=sub["Realized_RR"], name=RESULT_LABELS[r], marker_color=RESULT_COLORS[r],
                xbins=dict(size=0.25), opacity=0.9,
                hovertemplate="RR %{x}: %{y} trades<extra></extra>",
            ), row=1, col=2)
    style_figure(fig3, "Trade Result & RR Distribution", height=400)
    fig3.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"))
    fig3.update_yaxes(title_text="Trades", row=1, col=1, range=[0, max(counts.max() * 1.25, 1)])
    fig3.update_xaxes(tickangle=-20, row=1, col=1)
    fig3.update_xaxes(title_text="Realized RR (R)", row=1, col=2)
    show(fig3, "chart_results")

# ──────────────────────────────────────────────────────────────────────────────
# 10. CHART 4 - SESSION & TIME DISTRIBUTION (5-9 PM PKT)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🕔 Session & Time Distribution (5:00 – 9:00 PM PKT)</div>', unsafe_allow_html=True)
timed = df.dropna(subset=["Date_Time"]).copy()

if timed.empty:
    st.info("No valid timestamps found in Date_Time, so the session charts are unavailable.")
else:
    timed["Slot"] = timed["Date_Time"].apply(time_slot)
    timed["Hour_Decimal"] = timed["Date_Time"].dt.hour + timed["Date_Time"].dt.minute / 60

    col_c, col_d = st.columns(2)

    with col_c:
        grid = pd.MultiIndex.from_product([SLOTS, RESULT_ORDER], names=["Slot", "Result"])
        slot_counts = timed.groupby(["Slot", "Result"]).size().reindex(grid, fill_value=0).reset_index(name="Trades")
        fig4a = px.bar(
            slot_counts, x="Slot", y="Trades", color="Result",
            color_discrete_map=RESULT_COLORS, category_orders={"Slot": SLOTS, "Result": RESULT_ORDER},
        )
        style_figure(fig4a, "Execution Frequency by 30-min Slot", height=400)
        fig4a.update_xaxes(title="Slot start (PKT)")
        fig4a.update_yaxes(title="Trades")
        fig4a.update_layout(legend_title_text="")
        show(fig4a, "chart_slots")

    with col_d:
        fig4b = px.scatter(
            timed, x="Hour_Decimal", y="Realized_RR", color="Result",
            color_discrete_map=RESULT_COLORS, category_orders={"Result": RESULT_ORDER},
            hover_data={"Trade_Num": True, "Asset": True, "Hour_Decimal": False, "Realized_RR": ":+.2f"},
        )
        fig4b.update_traces(marker=dict(size=11, line=dict(width=1, color=BG)))
        fig4b.add_vrect(
            x0=SESSION_START_H, x1=SESSION_END_H, fillcolor="rgba(79,195,247,0.09)", line_width=0,
            annotation_text="Session window", annotation_position="top left", annotation_font=dict(color=BLUE),
        )
        fig4b.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"))
        style_figure(fig4b, "Time of Day vs. Realized RR", height=400)
        fig4b.update_xaxes(
            title="Time of day (PKT)", range=[15.5, 22.5],
            tickvals=list(range(16, 23)), ticktext=[f"{h - 12 if h > 12 else h}:00 {'PM' if h >= 12 else 'AM'}" for h in range(16, 23)],
        )
        fig4b.update_yaxes(title="Realized RR (R)", ticksuffix="R")
        fig4b.update_layout(legend_title_text="")
        show(fig4b, "chart_time_scatter")

# ──────────────────────────────────────────────────────────────────────────────
# 11. DISCIPLINE & TRADE-MANAGEMENT ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 Discipline & Trade-Management Analytics</div>', unsafe_allow_html=True)
chk_yes = df[df["Checklist_Completed"]]
chk_no = df[~df["Checklist_Completed"]]
outside_n = int((timed["Slot"] == "Outside window").sum()) if not timed.empty else 0
emotion_mask = df["Notes_FOMO_Mindset"].str.contains(r"fomo|revenge|chase|chased|bored|impatien|forced", case=False, regex=True)

d1, d2, d3, d4 = st.columns(4)
d1.metric("Checklist compliance", f"{len(chk_yes) / n_trades * 100:.0f}%", f"{len(chk_no)} trades skipped checklist", delta_color="off")
d2.metric(
    "Avg RR · checklist ✅ vs ❌",
    f"{fmt_r(chk_yes['Realized_RR'].mean() if len(chk_yes) else float('nan'))} / "
    f"{fmt_r(chk_no['Realized_RR'].mean() if len(chk_no) else float('nan'))}",
    help="Average realized RR when the pre-trade checklist was completed vs. skipped.",
)
d3.metric("Outside 5-9 PM window", f"{outside_n}", "trades broke the session rule" if outside_n else "all trades inside window", delta_color="off")
d4.metric(
    "Emotion-flagged notes", f"{int(emotion_mask.sum())}",
    "keyword scan: FOMO / revenge / chase / bored", delta_color="off",
)

# ──────────────────────────────────────────────────────────────────────────────
# 12. INTERACTIVE JOURNAL TABLE + CSV EXPORT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📒 Trade Journal</div>', unsafe_allow_html=True)

display = df[COLUMNS].copy()
display["Date_Time"] = display["Date_Time"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
display["Checklist_Completed"] = display["Checklist_Completed"].map({True: "✅", False: "❌"})
row_bg = {
    "Win": "rgba(0,230,118,0.14)",
    "Loss": "rgba(255,82,82,0.16)",
    "Trailed/BE": "rgba(255,179,0,0.13)",
}
results = df["Result"].reset_index(drop=True)


def color_rows(row: pd.Series):
    """Tint each row by trade result (green win / red loss / amber trail-BE)."""
    return [f"background-color: {row_bg[results.iloc[row.name]]}"] * len(row)


styled = (
    display.style.apply(color_rows, axis=1)
    .format(
        {
            "Entry_Price": fmt_price, "Stop_Loss": fmt_price, "Take_Profit": fmt_price,
            "Position_Risk_Pct": "{:.1f}%",
            "Realized_RR": "{:+.2f}R",
            "PnL_Dollar": "${:+,.2f}",
            "Ending_Balance": "${:,.2f}",
        }
    )
)
st.dataframe(styled, hide_index=True, height=min(60 + 35 * n_trades, 640), **STRETCH)

# Clean CSV export (no internal helper columns, timestamps as plain text)
export_df = df[COLUMNS].copy()
export_df["Date_Time"] = export_df["Date_Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download trade journal (CSV)",
    data=csv_bytes,
    file_name=f"forex_scaling_journal_{datetime.now(PKT):%Y%m%d}.csv",
    mime="text/csv",
)
