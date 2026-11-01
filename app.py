"""Retail Demand Forecasting - guided Streamlit app (CSE3141 Predictive Analytics mini project)
Flow: 1 Data -> 2 Clean -> 3 Explore -> 4 Models -> 5 Forecast  (+ Viva Help page)
"""
import itertools, warnings
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide")
SS = st.session_state

FREQ = {"Daily": ("D", 7), "Weekly": ("W", 52), "Monthly": ("MS", 12)}   # name -> (pandas rule, default season length)
DEFAULT_TEST = {"Daily": 60, "Weekly": 26, "Monthly": 24}
SRC_REAL = "Real data: US monthly retail sales (1992-2016)"
SRC_BIG = "Large store dataset: daily sales of 10 stores (87,660 rows)"
SRC_UP = "Upload my own CSV (e.g. from Kaggle)"
STEPS = ["1️⃣ Data", "2️⃣ Clean", "3️⃣ Explore", "4️⃣ Models", "5️⃣ Forecast", "📘 Viva Help"]


# ---------------------------------------------------------------- helpers
def tip(text):
    st.info("💡 **Simple explanation:** " + text)

def need(key, msg):
    if key not in SS:
        st.warning(msg); st.stop()

def reset_after(level):
    """Forget results of later steps when earlier data changes."""
    order = ["clean", "models"]
    for k in order[order.index(level):]:
        SS.pop(k, None)

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

def make_series(values, start, rule):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=rule), name="sales")

def pack(y):
    return tuple(np.asarray(y.values, dtype=float)), str(y.index[0].date())

@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path, parse_dates=["date"])

def line_fig(traces, height=380, ytitle="Sales"):
    fig = go.Figure()
    for name, s, style in traces:
        fig.add_scatter(x=s.index, y=s.values, name=name, **style)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=ytitle,
                      legend=dict(orientation="h", y=1.1))
    return fig


# ---------------------------------------------------------------- model code (cached)
@st.cache_data(show_spinner=False)
def run_models(values, start, rule, m, h):
    """Train 4 models on everything except the last h periods, test on the last h."""
    y = make_series(values, start, rule)
    train, test = y.iloc[:-h], y.iloc[-h:]
    preds, notes = {}, {}

    last = train.iloc[-m:].values                                   # 1. baseline: repeat last season
    preds["Seasonal Naive (baseline)"] = pd.Series([last[i % m] for i in range(h)], index=test.index)

    try:                                                            # 2. Holt-Winters (exponential smoothing)
        hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=m).fit()
        preds["Holt-Winters"] = pd.Series(hw.forecast(h).values, index=test.index)
    except Exception as e:
        notes["Holt-Winters"] = f"skipped ({type(e).__name__})"

    try:                                                            # 3. ARIMA - no seasonality
        ar = ARIMA(train, order=(2, 1, 2)).fit()
        preds["ARIMA(2,1,2)"] = pd.Series(ar.forecast(h).values, index=test.index)
    except Exception as e:
        notes["ARIMA"] = f"skipped ({type(e).__name__})"

    if m <= 12:                                                     # 4. SARIMA + grid search by AIC
        grid = list(itertools.product([0, 1], [0, 1, 2], [0, 1], [0, 1]))
    else:
        grid = [(1, 1, 0, 1), (0, 1, 0, 1), (1, 0, 0, 1)]
    best_aic, best = np.inf, None
    for p, q, P, Q in grid:
        try:
            r = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, m),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            if r.aic < best_aic:
                best_aic, best, best_fit = r.aic, (p, q, P, Q), r
        except Exception:
            pass
    sar_name = None
    train_rmse = None
    if best:
        p, q, P, Q = best
        sar_name = f"SARIMA({p},1,{q})x({P},1,{Q},{m})"
        preds[sar_name] = pd.Series(best_fit.forecast(h).values, index=test.index)
        fv = best_fit.fittedvalues.iloc[m + 1:]
        train_rmse = rmse(train.loc[fv.index], fv)

    rows = {}
    for name, p_ in preds.items():
        rows[name] = dict(RMSE=rmse(test, p_), MAE=float(mean_absolute_error(test, p_)),
                          MAPE=float(np.mean(np.abs((test.values - p_.values) / np.where(test.values == 0, np.nan, test.values))) * 100))
    res = pd.DataFrame(rows).T.sort_values("RMSE")
    return dict(res=res, preds=preds, test=test, train_tail=train.iloc[-min(len(train), 4 * m):],
                sarima=sar_name, sarima_cfg=best, aic=best_aic, train_rmse=train_rmse, notes=notes, h=h, m=m, rule=rule)


@st.cache_data(show_spinner=False)
def fit_forecast(values, start, rule, m, model_name, cfg, n):
    """Refit chosen model on ALL data and predict the next n periods."""
    y = make_series(values, start, rule)
    idx = pd.date_range(y.index[-1], periods=n + 1, freq=rule)[1:]
    lo = hi = None
    if model_name.startswith("SARIMA"):
        p, q, P, Q = cfg
        f = SARIMAX(y, order=(p, 1, q), seasonal_order=(P, 1, Q, m),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).get_forecast(n)
        mean, ci = f.predicted_mean.values, f.conf_int(alpha=0.05).values
        lo, hi = ci[:, 0], ci[:, 1]
    elif model_name.startswith("ARIMA"):
        f = ARIMA(y, order=(2, 1, 2)).fit().get_forecast(n)
        mean, ci = f.predicted_mean.values, f.conf_int(alpha=0.05).values
        lo, hi = ci[:, 0], ci[:, 1]
    elif model_name == "Holt-Winters":
        fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=m).fit()
        mean = fit.forecast(n).values
        band = 1.96 * float(np.std(fit.resid))
        lo, hi = mean - band, mean + band
    else:
        last = y.iloc[-m:].values
        mean = np.array([last[i % m] for i in range(n)])
    return pd.DataFrame({"forecast": mean, "lower": lo, "upper": hi}, index=idx)


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 Retail Demand Forecasting")
st.sidebar.caption("CSE3141 Predictive Analytics · mini project")
page = st.sidebar.radio("Go through the steps in order 👇", STEPS, key="page")
st.sidebar.divider()
if "work" in SS:
    st.sidebar.markdown(f"**Data loaded:** {len(SS['work'])} {SS['freq'].lower()} rows")
    st.sidebar.markdown(f"**Cleaned:** {'✅' if 'clean' in SS else '❌ not yet'}")
    st.sidebar.markdown(f"**Models trained:** {'✅' if 'models' in SS else '❌ not yet'}")
else:
    st.sidebar.markdown("No data loaded yet. Start with **Step 1**.")


# ================================================================= STEP 1
def step_data():
    st.header("Step 1 · Data")
    tip("Forecasting ke liye pehle sales ka purana data chahiye (date + sales). Yahan data choose karo, "
        "phir niche table mein **rows add / delete / edit** kar sakte ho.")
    src = st.radio("Choose data source", [SRC_REAL, SRC_BIG, SRC_UP], key="src")

    if src == SRC_REAL:
        raw = load_csv("data/us_retail_sales_monthly.csv"); dcol, vcol = "date", "sales"
        st.caption("Real monthly retail sales of the USA (US Census Bureau data, as shared in Facebook Prophet's example datasets).")
        dayfirst = False
    elif src == SRC_BIG:
        full = load_csv("data/large_store_sales.csv")
        st.caption("Same columns as Kaggle's *Store Sales* data (date, store, product family, sales, promotions, holiday). "
                   "Values are simulated, with missing values and outliers added on purpose so you can practise cleaning.")
        c1, c2 = st.columns(2)
        stores = c1.multiselect("Stores", sorted(full.store_nbr.unique()), default=[1])
        fams = c2.multiselect("Product families", sorted(full.family.unique()), default=["GROCERY"])
        raw = full[full.store_nbr.isin(stores) & full.family.isin(fams)]
        dcol, vcol, dayfirst = "date", "sales", False
        st.caption("Tip: choose 1 store + 1 family to see missing values/outliers in Step 2; choose many to see total sales.")
    else:
        up = st.file_uploader("Upload a CSV file (must have one date column and one sales column)", type="csv")
        if up is None:
            st.info("Upload a file to continue. Example: Kaggle *Walmart Sales Forecast* or *Store Sales - Time Series Forecasting* (train.csv)."); return
        raw = pd.read_csv(up)
        cols = list(raw.columns)
        guess_d = next((c for c in cols if "date" in c.lower()), cols[0])
        nums = [c for c in cols if pd.api.types.is_numeric_dtype(raw[c])]
        if not nums:
            st.error("No numeric column found for sales."); return
        guess_v = next((c for c in nums if any(k in c.lower() for k in ("sales", "revenue", "demand", "qty", "quantity"))), nums[-1])
        c1, c2, c3 = st.columns(3)
        dcol = c1.selectbox("Which column is the DATE?", cols, index=cols.index(guess_d))
        vcol = c2.selectbox("Which column is SALES?", nums, index=nums.index(guess_v))
        dayfirst = c3.checkbox("Dates are day-first (dd-mm-yyyy)", value=False)

    st.markdown(f"**Raw data:** {len(raw):,} rows × {raw.shape[1]} columns")
    st.dataframe(raw.head(100), height=220, width="stretch")
    if raw.empty:
        st.warning("Selection is empty - pick at least one store and one family."); return

    d = pd.to_datetime(raw[dcol], errors="coerce", dayfirst=dayfirst)
    ud = d.dropna().drop_duplicates().sort_values()      # unique dates only (many stores share the same date)
    gap = ud.diff().dropna().dt.days.median() if len(ud) > 2 else 1
    auto = "Daily" if gap <= 1 else "Weekly" if gap <= 8 else "Monthly"
    c1, c2 = st.columns(2)
    freq = c1.selectbox("How often is a sale recorded? (frequency)", list(FREQ), index=list(FREQ).index(auto),
                        help="Rows are added up (summed) into this frequency.")
    m = c2.number_input("Season length (how many rows before the pattern repeats?)", 2, 60, FREQ[freq][1],
                        help="Daily data repeats weekly -> 7. Monthly data repeats yearly -> 12. Weekly -> 52.")
    if st.button("✅ Load this data", type="primary"):
        t = pd.DataFrame({"date": d, "sales": pd.to_numeric(raw[vcol], errors="coerce")}).dropna(subset=["date"])
        s = t.groupby("date")["sales"].sum(min_count=1).resample(FREQ[freq][0]).sum(min_count=1)
        SS["work"] = s.rename("sales").reset_index()
        SS["freq"], SS["m"], SS["source"] = freq, int(m), src
        reset_after("clean")
        st.rerun()

    if "work" in SS:
        st.divider()
        st.subheader("Your working data (you can edit this)")
        tip("Ye wo table hai jispe aage ke saare steps chalenge. **Row delete:** left side checkbox tick karo → "
            "Delete dabao. **Row add:** table ke sabse neeche khali row mein date aur sales likho. "
            "**Value edit:** cell pe double-click. Phir **Save my edits** dabana mat bhoolna.")
        edited = st.data_editor(SS["work"], num_rows="dynamic", width="stretch", height=300, key="editor",
                                column_config={"date": st.column_config.DateColumn("Date"),
                                               "sales": st.column_config.NumberColumn("Sales", format="%.2f")})
        if st.button("💾 Save my edits"):
            e = edited.dropna(subset=["date"]).copy()
            e["date"] = pd.to_datetime(e["date"])
            e = e.groupby("date", as_index=False)["sales"].sum(min_count=1).sort_values("date")
            e = e.set_index("date")["sales"].asfreq(FREQ[SS["freq"]][0]).rename("sales").reset_index()
            SS["work"] = e
            reset_after("clean")
            st.success("Saved. Go to Step 2 · Clean."); st.rerun()
        w = SS["work"]
        st.plotly_chart(line_fig([("Sales", w.set_index("date")["sales"], dict(line=dict(color="#1f77b4")))], 260),
                        width="stretch")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(w):,}"); c2.metric("From", str(w.date.min().date())); c3.metric("To", str(w.date.max().date()))


# ================================================================= STEP 2
def step_clean():
    st.header("Step 2 · Clean the data")
    need("work", "Please load data in Step 1 first.")
    tip("Real data mein kuch dates ka sales missing hota hai aur kuch values galat/bahut bade (outliers) hoti hain. "
        "Model ko saaf data chahiye, isliye pehle inhe theek karte hain.")
    rule = FREQ[SS["freq"]][0]
    s = SS["work"].set_index("date")["sales"].asfreq(rule)
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = ((s < lo) | (s > hi)) & s.notna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(s):,}"); c2.metric("Missing values", int(s.isna().sum()))
    c3.metric("Outliers (IQR rule)", int(out.sum())); c4.metric("Allowed range", f"{lo:,.0f} – {hi:,.0f}")
    with st.expander("What is the IQR rule?"):
        st.write("IQR = Q3 − Q1. Any value below **Q1 − 1.5×IQR** or above **Q3 + 1.5×IQR** is called an outlier. "
                 "This is exactly what a box plot shows as dots outside the whiskers.")
    c1, c2 = st.columns(2)
    miss = c1.radio("Missing values →", ["Interpolate (recommended)", "Forward fill (copy previous value)", "Fill with median"])
    outl = c2.radio("Outliers →", ["Cap at IQR limits (recommended)", "Keep as they are"])
    if st.button("🧹 Apply cleaning", type="primary"):
        c = s.copy()
        c = (c.interpolate(limit_direction="both") if miss.startswith("Interp")
             else c.ffill().bfill() if miss.startswith("Forward") else c.fillna(c.median()))
        n_cap = 0
        if outl.startswith("Cap"):
            n_cap = int(((c < lo) | (c > hi)).sum()); c = c.clip(lo, hi)
        SS["clean"] = c; SS["clean_info"] = dict(missing=int(s.isna().sum()), capped=n_cap)
        reset_after("models")
    if "clean" in SS:
        info = SS["clean_info"]
        st.success(f"Done: {info['missing']} missing values filled, {info['capped']} outliers capped. You can go to Step 3.")
        fig = line_fig([("Before (raw)", s, dict(line=dict(color="lightgray"))),
                        ("After (clean)", SS["clean"], dict(line=dict(color="#1f77b4")))], 330)
        fig.add_scatter(x=s[out].index, y=s[out].values, mode="markers", name="Outliers", marker=dict(color="red", size=8))
        st.plotly_chart(fig, width="stretch")


# ================================================================= STEP 3
def step_explore():
    st.header("Step 3 · Explore the data (EDA)")
    need("clean", "Please finish Step 2 (click **Apply cleaning**) first.")
    y, m, freq = SS["clean"], SS["m"], SS["freq"]
    tip("EDA matlab data ko samajhna: average kitna hai, kab sales zyada hoti hai, koi pattern hai ya nahi. "
        "Model banane se pehle ye samajhna zaroori hai.")
    c = st.columns(6)
    for col, (k, v) in zip(c, {"Mean": y.mean(), "Median": y.median(), "Std dev": y.std(), "Variance": y.var(),
                                "Min": y.min(), "Max": y.max()}.items()):
        col.metric(k, f"{v:,.0f}")
    st.plotly_chart(line_fig([("Sales", y, dict(line=dict(color="#1f77b4")))], 300), width="stretch")

    a, b = st.columns(2)
    fig = go.Figure(go.Histogram(x=y.values, nbinsx=30, marker_color="#1f77b4"))
    fig.update_layout(title="Histogram (how sales are spread)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    a.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Box(y=y.values, name="Sales", marker_color="#1f77b4"))
    fig.update_layout(title="Box plot (dots = outliers)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    b.plotly_chart(fig, width="stretch")

    if freq == "Daily":
        prof = y.groupby(y.index.dayofweek).mean()
        prof.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in prof.index]
        ttl = "Average sales by day of week"
    else:
        names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        prof = y.groupby(y.index.month).mean()
        prof.index = [names[i - 1] for i in prof.index]
        ttl = "Average sales by month"
    fig = go.Figure(go.Bar(x=list(prof.index), y=prof.values, marker_color="#ff7f0e"))
    fig.update_layout(title=ttl + " (this is the seasonality)", height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Trend · Seasonality · Noise")
    tip("Time series 3 hisso se bani hoti hai: **Trend** (lambe time mein upar/neeche), **Seasonality** (har hafte/saal repeat hone wala pattern) "
        "aur **Noise** (random utaar-chadhav). Forecasting model inhi ko pakadte hain.")
    if len(y) >= 2 * m:
        dec = seasonal_decompose(y, model="additive", period=m)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Seasonality", "Noise (residual)"))
        for i, comp in enumerate([dec.trend, dec.seasonal, dec.resid], 1):
            fig.add_scatter(x=comp.index, y=comp.values, row=i, col=1, showlegend=False)
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Not enough rows for decomposition with this season length.")

    st.subheader("Is the data stationary? (ADF test)")
    p = adfuller(y.dropna())[1]; p1 = adfuller(y.diff().dropna())[1]
    c1, c2 = st.columns(2)
    c1.metric("p-value (original data)", f"{p:.4f}", "stationary" if p < 0.05 else "NOT stationary", delta_color="normal" if p < 0.05 else "inverse")
    c2.metric("p-value (after 1 differencing)", f"{p1:.4f}", "stationary" if p1 < 0.05 else "NOT stationary", delta_color="normal" if p1 < 0.05 else "inverse")
    tip("Stationary = data ka average/variance time ke saath nahi badalta. ARIMA ko stationary data chahiye. "
        "**p-value < 0.05 → stationary.** Agar original data stationary nahi hai to hum *differencing* (aaj − kal) karte hain; "
        "isi liye ARIMA/SARIMA mein **d = 1** use hota hai.")


# ================================================================= STEP 4
def step_models():
    st.header("Step 4 · Build and compare models")
    need("clean", "Please finish Steps 1–3 first.")
    y, m, freq = SS["clean"], SS["m"], SS["freq"]
    rule = FREQ[freq][0]
    tip("Hum 4 models banate hain aur dekhte hain kaun sabse achha future predict karta hai. **Last kuch rows hum chhupa lete hain (test set)**, "
        "model ko sirf purana data (train set) dete hain, phir uski prediction ko asli sales se compare karte hain. "
        "Jiska error sabse kam, wo winner.")
    c1, c2 = st.columns(2)
    h = c1.number_input(f"Test set size (last how many {freq.lower()} rows to hide?)", 5, max(6, len(y) // 3),
                        min(DEFAULT_TEST[freq], max(6, len(y) // 4)))
    c2.markdown("**Models:** Seasonal Naive (baseline) · Holt-Winters · ARIMA · SARIMA (grid search)")
    if len(y) - h < 2 * m + 5:
        st.warning("Too little training data for this season length. Reduce test size or season length."); return
    if st.button("🚀 Train and compare models", type="primary"):
        with st.spinner("Training… daily data can take around a minute. Please wait."):
            vals, start = pack(y)
            SS["models"] = run_models(vals, start, rule, m, int(h))
    if "models" not in SS:
        return
    R = SS["models"]; res = R["res"]
    for k, v in R["notes"].items(): st.caption(f"{k}: {v}")
    winner = res.index[0]
    st.success(f"🏆 Winner: **{winner}**  (RMSE {res.iloc[0]['RMSE']:,.1f}, MAPE {res.iloc[0]['MAPE']:.1f}%)")
    st.dataframe(res.round(2).style.highlight_min(axis=0, color="#c7f0c7"), width="stretch")
    with st.expander("What are RMSE, MAE, MAPE?"):
        st.markdown("- **MAE** – average mistake in sales units (e.g. off by 100 on average).\n"
                    "- **RMSE** – like MAE but big mistakes are punished more. Main score used here.\n"
                    "- **MAPE** – average mistake in % (3% = model is about 97% right).\n\n"
                    "**Lower is better** for all three.")
    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    traces = [("History (train)", R["train_tail"], dict(line=dict(color="lightgray"))),
              ("Actual (hidden test)", R["test"], dict(line=dict(color="black", width=3)))]
    for i, (n, p_) in enumerate(R["preds"].items()):
        traces.append((n, p_, dict(line=dict(color=colors[i % 5], width=2 if n != winner else 4, dash=None if n == winner else "dot"))))
    st.plotly_chart(line_fig(traces, 420), width="stretch")
    tip("Kaali moti line **asli sales** hai. Jis model ki line kaali line ke sabse paas chale, wo best hai. "
        "ARIMA aksar seedhi line ban jaata hai kyunki usme seasonality ka hissa nahi hota; isi liye SARIMA/Holt-Winters better hote hain.")
    if R["train_rmse"] is not None and R["sarima"]:
        st.markdown(f"**Overfitting check (SARIMA):** error on training data = {R['train_rmse']:,.1f}, "
                    f"error on unseen test data = {res.loc[R['sarima'], 'RMSE']:,.1f}. "
                    "If test error is far bigger than train error, the model has memorised instead of learned.")
    if R["sarima_cfg"]:
        st.caption(f"Grid search tried several (p, q, P, Q) combinations and kept the one with the lowest AIC: {R['sarima']}.")


# ================================================================= STEP 5
def step_forecast():
    st.header("Step 5 · Forecast the future")
    need("models", "Please train the models in Step 4 first.")
    y, m, freq = SS["clean"], SS["m"], SS["freq"]
    R = SS["models"]; rule = FREQ[freq][0]
    tip("Ab sabse achha model **poore data par dobara train** hota hai (kyunki ab test set chhupana nahi hai) aur aage ke dino/hafton/mahino ki sales predict karta hai. "
        "Shaded area **confidence interval** hai: asli sales iske andar aane ki 95% sambhavna hai.")
    c1, c2 = st.columns(2)
    names = list(R["res"].index)
    model_name = c1.selectbox("Model to use", names, index=0)
    n = c2.slider(f"How many {freq.lower()} periods to forecast?", 3, {"Daily": 90, "Weekly": 52, "Monthly": 36}[freq], {"Daily": 30, "Weekly": 12, "Monthly": 12}[freq])
    cfg = R["sarima_cfg"]
    vals, start = pack(y)
    with st.spinner("Fitting final model…"):
        fc = fit_forecast(vals, start, rule, m, model_name, cfg, int(n))
    hist = y.iloc[-min(len(y), 6 * m):]
    fig = line_fig([("History", hist, dict(line=dict(color="gray"))),
                    ("Forecast", fc["forecast"], dict(line=dict(color="crimson", width=3)))], 430)
    if fc["upper"].notna().all():
        fig.add_scatter(x=fc.index, y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fc.index, y=fc["lower"], fill="tonexty", fillcolor="rgba(220,20,60,0.15)", line=dict(width=0), name="95% interval")
    st.plotly_chart(fig, width="stretch")
    a, b, c = st.columns(3)
    a.metric("Average forecast", f"{fc['forecast'].mean():,.0f}")
    b.metric("Highest period", fc["forecast"].idxmax().strftime("%d %b %Y"), f"{fc['forecast'].max():,.0f}")
    c.metric("Lowest period", fc["forecast"].idxmin().strftime("%d %b %Y"), f"{fc['forecast'].min():,.0f}")
    st.markdown("**How a shop owner uses this:** stock up before the highest periods, avoid over-ordering before the lowest ones, "
                "and plan staff and offers around the expected demand.")
    out = fc.round(0); out.index.name = "date"
    st.dataframe(out, width="stretch", height=250)
    st.download_button("⬇️ Download forecast as CSV", out.to_csv().encode(), "forecast.csv", "text/csv")


# ================================================================= VIVA HELP
def step_viva():
    st.header("📘 Viva Help - how to explain this project")
    st.markdown("""
### 30-second explanation (say this first)
*"Shops need to know how much they will sell in the coming days or months so they can order the right stock.
My project predicts future retail sales from past sales using time-series models. I load data, clean it (missing values and outliers),
explore it (trend and seasonality), build four models (a simple baseline, Holt-Winters, ARIMA and SARIMA), compare them on data the models
did not see, and use the best one to forecast the future. It is deployed as a web app with Streamlit."*

### The 5 steps (this matches the app)
| Step | What I do | Why | Syllabus |
|---|---|---|---|
| 1 Data | Load sales data, choose frequency | Need history to learn from | Data collection |
| 2 Clean | Fill missing values, cap outliers (IQR) | Bad data → bad model | Preprocessing |
| 3 Explore | Mean, median, variance, std, plots, decomposition, ADF test | Understand patterns before modelling | EDA, time-series basics |
| 4 Models | Baseline, Holt-Winters, ARIMA, SARIMA + grid search, train/test split, RMSE/MAE/MAPE | Find the most accurate model | Time series, validation, tuning |
| 5 Forecast | Refit best model on all data, predict future with confidence interval | The actual business use | Forecasting applications |
""")
    st.subheader("Terms in simple words (click to open)")
    terms = {
        "Time series": "Data collected over time in order (daily/monthly sales). Order matters, so we never shuffle it.",
        "Trend / Seasonality / Noise": "Trend = long-term rise or fall. Seasonality = pattern that repeats (weekends sell more, December sells more). Noise = random ups and downs.",
        "Stationary & ADF test": "Stationary means the average and spread do not change over time. ADF test checks this: p-value < 0.05 means stationary. If not, we difference the data (today − yesterday), which is the 'd' in ARIMA.",
        "ARIMA(p,d,q)": "AR(p): uses the last p values. I(d): how many times we difference. MA(q): uses the last q errors. It cannot handle seasonality.",
        "SARIMA(p,d,q)(P,D,Q,m)": "ARIMA plus a seasonal part. m is the season length (7 for daily, 12 for monthly). Best when data has a repeating pattern.",
        "Holt-Winters (exponential smoothing)": "A weighted average where recent data gets more weight. It tracks level, trend and seasonality.",
        "Baseline (Seasonal Naive)": "Simplest model: 'next week will be like last week'. A real model must beat it, otherwise it is not useful.",
        "Train / test split": "Train set = data the model learns from. Test set = last part we hide and use for marking the model. For time series the split is by time, never random (random would leak the future).",
        "RMSE / MAE / MAPE": "Different ways to measure prediction error. MAE = average error. RMSE = punishes big errors more. MAPE = error in percent. Lower is better.",
        "Grid search & AIC": "Grid search tries many (p,q,P,Q) combinations. AIC is a score that rewards good fit but punishes unnecessary complexity; the lowest AIC wins.",
        "Overfitting": "Model memorises training data and fails on new data. Check: test error much bigger than train error.",
        "IQR / outliers": "IQR = Q3 − Q1. Values outside Q1 − 1.5·IQR and Q3 + 1.5·IQR are outliers. We cap them instead of deleting rows so the time order stays intact.",
        "Confidence interval": "The shaded range where actual sales are expected to fall 95% of the time. It gets wider the further we forecast.",
    }
    for k, v in terms.items():
        with st.expander(k):
            st.write(v)
    st.subheader("Questions the teacher may ask")
    qa = {
        "Why did you choose SARIMA / the winning model?": "Retail sales have a repeating pattern (weekly or yearly). SARIMA models trend and seasonality together, and it had the lowest error on the hidden test data in Step 4.",
        "Why did ARIMA perform badly?": "ARIMA has no seasonal component, so its forecast flattens into an almost straight line while real sales go up and down regularly.",
        "How did you choose the parameters?": "Grid search: I tried many combinations and picked the one with the lowest AIC on the training data, then checked it on the test data.",
        "Why not shuffle the data for train/test?": "In time series the future must not be visible during training. Shuffling would leak future information and give fake good results.",
        "How did you handle missing values and outliers?": "Missing values by interpolation; outliers detected with the IQR rule and capped at the limits instead of deleting them.",
        "How do you know it is not overfitting?": "I compare training error with test error and the model also beats the baseline on unseen data.",
        "What are the limitations?": "It uses only past sales, not promotions, holidays, price or weather. Long-range forecasts get uncertain. Improvement: SARIMAX with extra variables, or machine-learning models with lag features.",
        "What are the ethical issues?": "Over-forecasting causes waste and under-forecasting causes empty shelves. Customer data should be anonymised, and forecasts should support human decisions rather than replace them.",
        "How is it deployed?": "The app is written in Streamlit, the code is on GitHub, and Streamlit Community Cloud runs it and gives a public link.",
        "Where did the data come from?": "Option 1 is real US Census retail sales data. Option 2 is a large store dataset with the same columns as Kaggle's Store Sales data; its values are simulated. You can also upload any Kaggle CSV in Step 1.",
    }
    for k, v in qa.items():
        with st.expander(k):
            st.write(v)
    st.warning("Be honest in the viva: the large store dataset is simulated (same structure as Kaggle's Store Sales). "
               "The monthly US retail dataset is real. The app works on any real CSV you upload.")


{"1️⃣ Data": step_data, "2️⃣ Clean": step_clean, "3️⃣ Explore": step_explore,
 "4️⃣ Models": step_models, "5️⃣ Forecast": step_forecast, "📘 Viva Help": step_viva}[page]()
