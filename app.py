import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.express as px

st.set_page_config(page_title="Zero-Waste Grocery AI", page_icon="🥬", layout="wide")

# ---------- Synthetic data and models ----------
@st.cache_data
def make_data(seed=17, days=120):
    rng = np.random.default_rng(seed)
    products = [
        ("MILK-01", "Milk", "Dairy", 5, 1.35, 0.72),
        ("BREAD-01", "Bread", "Bakery", 3, 2.00, 1.05),
        ("YOG-01", "Yogurt", "Dairy", 14, 0.95, 0.48),
        ("SPIN-01", "Spinach", "Produce", 5, 1.80, 0.80),
        ("TOM-01", "Tomatoes", "Produce", 7, 2.40, 1.10),
        ("CHICK-01", "Chicken", "Meat", 4, 5.50, 3.50),
    ]
    stores = ["Store A", "Store B", "Store C"]
    rows = []
    start = date.today() - timedelta(days=days)
    for store_i, store in enumerate(stores):
        for sku, name, cat, shelf, price, cost in products:
            base_demand = rng.integers(12, 34)
            for d in range(days):
                dt = start + timedelta(days=d)
                weekend = dt.weekday() >= 5
                promo = int(rng.random() < 0.12)
                event = int(rng.random() < 0.06)
                temp = float(np.clip(rng.normal(25, 5), 10, 40))
                weather_factor = 1 + (0.12 if temp > 32 and cat in ["Dairy", "Produce"] else 0)
                demand = max(0, int(rng.poisson(base_demand * (1.18 if weekend else 1) *
                                               (1.25 if promo else 1) *
                                               (1.15 if event else 1) * weather_factor)))
                received = max(0, int(rng.normal(base_demand * 1.05, 5))) if d % 2 == 0 else 0
                stock = max(0, int(base_demand * (1.1 + rng.normal(0, .25))))
                sold = min(demand, stock + received)
                waste = max(0, int(max(0, stock + received - sold) * (0.06 if cat == "Produce" else 0.025)))
                rows.append({
                    "date": dt, "store_id": store, "sku_id": sku, "product": name,
                    "category": cat, "shelf_life_days": shelf, "units_sold": sold,
                    "stock_on_hand": stock, "units_received": received,
                    "price": price * (0.9 if promo else 1), "base_price": price,
                    "unit_cost": cost, "promotion": promo, "waste_units": waste,
                    "temperature_c": round(temp, 1), "local_event": event,
                    "days_to_expiry": int(rng.integers(0, shelf + 1))
                })
    return pd.DataFrame(rows)

def train_forecast(data):
    """Forecast demand using each store/SKU's trailing 7-day average.

    This intentionally avoids compiled ML dependencies so the demo can run
    on locked-down Windows computers. The final 14 days are held out for MAE.
    """
    data = data.copy().sort_values(["store_id", "sku_id", "date"])
    groups = data.groupby(["store_id", "sku_id"])["units_sold"]
    # Shift first so each day's forecast uses only earlier sales.
    data["forecast"] = groups.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
    )
    overall_mean = float(data["units_sold"].mean())
    data["forecast"] = data["forecast"].fillna(overall_mean).clip(lower=0)

    cutoff = data["date"].max() - timedelta(days=14)
    test = data[data["date"] > cutoff].copy()
    train = data[data["date"] <= cutoff]
    # Compare against a simple per-store/per-SKU training-period average.
    means = train.groupby(["store_id", "sku_id"])["units_sold"].mean().rename("baseline")
    test = test.join(means, on=["store_id", "sku_id"])
    if len(test):
        model_mae = float(np.mean(np.abs(test["units_sold"] - test["forecast"])))
    else:
        model_mae = 0.0
    return data, model_mae

def recommendation_engine(row):
    demand = max(0, float(row["forecast"]))
    stock = int(row["stock_on_hand"])
    incoming = int(row["units_received"])
    expiry = int(row["days_to_expiry"])
    shelf = int(row["shelf_life_days"])
    price = float(row["base_price"])
    # Business rules are deliberately visible and configurable for a POC.
    if expiry <= 1 and stock >= 4:
        discount = 30 if expiry == 0 else 20
        return {
            "action": "Markdown",
            "timing": "Now / next trading period",
            "quantity": f"{stock} units eligible for review",
            "markdown": f"{discount}%",
            "confidence": min(0.96, 0.68 + 0.04 * stock / 10),
            "expected_effect": "May increase sell-through and reduce expiry waste; margin may decrease.",
            "evidence": f"{stock} units on hand; {expiry} day(s) to expiry; forecast demand {demand:.1f} units/day.",
            "priority": "High"
        }
    available = stock + incoming
    target = demand * 2.0
    if available < max(5, demand * 0.75):
        qty = max(0, int(np.ceil(target - available)))
        return {
            "action": "Replenish",
            "timing": "Next feasible delivery",
            "quantity": f"Order {qty} units",
            "markdown": "None",
            "confidence": min(0.93, 0.65 + demand / 100),
            "expected_effect": "Can reduce stockout risk; confirm supplier capacity and lead time.",
            "evidence": f"{available} units available including receipts; forecast demand {demand:.1f} units/day.",
            "priority": "High"
        }
    if stock > max(10, demand * 2.5) and expiry <= max(3, shelf * 0.4):
        return {
            "action": "Inter-store transfer",
            "timing": "Today, if transfer arrives before expiry",
            "quantity": f"Review transfer of {max(1, int(stock - demand*2))} units",
            "markdown": "None",
            "confidence": 0.72,
            "expected_effect": "May rebalance excess stock to a store with higher demand; transport time/cost must be checked.",
            "evidence": f"{stock} units on hand; {expiry} day(s) to expiry; forecast demand {demand:.1f} units/day.",
            "priority": "Medium"
        }
    return {
        "action": "Hold / monitor",
        "timing": "Recheck tomorrow",
        "quantity": "No order recommended",
        "markdown": "None",
        "confidence": 0.70,
        "expected_effect": "Avoids unnecessary replenishment; continue monitoring demand and freshness.",
        "evidence": f"{stock} units on hand, {incoming} incoming; forecast demand {demand:.1f} units/day; {expiry} day(s) to expiry.",
        "priority": "Low"
    }

data = make_data()
model_data, model_mae = train_forecast(data)
latest_date = model_data["date"].max()
latest = model_data[model_data["date"] == latest_date].copy()
latest["spoilage_risk"] = np.where(
    (latest["days_to_expiry"] <= 1) & (latest["stock_on_hand"] >= 4), "High",
    np.where(latest["days_to_expiry"] <= 3, "Medium", "Low")
)
latest["stockout_risk"] = np.where(latest["stock_on_hand"] + latest["units_received"] < latest["forecast"], "High", "Low")
latest["recommendation"] = latest.apply(recommendation_engine, axis=1)
recs = pd.json_normalize(latest["recommendation"])
latest = pd.concat([latest.reset_index(drop=True), recs.reset_index(drop=True)], axis=1)
latest["waste_rate"] = latest["waste_units"] / (latest["units_sold"] + latest["waste_units"] + 1)

if "decisions" not in st.session_state:
    st.session_state.decisions = {}
if "overrides" not in st.session_state:
    st.session_state.overrides = {}

# ---------- UI ----------
st.title("🥬 Zero-Waste Grocery AI")
st.caption("Agentic decision-support proof of concept • Synthetic data • Human approval required")
st.info("This demo uses simulated grocery data and illustrative business rules. It is not connected to live POS, ERP, or warehouse systems.")

with st.sidebar:
    st.header("Control panel")
    selected_store = st.selectbox("Store", ["All stores"] + sorted(latest.store_id.unique()))
    selected_category = st.selectbox("Category", ["All categories"] + sorted(latest.category.unique()))
    st.markdown("---")
    st.subheader("Business rules")
    st.caption("The demo uses a 2-day stock target, expiry-triggered markdowns, and stockout alerts. Review suggestions before acting.")

view = latest.copy()
if selected_store != "All stores":
    view = view[view.store_id == selected_store]
if selected_category != "All categories":
    view = view[view.category == selected_category]

total_stock = int(view.stock_on_hand.sum())
high_risk = int((view.spoilage_risk == "High").sum())
stockout_alerts = int((view.stockout_risk == "High").sum())
waste_units = int(view.waste_units.sum())
c1,c2,c3,c4 = st.columns(4)
c1.metric("Units on hand", f"{total_stock:,}")
c2.metric("High spoilage-risk SKUs", high_risk)
c3.metric("Potential stockout alerts", stockout_alerts)
c4.metric("Recent simulated waste", f"{waste_units:,} units")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Forecast & Freshness", "AI Recommendations", "Planner Decisions", "Evaluation"])

with tab1:
    st.subheader("Store inventory snapshot")
    st.dataframe(view[["store_id","product","category","stock_on_hand","units_received","forecast","days_to_expiry","spoilage_risk","stockout_risk"]].rename(columns={
        "store_id":"Store","product":"Product","category":"Category","stock_on_hand":"On hand",
        "units_received":"Incoming","forecast":"Forecast units/day","days_to_expiry":"Days to expiry",
        "spoilage_risk":"Spoilage risk","stockout_risk":"Stockout risk"
    }), use_container_width=True, hide_index=True)
    chart = view.groupby("category", as_index=False).agg(stock=("stock_on_hand","sum"), forecast=("forecast","sum"))
    st.plotly_chart(px.bar(chart, x="category", y=["stock","forecast"], barmode="group", title="Current stock vs. forecast demand (illustrative)"), use_container_width=True)

with tab2:
    st.subheader("Demand forecast and freshness risk")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.bar(view, x="product", y="forecast", color="store_id", title="Predicted daily demand by product"), use_container_width=True)
    with right:
        risk_counts = view.groupby(["product","spoilage_risk"], as_index=False).size()
        st.plotly_chart(px.bar(risk_counts, x="product", y="size", color="spoilage_risk", title="Freshness risk flags"), use_container_width=True)
    st.caption(f"Trailing 7-day average holdout MAE: {model_mae:.2f} units/day. Evaluation uses a chronological 14-day holdout; the dataset is synthetic.")

with tab3:
    st.subheader("Coordinated recommendations")
    st.caption("Agents propose actions; the coordinator applies expiry and stock-availability rules. All actions remain proposals until approved.")
    for _, row in view.iterrows():
        key = f"{row.store_id}|{row.sku_id}"
        rec = row.to_dict()
        with st.container(border=True):
            a,b = st.columns([3,1])
            a.markdown(f"**{row['product']} — {row['store_id']}**")
            a.write(f"Action: **{row['action']}** · Priority: **{row['priority']}**")
            a.write(f"Timing: {row['timing']} | Quantity: {row['quantity']} | Markdown: {row['markdown']}")
            a.write(f"Confidence: {row['confidence']:.0%}")
            a.caption(f"Expected effect: {row['expected_effect']}")
            a.caption(f"Evidence: {row['evidence']}")
            with b:
                decision = st.session_state.decisions.get(key, "Pending")
                st.write(f"Status: **{decision}**")
                custom_qty = st.text_input("Modify quantity/action", value=st.session_state.overrides.get(key, row["quantity"]), key=f"edit_{key}")
                if st.button("Save modification", key=f"save_{key}"):
                    st.session_state.overrides[key] = custom_qty
                    st.session_state.decisions[key] = "Modified"
                    st.rerun()
                x,y = st.columns(2)
                if x.button("Approve", key=f"approve_{key}"):
                    st.session_state.decisions[key] = "Approved"
                    st.rerun()
                if y.button("Reject", key=f"reject_{key}"):
                    st.session_state.decisions[key] = "Rejected"
                    st.rerun()

with tab4:
    st.subheader("Planner decision log")
    log_rows = []
    for _, row in latest.iterrows():
        key = f"{row.store_id}|{row.sku_id}"
        log_rows.append({
            "store_id": row.store_id, "sku_id": row.sku_id, "product": row.product,
            "proposed_action": row.action, "proposed_quantity": row.quantity,
            "planner_override": st.session_state.overrides.get(key, ""),
            "decision": st.session_state.decisions.get(key, "Pending"),
            "recorded_at": "Current session (simulated)"
        })
    log = pd.DataFrame(log_rows)
    st.dataframe(log, use_container_width=True, hide_index=True)
    st.download_button("Download decision log CSV", log.to_csv(index=False).encode("utf-8"), "planner_decision_log.csv", "text/csv")

with tab5:
    st.subheader("Evaluation against simple baselines")
    st.markdown("**Forecast method:** trailing 7-day average using prior sales only. Compare its holdout MAE with the historical-mean baseline.")
    eval_data = model_data.copy()
    eval_data["dow"] = pd.to_datetime(eval_data["date"]).dt.dayofweek
    cutoff = eval_data["date"].max() - timedelta(days=14)
    test = eval_data[eval_data["date"] > cutoff].copy()
    train = eval_data[eval_data["date"] <= cutoff]
    means = train.groupby(["store_id","sku_id"])["units_sold"].mean().rename("baseline")
    test = test.join(means, on=["store_id","sku_id"])
    baseline_mae = float(np.mean(np.abs(test["units_sold"] - test["baseline"])))
    m1,m2,m3 = st.columns(3)
    m1.metric("Trailing 7-day MAE", f"{model_mae:.2f}")
    m2.metric("Historical mean MAE", f"{baseline_mae:.2f}")
    m3.metric("MAE improvement", f"{(baseline_mae-model_mae):.2f}", help="Positive means the trailing 7-day forecast has lower MAE.")
    st.markdown("**Operational metrics (simulated snapshot)**")
    waste_rate = float(view.waste_units.sum() / max(1, (view.units_sold + view.waste_units).sum()))
    stockout_share = float((view.stock_on_hand + view.units_received < view.forecast).mean())
    e1,e2,e3 = st.columns(3)
    e1.metric("Observed simulated waste rate", f"{waste_rate:.1%}")
    e2.metric("SKUs flagged for stockout", f"{stockout_share:.1%}")
    e3.metric("Planner approvals", sum(v == "Approved" for v in st.session_state.decisions.values()))
    st.warning("Synthetic data is for demonstrating workflow only. For a formal evaluation, use a separate fixed test dataset and compare waste, stockouts, margin, and recommendation quality under identical constraints.")

st.markdown("---")
st.caption("Zero-Waste Grocery AI • Prototype only • Recommendations require human review")
