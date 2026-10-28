
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Federated Traffic Intelligence",
    page_icon="🌐",
    layout="wide"
)

DATA_PATH = Path(__file__).parent / "geant-flat-tms (1).csv"

# Recorded results from the research paper
RESULTS = pd.DataFrame({
    "Model": [
        "Persistence",
        "Local AR(12)",
        "Centralized AR(12)",
        "FedAvg AR(12)",
        "Local + Seasonal"
    ],
    "MAE (log1p)": [0.5249, 0.5066, 0.5285, 0.5293, 0.5006],
    "RMSE (log1p)": [0.9454, 0.8282, 0.8429, 0.8432, 0.8215],
    "WAPE (%)": [10.86, 12.35, 19.60, 19.73, 12.04]
})


@st.cache_data
def load_data(path):
    # The downloaded CSV has no header row.
    df = pd.read_csv(path, header=None)
    df.columns = ["Timestamp"] + [
        f"Flow_{i:03d}" for i in range(1, df.shape[1])
    ]
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%Y-%m-%d-%H-%M",
        errors="coerce"
    )
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


st.sidebar.title("🌐 Traffic Intelligence")
st.sidebar.caption("Federated forecasting research prototype")
page = st.sidebar.radio(
    "Navigation",
    ["Dataset Explorer", "Federated Simulation", "Results"]
)

st.title("🌐 Federated Traffic Prediction")
st.caption(
    "Privacy-aware network forecasting | GÉANT traffic-matrix dataset"
)

try:
    df = load_data(str(DATA_PATH))
    flow_cols = list(df.columns[1:])
    traffic = df[flow_cols].fillna(0).clip(lower=0)
    aggregate = traffic.sum(axis=1)

except Exception as e:
    st.error(f"Unable to load the CSV: {e}")
    st.info("Check that the CSV filename matches DATA_PATH.")
    st.stop()


if page == "Dataset Explorer":
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Traffic flows", f"{len(flow_cols):,}")
    c3.metric("Numeric measurements", f"{traffic.size:,}")

    st.success("Dataset loaded successfully.")

    st.subheader("Historical Aggregate Traffic")
    chart_df = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Aggregate Traffic": aggregate
    }).dropna(subset=["Timestamp"])

    fig = px.line(
        chart_df.iloc[::max(1, len(chart_df) // 2000)],
        x="Timestamp",
        y="Aggregate Traffic",
        title="Traffic across all network flows"
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    selected_flow = st.selectbox("Inspect an individual flow", flow_cols)
    flow_df = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Traffic": traffic[selected_flow]
    }).dropna(subset=["Timestamp"])

    fig2 = px.line(
        flow_df.iloc[::max(1, len(flow_df) // 1500)],
        x="Timestamp", y="Traffic",
        title=f"Historical traffic — {selected_flow}"
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Preview dataset"):
        st.dataframe(df.head(10), use_container_width=True)



elif page == "Federated Simulation":
    import time

    st.subheader("Live Federated Learning Experiment")
    st.write(
        "Four simulated clients train a shared autoregressive "
        "traffic forecasting model using FedAvg."
    )

    st.info(
        "Live demonstration: this is a lightweight experiment on "
        "selected traffic flows, not the exact experiment reported "
        "in the research paper."
    )

    client_groups = np.array_split(np.array(flow_cols), 4)

    # Select a small, reproducible subset for a fast live demo.
    selected_groups = [list(group[:8]) for group in client_groups]

    cards = st.columns(4)
    for i, (card, group) in enumerate(
        zip(cards, selected_groups)
    ):
        with card:
            st.metric(f"Client {i + 1}", f"{len(group)} flows")
            st.caption("Local training partition")

    st.divider()

    st.markdown("### Training configuration")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients", "4")
    c2.metric("Federated rounds", "50")
    c3.metric("Local updates per round", "3")

    if st.button("🚀 Start Actual FedAvg Training", type="primary"):

        rng = np.random.default_rng(42)
        window = 12
        learning_rate = 0.03
        rounds = 50
        local_steps = 3
        batch_size = 512

        split = int(len(traffic) * 0.8)
        client_data = []

        # Build chronological lagged examples for each client.
        for group in selected_groups:
            X_parts, y_parts = [], []
            X_test_parts, y_test_parts = [], []

            for flow in group:
                series = (
                    np.log1p(
                        traffic[flow].to_numpy(dtype=float)
                    ) / 15.0
                )

                windows = np.lib.stride_tricks.sliding_window_view(
                    series, window + 1
                )

                X = windows[:, :window].copy()
                y = windows[:, window].copy()

                target_positions = np.arange(
                    window, len(series)
                )
                train_mask = target_positions < split
                test_mask = target_positions >= split

                if train_mask.any():
                    X_parts.append(X[train_mask])
                    y_parts.append(y[train_mask])

                if test_mask.any():
                    X_test_parts.append(X[test_mask])
                    y_test_parts.append(y[test_mask])

            client_data.append({
                "X": np.concatenate(X_parts),
                "y": np.concatenate(y_parts),
                "X_test": np.concatenate(X_test_parts),
                "y_test": np.concatenate(y_test_parts)
            })

        # One shared linear autoregressive model.
        global_w = np.zeros(window)
        global_b = 0.0
        history = []
        progress = st.progress(0)
        status = st.empty()
        chart_area = st.empty()

        for round_num in range(rounds):
            client_weights = []
            client_biases = []
            client_sizes = []

            # Each client starts from the current global model.
            for client in client_data:
                X_local = client["X"]
                y_local = client["y"]

                w = global_w.copy()
                b = global_b

                # Three genuine local gradient updates.
                for _ in range(local_steps):
                    n = len(y_local)
                    indices = rng.choice(
                        n,
                        size=min(batch_size, n),
                        replace=False
                    )

                    X_batch = X_local[indices]
                    y_batch = y_local[indices]

                    error = X_batch @ w + b - y_batch

                    grad_w = (
                        X_batch.T @ error / len(indices)
                        + 0.001 * w
                    )
                    grad_b = error.mean()

                    w -= learning_rate * grad_w
                    b -= learning_rate * grad_b

                client_weights.append(w)
                client_biases.append(b)
                client_sizes.append(len(y_local))

            # Weighted FedAvg aggregation.
            weights = np.asarray(client_sizes, dtype=float)
            weights /= weights.sum()

            global_w = sum(
                weight * model_w
                for weight, model_w in zip(
                    weights, client_weights
                )
            )
            global_b = sum(
                weight * model_b
                for weight, model_b in zip(
                    weights, client_biases
                )
            )

            # Evaluate on held-out future observations.
            errors = []
            for client in client_data:
                predictions = (
                    client["X_test"] @ global_w + global_b
                )
                errors.extend(
                    np.abs(predictions - client["y_test"])
                )

            test_mae = float(np.mean(errors))
            history.append({
                "Round": round_num + 1,
                "Test MAE (scaled log1p)": test_mae
            })

            progress.progress((round_num + 1) / rounds)
            status.info(
                f"Completed round {round_num + 1}/{rounds} "
                f"| Held-out MAE: {test_mae:.5f}"
            )

        st.session_state["fed_history"] = history
        st.session_state["fed_final_weights"] = global_w
        st.session_state["fed_final_bias"] = global_b

        st.success("Actual local training and FedAvg completed!")

    if "fed_history" in st.session_state:
        history_df = pd.DataFrame(
            st.session_state["fed_history"]
        )

        st.subheader("Global Model Convergence")
        fig = px.line(
            history_df,
            x="Round",
            y="Test MAE (scaled log1p)",
            markers=True,
            title="Held-out forecasting error across FedAvg rounds"
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        final_mae = history_df.iloc[-1][
            "Test MAE (scaled log1p)"
        ]
        st.metric("Final held-out MAE", f"{final_mae:.5f}")

        st.caption(
            "This metric belongs to the live demo run and is not "
            "directly comparable to the paper's reported metrics. "
            "The model is a lightweight linear autoregressor."
        )

        
        st.divider()
        st.subheader("🔮 Live Traffic Forecast")

        available_flows = [
            flow for group in selected_groups for flow in group
        ]

        chosen_flow = st.selectbox(
            "Select a traffic flow",
            available_flows,
            key="forecast_flow"
        )

        if st.button("Generate 12-Step Forecast", type="primary"):
            raw_series = traffic[chosen_flow].to_numpy(dtype=float)
            scaled_series = np.log1p(
                np.clip(raw_series, 0, None)
            ) / 15.0

            # Use the last 12 observations as input.
            context = list(scaled_series[-12:])
            forecast_scaled = []

            max_scaled = np.log1p(
                max(float(traffic[chosen_flow].max()), 0)
            ) / 15.0

            for _ in range(12):
                next_value = (
                    np.dot(
                        st.session_state["fed_final_weights"],
                        np.array(context[-12:])
                    )
                    + st.session_state["fed_final_bias"]
                )

                next_value = float(
                    np.clip(next_value, 0, max_scaled)
                )

                forecast_scaled.append(next_value)
                context.append(next_value)

            forecast_values = np.expm1(
                np.array(forecast_scaled) * 15.0
            )

            timestamps = df["Timestamp"].dropna()
            last_time = timestamps.iloc[-1]

            historical_times = timestamps.iloc[-24:].reset_index(
                drop=True
            )
            historical_values = raw_series[-24:]

            future_times = pd.date_range(
                start=last_time + pd.Timedelta(minutes=15),
                periods=12,
                freq="15min"
            )

            chart_data = pd.concat([
                pd.DataFrame({
                    "Time": historical_times,
                    "Traffic": historical_values,
                    "Series": "Observed traffic"
                }),
                pd.DataFrame({
                    "Time": future_times,
                    "Traffic": forecast_values,
                    "Series": "Model forecast"
                })
            ], ignore_index=True)

            fig_forecast = px.line(
                chart_data,
                x="Time",
                y="Traffic",
                color="Series",
                markers=True,
                title=f"Traffic Forecast — {chosen_flow}"
            )

            fig_forecast.update_layout(
                template="plotly_white",
                xaxis_title="Time",
                yaxis_title="Traffic volume"
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )

            st.metric(
                "Predicted traffic at step 12",
                f"{forecast_values[-1]:,.2f}"
            )

            st.caption(
                "The forecast uses the lightweight global model "
                "trained in this demo. It is an illustrative "
                "recursive forecast, not a validated paper result."
            )



else:
    st.subheader("Experimental Results")
    st.caption(
        "Recorded metrics from the research paper — not newly "
        "computed by this dashboard."
    )

    st.dataframe(
        RESULTS.style.format({
            "MAE (log1p)": "{:.4f}",
            "RMSE (log1p)": "{:.4f}",
            "WAPE (%)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Choose a metric",
        ["MAE (log1p)", "RMSE (log1p)", "WAPE (%)"]
    )

    fig = px.bar(
        RESULTS.sort_values(metric),
        x="Model",
        y=metric,
        title=f"Model comparison — {metric}",
        text=metric
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Main findings")
    st.markdown("""
    - **Best log-space MAE:** Local + Seasonal (0.5006).
    - **Best log-space RMSE:** Local + Seasonal (0.8215).
    - **Best raw-scale WAPE:** Persistence (10.86%).
    - FedAvg is close to centralized training but does not beat
      the strongest reported baselines.
    """)

    st.caption(
        "Source: results reported in the submitted research paper."
    )

import time

st.divider()
st.subheader("▶ Interactive Federated Workflow")

st.write(
    "Demonstrate how four clients participate in federated learning "
    "without pooling their original traffic examples."
)

if st.button("Run Federated Learning Demonstration", type="primary"):
    progress = st.progress(0)
    status = st.empty()

    stages = [
        "1/5 — Initializing the global model",
        "2/5 — Clients prepare their local traffic partitions",
        "3/5 — Clients perform local-training stage (workflow preview)",
        "4/5 — Coordinator aggregates model updates (conceptual preview)",
        "5/5 — Global model is ready for distribution"
    ]

    for i, stage in enumerate(stages):
        status.info(stage)
        progress.progress((i + 1) / len(stages))
        time.sleep(0.6)

    st.success("Workflow walkthrough completed!")

    st.info(
        "This is an interactive workflow visualization. "
        "It does not execute the paper's FedAvg training algorithm."
    )

    st.markdown("### What happened?")
    st.markdown("""
    - Original traffic examples remain in their assigned client partitions.
    - Clients are shown participating in local training.
    - The coordinator stage illustrates the aggregation concept.
    - The resulting global model can be distributed to participating clients.
    """)
