
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Rainfall Sensor Anomaly Detection",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    files = {
        "readings": "test_sensor_readings.csv",
        "models": "test_model_comparison.csv",
        "validation": "validation_model_comparison.csv",
        "stations": "station_performance.csv",
        "faults": "fault_type_detection.csv",
        "episodes": "episode_detection.csv",
        "details": "episode_details.csv"
    }
    data = {
        key: pd.read_csv(DATA_DIR / filename)
        for key, filename in files.items()
    }
    data["readings"]["station_id"] = (
        data["readings"]["station_id"].astype(str)
    )
    data["readings"]["timestamp"] = pd.to_datetime(
        data["readings"]["timestamp"]
    )
    data["stations"]["station_id"] = (
        data["stations"]["station_id"].astype(str)
    )
    return data

try:
    d = load_data()
except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()

readings = d["readings"]
models = d["models"].sort_values("f1", ascending=False)
model_names = models["model"].tolist()

st.sidebar.title("🌧️ Rainfall Analytics")
st.sidebar.caption("Capstone | Six-model evaluation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Sensor Explorer",
        "Model Comparison",
        "Fault Episodes",
        "Historical Investigation"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Retrospective evaluation using synthetically "
    "injected faults. Historical anomalies are not "
    "independently verified."
)

st.title("Rainfall Sensor Anomaly Detection")
st.caption(
    "Interactive exploration of machine-learning "
    "results from four rainfall monitoring stations"
)

if page == "Overview":

    st.header("Project Overview")

    best = models.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test observations", f"{len(readings):,}")
    c2.metric("Rainfall stations", readings["station_id"].nunique())
    c3.metric("ML models", len(models))
    c4.metric("Best test F1", f"{best['f1']:.2%}")

    st.subheader("Six-Model Test Performance")

    fig = px.bar(
        models,
        x="f1",
        y="model",
        orientation="h",
        color="f1",
        color_continuous_scale="Blues",
        text=models["f1"].map(lambda x: f"{x:.2%}"),
        labels={"f1": "F1-score", "model": "Model"}
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=440
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 0.85])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Research Findings")
    st.write(
        "Random Forest achieved the highest retrospective "
        "test F1-score. Across all six evaluated models, "
        "large synthetic spikes were much easier to detect "
        "than persistent flatline and stuck-at-zero faults."
    )

    st.warning(
        "The test evaluation is retrospective, and these "
        "results should not be interpreted as accuracy "
        "against independently verified real sensor faults."
    )

elif page == "Sensor Explorer":

    st.header("Interactive Rainfall Sensor Explorer")

    c1, c2 = st.columns(2)

    selected_station = c1.selectbox(
        "Select rainfall station",
        sorted(readings["station_id"].unique())
    )

    selected_model = c2.selectbox(
        "Select anomaly-detection model",
        model_names
    )

    station_data = (
        readings[
            readings["station_id"] == selected_station
        ]
        .sort_values("timestamp")
        .copy()
    )

    min_date = station_data["timestamp"].min().date()
    max_date = station_data["timestamp"].max().date()

    dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(dates) == 2:
        start_date, end_date = dates
        station_data = station_data[
            station_data["timestamp"].dt.date.between(
                start_date, end_date
            )
        ]

    prediction_col = f"{selected_model}_prediction"
    score_col = f"{selected_model}_score"

    anomaly_rows = station_data[
        station_data[prediction_col] == 1
    ]

    c1, c2, c3 = st.columns(3)
    c1.metric("Readings shown", f"{len(station_data):,}")
    c2.metric("Model alerts", f"{len(anomaly_rows):,}")
    c3.metric(
        "Modified synthetic readings",
        f"{int(station_data['synthetic_label'].sum()):,}"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=station_data["timestamp"],
        y=station_data["original_value"],
        name="Original rainfall",
        mode="lines",
        line=dict(color="#64748b", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=station_data["timestamp"],
        y=station_data["simulated_value"],
        name="Simulated rainfall",
        mode="lines",
        line=dict(color="#2563eb", width=1.6)
    ))

    modified_rows = station_data[
        station_data["synthetic_label"] == 1
    ]

    fig.add_trace(go.Scatter(
        x=modified_rows["timestamp"],
        y=modified_rows["simulated_value"],
        mode="markers",
        name="Synthetic modifications",
        marker=dict(
            color="#f59e0b",
            size=9,
            symbol="x"
        )
    ))

    fig.add_trace(go.Scatter(
        x=anomaly_rows["timestamp"],
        y=anomaly_rows["simulated_value"],
        mode="markers",
        name="Model alerts",
        marker=dict(
            color="#dc2626", size=8,
            symbol="circle-open", line=dict(width=2)
        )
    ))

    fig.update_layout(
        title=f"Rainfall Timeline — Station {selected_station}",
        xaxis_title="Timestamp",
        yaxis_title="Rainfall reading (mm)",
        height=520,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Red markers indicate model predictions, not "
        "independently confirmed sensor faults."
    )

    st.subheader("Explore Individual Readings")

    display_cols = [
        "timestamp",
        "original_value",
        "simulated_value",
        "fault_type",
        "synthetic_label",
        "fault_active",
        score_col,
        prediction_col
    ]

    reading_filter = st.selectbox(
        "Filter reading details",
        [
            "All readings",
            "Modified synthetic readings",
            "Active synthetic fault hours",
            "Model alerts"
        ]
    )

    filtered_table = station_data.copy()

    if reading_filter == "Modified synthetic readings":
        filtered_table = filtered_table[
            filtered_table["synthetic_label"] == 1
        ]
    elif reading_filter == "Active synthetic fault hours":
        filtered_table = filtered_table[
            filtered_table["fault_active"] == 1
        ]
    elif reading_filter == "Model alerts":
        filtered_table = filtered_table[
            filtered_table[prediction_col] == 1
        ]

    st.caption(
        f"Displaying {len(filtered_table):,} readings "
        f"out of {len(station_data):,} in the selected period."
    )

    st.dataframe(
        filtered_table[display_cols],
        use_container_width=True,
        hide_index=True
    )

elif page == "Model Comparison":

    st.header("Compare Machine-Learning Models")

    metric = st.selectbox(
        "Select comparison metric",
        [
            "f1",
            "precision",
            "recall",
            "average_precision"
        ],
        format_func=lambda x: {
            "f1": "F1-score",
            "precision": "Precision",
            "recall": "Recall",
            "average_precision": "Average precision"
        }[x]
    )

    fig = px.bar(
        models.sort_values(metric, ascending=False),
        x="model",
        y=metric,
        color="model",
        labels={
            "model": "Machine-learning model",
            metric: metric.replace("_", " ").title()
        }
    )

    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(
        showlegend=False,
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Test Metrics")
    st.dataframe(
        models,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Station-Level Model Performance")

    station_pivot = d["stations"].pivot(
        index="model",
        columns="station_id",
        values="f1"
    )

    # Station IDs are identifiers, not numerical quantities.
    station_pivot.columns = [
        "Station " + str(s)
        for s in station_pivot.columns
    ]

    fig_station = px.imshow(
        station_pivot,
        text_auto=".2f",
        color_continuous_scale="Blues",
        zmin=0, zmax=1,
        labels={"color": "F1-score"}
    )
    # Allow the heatmap to fill the available width
    fig_station.update_yaxes(scaleanchor=None)
    fig_station.update_layout(
        height=470,
        margin=dict(l=90, r=30, t=35, b=55)
    )

    st.plotly_chart(
        fig_station,
        use_container_width=True
    )

elif page == "Fault Episodes":

    st.header("Synthetic Fault Episode Analysis")

    episodes = d["episodes"].copy()

    selected_fault = st.selectbox(
        "Select fault type",
        [
            "All",
            "Synthetic Spike",
            "Flatline",
            "Stuck at Zero"
        ]
    )

    if selected_fault != "All":
        episodes = episodes[
            episodes["fault_type"] == selected_fault
        ]

    fig = px.bar(
        episodes,
        x="model",
        y="episode_detection_rate",
        color="fault_type",
        barmode="group",
        labels={
            "episode_detection_rate": "Episode detection",
            "fault_type": "Fault type",
            "model": "Model"
        }
    )

    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_layout(height=490)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Episode Detection Details")
    st.dataframe(
        episodes,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "An episode is counted as detected if the model "
        "raises at least one alert during its active period. "
        "This is distinct from hourly modification detection."
    )

    st.divider()
    st.subheader("Investigate an Individual Fault Episode")

    st.write(
        "Select a model, fault type and station to inspect "
        "an injected fault in its rainfall timeline."
    )

    details = d["details"].copy()
    details["station_id"] = details["station_id"].astype(str)

    c1, c2, c3 = st.columns(3)

    investigation_model = c1.selectbox(
        "Investigation model",
        model_names,
        key="episode_model"
    )

    investigation_fault = c2.selectbox(
        "Investigation fault type",
        ["Flatline", "Stuck at Zero", "Synthetic Spike"],
        key="episode_fault"
    )

    available = details[
        (details["model"] == investigation_model)
        & (details["fault_type"] == investigation_fault)
    ].copy()

    investigation_station = c3.selectbox(
        "Investigation station",
        sorted(available["station_id"].unique()),
        key="episode_station"
    )

    available = available[
        available["station_id"] == investigation_station
    ].copy()

    if available.empty:
        st.warning("No episodes match the selected filters.")
    else:
        available["fault_episode_id"] = (
            available["fault_episode_id"].astype(str)
        )

        selected_episode = st.selectbox(
            "Select synthetic fault episode",
            available["fault_episode_id"].tolist(),
            key="selected_fault_episode"
        )

        episode_row = available[
            available["fault_episode_id"] == selected_episode
        ].iloc[0]

        station_rows = readings[
            readings["station_id"] == investigation_station
        ].sort_values("timestamp").copy()

        selected_rows = station_rows[
            station_rows["fault_episode_id"].astype(str)
            == selected_episode
        ]

        selected_rows = selected_rows[
            selected_rows["fault_active"] == 1
        ]

        if selected_rows.empty:
            st.error(
                "Could not match this episode to sensor readings."
            )
        else:
            episode_start = selected_rows["timestamp"].min()
            episode_end = selected_rows["timestamp"].max()

            view_start = episode_start - pd.Timedelta(hours=12)
            view_end = episode_end + pd.Timedelta(hours=12)

            timeline = station_rows[
                station_rows["timestamp"].between(
                    view_start, view_end
                )
            ].copy()

            prediction_col = (
                f"{investigation_model}_prediction"
            )

            alert_rows = timeline[
                timeline[prediction_col] == 1
            ]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Episode duration",
                f"{len(selected_rows)} hours"
            )
            c2.metric(
                "Modified readings",
                int(selected_rows["synthetic_label"].sum())
            )
            c3.metric(
                "Model alert hours",
                int(selected_rows[prediction_col].sum())
            )
            c4.metric(
                "Episode detected",
                "Yes" if int(episode_row["detected"]) else "No"
            )

            fig_episode = go.Figure()

            fig_episode.add_trace(go.Scatter(
                x=timeline["timestamp"],
                y=timeline["original_value"],
                mode="lines+markers",
                name="Original reading",
                line=dict(color="#64748b", width=2),
                marker=dict(size=4)
            ))

            fig_episode.add_trace(go.Scatter(
                x=timeline["timestamp"],
                y=timeline["simulated_value"],
                mode="lines+markers",
                name="Simulated reading",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=4)
            ))

            modified = selected_rows[
                selected_rows["synthetic_label"] == 1
            ]

            fig_episode.add_trace(go.Scatter(
                x=modified["timestamp"],
                y=modified["simulated_value"],
                mode="markers",
                name="Modified readings",
                marker=dict(
                    color="#f59e0b",
                    symbol="x",
                    size=11
                )
            ))

            fig_episode.add_trace(go.Scatter(
                x=alert_rows["timestamp"],
                y=alert_rows["simulated_value"],
                mode="markers",
                name="Model alerts",
                marker=dict(
                    color="#dc2626",
                    symbol="circle-open",
                    size=12,
                    line=dict(width=2)
                )
            ))

            fig_episode.add_vrect(
                x0=episode_start,
                x1=episode_end + pd.Timedelta(hours=1),
                fillcolor="#f59e0b",
                opacity=0.13,
                layer="below",
                line_width=0,
                annotation_text="Injected fault period",
                annotation_position="top left"
            )

            fig_episode.update_layout(
                title=(
                    f"{investigation_fault} — "
                    f"Station {investigation_station}"
                ),
                xaxis_title="Timestamp",
                yaxis_title="Rainfall reading (mm)",
                hovermode="x unified",
                height=480,
                legend=dict(orientation="h", y=-0.25)
            )

            st.plotly_chart(
                fig_episode,
                use_container_width=True
            )

            st.caption(
                "Shaded area marks the active synthetic fault. "
                "Orange crosses mark numerically modified readings; "
                "red circles mark model alerts. "
                "All model thresholds remain unchanged."
            )

            st.subheader("Episode Reading Details")

            episode_display = timeline[[
                "timestamp",
                "original_value",
                "simulated_value",
                "fault_active",
                "synthetic_label",
                prediction_col
            ]].copy()

            st.dataframe(
                episode_display,
                use_container_width=True,
                hide_index=True
            )



elif page == "Historical Investigation":

    st.header("Historical Investigation — Puni Hill Reserve")

    st.warning(
        "UNVERIFIED HISTORICAL DISAGREEMENT: "
        "These observations do not establish that the "
        "Puni Hill rainfall sensor was faulty."
    )

    @st.cache_data
    def load_puni_data():
        hourly = pd.read_csv(
            DATA_DIR / "puni_historical_hourly.csv"
        )
        hourly["timestamp"] = pd.to_datetime(
            hourly["timestamp"]
        )

        stations = pd.read_csv(
            DATA_DIR / "puni_nearby_stations.csv",
            dtype={"station_id": str}
        )

        coverage = pd.read_csv(
            DATA_DIR / "puni_neighbour_coverage.csv",
            dtype={"station_id": str}
        )

        categories = pd.read_csv(
            DATA_DIR / "puni_investigation_categories.csv"
        )

        return hourly, stations, coverage, categories

    hourly, nearby, coverage, categories = load_puni_data()

    hourly = hourly.sort_values("timestamp").copy()

    high_priority = int(
        hourly["investigation_category"].eq(
            "High-priority disagreement"
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Station", "742826")
    c2.metric("Hours investigated", f"{len(hourly):,}")
    c3.metric(
        "Puni recorded rainfall",
        f"{hourly['puni_rainfall_mm'].sum():.1f} mm"
    )
    c4.metric(
        "High-priority hours",
        f"{high_priority:,}"
    )

    st.caption(
        "Investigation period: 28 January – 4 February 2023. "
        "Puni Hill reported zero rainfall throughout the "
        "192-hour period."
    )

    st.divider()
    st.subheader("Interactive Historical Rainfall Timeline")

    all_categories = sorted(
        hourly["investigation_category"]
        .dropna().unique().tolist()
    )

    selected_categories = st.multiselect(
        "Highlight investigation categories",
        options=all_categories,
        default=["High-priority disagreement"]
    )

    show_lstm = st.checkbox(
        "Show earlier exploratory LSTM flags",
        value=False
    )

    plot_data = hourly.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_data["timestamp"],
        y=plot_data["puni_rainfall_mm"],
        name="Puni Hill rainfall",
        mode="lines",
        line=dict(color="#dc2626", width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=plot_data["timestamp"],
        y=plot_data["neighbour_mean_mm"],
        name="Neighbour mean rainfall",
        mode="lines",
        line=dict(color="#2563eb", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=plot_data["timestamp"],
        y=plot_data["neighbour_max_mm"],
        name="Neighbour maximum rainfall",
        mode="lines",
        line=dict(
            color="#64748b",
            width=1.5,
            dash="dot"
        )
    ))

    highlighted = plot_data[
        plot_data["investigation_category"].isin(
            selected_categories
        )
    ]

    fig.add_trace(go.Scatter(
        x=highlighted["timestamp"],
        y=highlighted["neighbour_mean_mm"],
        name="Selected disagreement hours",
        mode="markers",
        marker=dict(
            color="#f59e0b",
            size=9,
            symbol="diamond"
        ),
        customdata=highlighted[
            ["investigation_category"]
        ].to_numpy(),
        hovertemplate=(
            "%{x}<br>"
            "Neighbour mean: %{y:.3f} mm<br>"
            "Category: %{customdata[0]}"
            "<extra></extra>"
        )
    ))

    if show_lstm:
        flagged = plot_data[
            plot_data["lstm_hourly_flag"] == 1
        ]

        fig.add_trace(go.Scatter(
            x=flagged["timestamp"],
            y=flagged["neighbour_mean_mm"],
            name="Exploratory LSTM flags",
            mode="markers",
            marker=dict(
                color="#9333ea",
                size=13,
                symbol="x"
            )
        ))

    fig.update_layout(
        title="Puni Hill vs Neighbouring Rainfall Measurements",
        xaxis_title="Date and time",
        yaxis_title="Hourly rainfall (mm)",
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", y=-0.25)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Neighbour mean and maximum are summaries of "
        "available neighbouring station readings, not "
        "estimates of the true rainfall at Puni Hill. "
        "LSTM flags, when shown, come from an earlier "
        "exploratory analysis."
    )

    st.divider()
    st.subheader("Investigation Category Breakdown")

    category_fig = px.bar(
        categories,
        x="investigation_category",
        y="hours",
        text="hours",
        color="investigation_category",
        labels={
            "investigation_category": "Investigation category",
            "hours": "Number of hours"
        }
    )

    category_fig.update_layout(
        showlegend=False,
        height=370,
        xaxis_tickangle=-15
    )

    st.plotly_chart(
        category_fig,
        use_container_width=True
    )

    st.divider()
    st.subheader("Nearby Monitoring Stations")

    st.write(
        "These stations provide surrounding rainfall context. "
        "Differences can also arise from genuine spatial "
        "variation in rainfall."
    )

    if {"latitude", "longitude"}.issubset(nearby.columns):
        map_data = nearby.dropna(
            subset=["latitude", "longitude"]
        ).copy()

        if not map_data.empty:
            st.map(
                map_data,
                latitude="latitude",
                longitude="longitude"
            )

    st.dataframe(
        nearby,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Station Data Coverage")

    st.dataframe(
        coverage,
        hide_index=True,
        use_container_width=True
    )

    st.divider()
    st.subheader("Hourly Investigation Records")

    category_filter = st.selectbox(
        "Filter historical observations",
        ["All categories"] + all_categories,
        key="puni_category_filter"
    )

    filtered = hourly.copy()

    if category_filter != "All categories":
        filtered = filtered[
            filtered["investigation_category"]
            == category_filter
        ]

    st.write(f"Displaying {len(filtered)} hourly records.")

    st.dataframe(
        filtered,
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "Download filtered historical records",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="puni_historical_investigation_filtered.csv",
        mime="text/csv"
    )

    st.info(
        "Research interpretation: the repeated zero readings "
        "at Puni Hill despite rainfall recorded by surrounding "
        "stations justify further investigation. Verification "
        "would require independent sensor maintenance records, "
        "quality-control evidence or other reliable ground truth. "
        "The historical investigation is separate from the "
        "synthetic-fault six-model evaluation."
    )


st.divider()
st.caption(
    "Rainfall Sensor Anomaly Detection Capstone | "
    "Revised retrospective synthetic-fault evaluation"
)
