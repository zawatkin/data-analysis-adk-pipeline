import os
import json
import asyncio
from typing import AsyncGenerator
import requests
import pandas as pd
import plotly.express as px
from google.cloud import storage
from dotenv import load_dotenv

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load GCS configuration from .env (no Gemini API key required)
load_dotenv(override=True)

# Pipeline Configuration
LATITUDE = 39.7684   # Indianapolis, IN
LONGITUDE = -86.1581
LOCATION_NAME = "Indianapolis, IN"
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "zawatkin-weather-pipeline-2026")

RAW_FILE = "raw_weather.json"
PROCESSED_FILE = "processed_daily_weather.csv"
HTML_FILE = "weather_visualization.html"


# --- Specialized ADK Pipeline Agents (Deterministic BaseAgent Subclasses) ---

class IngestionAgent(BaseAgent):
    """ADK Agent 1: Fetches raw hourly forecast data from the Open-Meteo public API."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={LATITUDE}&longitude={LONGITUDE}&hourly=temperature_2m&timezone=auto"
        )
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        raw_data = response.json()

        with open(RAW_FILE, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)

        record_count = len(raw_data.get("hourly", {}).get("time", []))
        msg = f"[1/4 IngestionAgent] Ingested {record_count} hourly weather records into '{RAW_FILE}'."

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=msg)]),
            actions=EventActions(state_delta={"raw_file": RAW_FILE, "raw_records": record_count}),
        )


class TransformationAgent(BaseAgent):
    """ADK Agent 2: Cleans, enriches, and aggregates hourly records into daily temperature metrics."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        raw_path = ctx.session.state.get("raw_file", RAW_FILE)
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        hourly = raw_data.get("hourly", {})
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly.get("time", [])),
            "temp_c": hourly.get("temperature_2m", []),
        })

        # 1. Filtering: Remove missing readings
        df = df.dropna(subset=["temp_c"]).copy()

        # 2. Enrichment: Convert Celsius to Fahrenheit and extract date
        df["temp_f"] = (df["temp_c"] * 9 / 5) + 32
        df["date"] = df["timestamp"].dt.date

        # 3. Grouping & Aggregation: Calculate daily min, max, average, and count
        daily_summary = (
            df.groupby("date", as_index=False)
            .agg(
                min_temp_f=("temp_f", "min"),
                max_temp_f=("temp_f", "max"),
                avg_temp_f=("temp_f", "mean"),
                hourly_readings=("temp_f", "count"),
            )
            .round(2)
        )

        daily_summary.to_csv(PROCESSED_FILE, index=False)
        msg = (
            f"[2/4 TransformationAgent] Transformed {len(df)} hourly rows into "
            f"{len(daily_summary)} daily summary rows in '{PROCESSED_FILE}'."
        )

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=msg)]),
            actions=EventActions(state_delta={"processed_file": PROCESSED_FILE, "daily_rows": len(daily_summary)}),
        )


class StorageAgent(BaseAgent):
    """ADK Agent 3: Uploads raw JSON and processed CSV datasets to Google Cloud Storage."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        raw_path = ctx.session.state.get("raw_file", RAW_FILE)
        processed_path = ctx.session.state.get("processed_file", PROCESSED_FILE)

        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            uploaded = []
            for file_path in [raw_path, processed_path]:
                blob = bucket.blob(f"weather_pipeline/{file_path}")
                blob.upload_from_filename(file_path)
                uploaded.append(file_path)
            msg = f"[3/4 StorageAgent] Uploaded {uploaded} to gs://{GCS_BUCKET}/weather_pipeline/."
            cloud_status = "uploaded"
        except Exception as e:
            msg = f"[3/4 StorageAgent] Cloud upload notice (local datasets preserved): {e}"
            cloud_status = "local_fallback"

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=msg)]),
            actions=EventActions(state_delta={"cloud_status": cloud_status}),
        )


class VisualizationAgent(BaseAgent):
    """ADK Agent 4: Generates an interactive HTML dashboard website from the processed dataset."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        processed_path = ctx.session.state.get("processed_file", PROCESSED_FILE)
        df = pd.read_csv(processed_path)

        fig = px.line(
            df,
            x="date",
            y=["min_temp_f", "avg_temp_f", "max_temp_f"],
            markers=True,
            title=f"7-Day Temperature Forecast Trend — {LOCATION_NAME} (°F)",
            labels={"value": "Temperature (°F)", "date": "Date", "variable": "Metric"},
        )
        fig.update_layout(template="plotly_white", hovermode="x unified")
        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        weekly_high = df["max_temp_f"].max()
        weekly_low = df["min_temp_f"].min()
        weekly_avg = round(df["avg_temp_f"].mean(), 2)
        total_readings = int(df["hourly_readings"].sum())

        table_rows = "\n".join(
            f"<tr>"
            f"<td>{row['date']}</td>"
            f"<td>{row['min_temp_f']:.2f} °F</td>"
            f"<td>{row['avg_temp_f']:.2f} °F</td>"
            f"<td>{row['max_temp_f']:.2f} °F</td>"
            f"<td>{int(row['hourly_readings'])}</td>"
            f"</tr>"
            for _, row in df.iterrows()
        )

        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Weather Data Engineering Pipeline Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #f4f6f9;
            color: #1f2937;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1050px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            color: white;
            padding: 24px 32px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 14px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }}
        .kpi-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: #6b7280;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: 700;
            color: #111827;
        }}
        .card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            margin-bottom: 24px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
        }}
        th, td {{
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{
            background-color: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        tr:hover {{
            background-color: #f9fafb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Agentic Weather Data Pipeline Dashboard</h1>
            <p>Location: {LOCATION_NAME} ({LATITUDE}, {LONGITUDE}) | Source: Open-Meteo API | Bucket: gs://{GCS_BUCKET}/weather_pipeline/</p>
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Forecast High</div>
                <div class="kpi-value">{weekly_high:.1f} °F</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Forecast Low</div>
                <div class="kpi-value">{weekly_low:.1f} °F</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Temp</div>
                <div class="kpi-value">{weekly_avg:.1f} °F</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Hourly Records</div>
                <div class="kpi-value">{total_readings}</div>
            </div>
        </div>

        <div class="card">
            {chart_html}
        </div>

        <div class="card">
            <h2 style="margin-top: 0; font-size: 18px;">Daily Aggregated Weather Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Min Temperature</th>
                        <th>Avg Temperature</th>
                        <th>Max Temperature</th>
                        <th>Hourly Readings</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

        with open(HTML_FILE, "w", encoding="utf-8") as f:
            f.write(html_page)

        msg = f"[4/4 VisualizationAgent] Built interactive web dashboard at '{HTML_FILE}'."
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=msg)]),
            actions=EventActions(state_delta={"html_file": HTML_FILE}),
        )


# --- Multi-Agent ADK Sequential Workflow Orchestrator ---

ingestion_agent = IngestionAgent(
    name="ingestion_agent",
    description="Fetches raw hourly weather data from the Open-Meteo API.",
)
transformation_agent = TransformationAgent(
    name="transformation_agent",
    description="Cleans and aggregates hourly weather records into daily summaries.",
)
storage_agent = StorageAgent(
    name="storage_agent",
    description="Uploads raw and processed datasets to Google Cloud Storage.",
)
visualization_agent = VisualizationAgent(
    name="visualization_agent",
    description="Generates an interactive HTML dashboard from processed weather data.",
)

pipeline_agent = SequentialAgent(
    name="weather_data_pipeline_agent",
    description="Orchestrates the 4-stage weather data engineering workflow sequentially.",
    sub_agents=[
        ingestion_agent,
        transformation_agent,
        storage_agent,
        visualization_agent,
    ],
)


async def run_agentic_pipeline():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="weather_etl_app", user_id="student_user"
    )
    runner = Runner(
        agent=pipeline_agent,
        app_name="weather_etl_app",
        session_service=session_service,
    )

    user_message = types.Content(
        role="user",
        parts=[types.Part(text="Run the end-to-end weather data engineering pipeline.")],
    )

    print("Starting Multi-Agent ADK Weather Data Pipeline...\n")
    async for event in runner.run_async(
        user_id="student_user", session_id=session.id, new_message=user_message
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)

    print("\nPipeline completed! Opening visualization website...")


if __name__ == "__main__":
    asyncio.run(run_agentic_pipeline())
