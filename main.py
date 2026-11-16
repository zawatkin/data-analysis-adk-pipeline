import os
import json
import asyncio
import requests
import pandas as pd
import plotly.express as px
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load GOOGLE_API_KEY and AWS credentials from local .env
load_dotenv()

# Configuration
LATITUDE = 39.7684   # Indianapolis, IN
LONGITUDE = -86.1581
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "your-target-s3-bucket-name")

RAW_FILE = "raw_weather.json"
PROCESSED_FILE = "processed_daily_weather.csv"
HTML_FILE = "weather_visualization.html"


# --- ADK Pipeline Tools ---

def ingest_weather_data(latitude: float = LATITUDE, longitude: float = LONGITUDE) -> str:
    """Fetches hourly temperature forecast data from the public Open-Meteo API and saves raw JSON."""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}&hourly=temperature_2m&timezone=auto"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    raw_data = response.json()

    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    record_count = len(raw_data.get("hourly", {}).get("time", []))
    return f"Successfully ingested {record_count} hourly records into {RAW_FILE}."


def transform_weather_data() -> str:
    """Filters, enriches, and groups hourly weather data into daily aggregated summary statistics."""
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    hourly = raw_data.get("hourly", {})
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly.get("time", [])),
        "temp_c": hourly.get("temperature_2m", []),
    })

    # 1. Filtering: Drop any missing readings
    df = df.dropna(subset=["temp_c"]).copy()

    # 2. Enrichment: Convert Celsius to Fahrenheit and extract calendar date
    df["temp_f"] = (df["temp_c"] * 9 / 5) + 32
    df["date"] = df["timestamp"].dt.date

    # 3. Grouping & Aggregation: Compute daily min, max, mean, and reading count
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
    return f"Processed {len(df)} valid hourly rows into {len(daily_summary)} daily summary rows in {PROCESSED_FILE}."


def upload_data_to_cloud(bucket_name: str = S3_BUCKET) -> str:
    """Uploads raw JSON and processed CSV datasets to AWS S3 cloud object storage."""
    s3 = boto3.client("s3")
    uploaded = []
    try:
        for file_path in [RAW_FILE, PROCESSED_FILE]:
            s3.upload_file(file_path, bucket_name, f"weather_pipeline/{file_path}")
            uploaded.append(file_path)
        return f"Uploaded {uploaded} to s3://{bucket_name}/weather_pipeline/."
    except NoCredentialsError:
        return "AWS credentials not configured locally; skipped S3 upload (local files preserved)."
    except Exception as e:
        return f"Cloud storage upload notice: {e}"


def build_visualization() -> str:
    """Generates an interactive HTML chart visualizing the aggregated daily temperature trends."""
    df = pd.read_csv(PROCESSED_FILE)
    fig = px.line(
        df,
        x="date",
        y=["min_temp_f", "avg_temp_f", "max_temp_f"],
        markers=True,
        title="Daily Temperature Forecast Summary (°F)",
        labels={"value": "Temperature (°F)", "date": "Date", "variable": "Metric"},
    )
    fig.write_html(HTML_FILE)
    return f"Generated interactive visualization at {HTML_FILE}."


# --- Google ADK Agent Definition ---

pipeline_agent = Agent(
    name="weather_data_pipeline_agent",
    model="gemini-3.8-flash",
    description="Orchestrates an end-to-end weather data engineering pipeline.",
    instruction=(
        "You are a data engineering pipeline orchestrator. Execute the pipeline tools in strict order: "
        "1) Call `ingest_weather_data` to pull raw API data. "
        "2) Call `transform_weather_data` to filter and aggregate the data into daily metrics. "
        "3) Call `upload_data_to_cloud` to store the raw and processed data in cloud storage. "
        "4) Call `build_visualization` to generate the HTML dashboard. "
        "After all four tools succeed, provide a concise summary of the pipeline execution."
    ),
    tools=[
        ingest_weather_data,
        transform_weather_data,
        upload_data_to_cloud,
        build_visualization,
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
        parts=[types.Part(text="Run the end-to-end weather data pipeline now.")],
    )

    print("Starting ADK Data Engineering Pipeline...\n")
    async for event in runner.run_async(
        user_id="student_user", session_id=session.id, new_message=user_message
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(run_agentic_pipeline())
