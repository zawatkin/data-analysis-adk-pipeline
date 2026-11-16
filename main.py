import os
import requests
import pandas as pd
import plotly.express as px
import boto3
from botocore.exceptions import NoCredentialsError

class WeatherDataAgent:
    """Agentic orchestrator for the weather data pipeline."""
    
    def __init__(self, latitude, longitude, bucket_name):
        self.latitude = latitude
        self.longitude = longitude
        self.bucket_name = bucket_name
        self.api_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m&timezone=auto"
        self.output_html = "weather_visualization.html"

    def extract_data(self):
        """Extracts hourly temperature data from the public weather API."""
        print("Extracting data from API...")
        response = requests.get(self.api_url)
        response.raise_for_status()
        return response.json()

    def process_metrics(self, raw_data):
        """Processes raw JSON into a pandas DataFrame."""
        print("Processing temperature metrics...")
        hourly_data = raw_data.get('hourly', {})
        df = pd.DataFrame({
            'Time': pd.to_datetime(hourly_data.get('time', [])),
            'Temperature (C)': hourly_data.get('temperature_2m', [])
        })
        return df

    def generate_visualization(self, df):
        """Generates a local HTML visualization of the data."""
        print(f"Generating local HTML visualization: {self.output_html}...")
        fig = px.line(df, x='Time', y='Temperature (C)', title='Hourly Temperature Forecast')
        fig.write_html(self.output_html)

    def upload_to_cloud(self):
        """Uploads the generated HTML file to an AWS S3 bucket."""
        print(f"Uploading {self.output_html} to S3 bucket: {self.bucket_name}...")
        s3 = boto3.client('s3')
        try:
            s3.upload_file(self.output_html, self.bucket_name, self.output_html)
            print("Upload successful!")
        except FileNotFoundError:
            print("The file was not found.")
        except NoCredentialsError:
            print("AWS credentials not found. Skipping cloud upload. Please configure AWS CLI.")

    def run_pipeline(self):
        """Executes the pipeline steps in sequence."""
        try:
            raw_data = self.extract_data()
            processed_df = self.process_metrics(raw_data)
            self.generate_visualization(processed_df)
            self.upload_to_cloud()
            print("Pipeline execution complete.")
        except Exception as e:
            print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    # Coordinates for Indianapolis as an example
    LATITUDE = 39.7684
    LONGITUDE = -86.1581
    S3_BUCKET = "your-target-s3-bucket-name"
    
    agent = WeatherDataAgent(LATITUDE, LONGITUDE, S3_BUCKET)
    agent.run_pipeline()
