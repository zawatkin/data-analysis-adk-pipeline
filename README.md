# Data Analysis ADK Pipeline

Agent-driven Python pipeline for extracting, processing, and visualizing weather API data.

## Overview
This project acts as an automated agent that:
1. **Extracts** hourly temperature data from the public Open-Meteo API.
2. **Processes** the JSON metrics into a structured Pandas DataFrame.
3. **Visualizes** the data by generating an interactive local HTML line chart using Plotly.
4. **Uploads** the final HTML output to an AWS S3 cloud storage bucket.

## Setup Instructions
1. Ensure you have Python 3.8+ installed.
2. Clone this repository.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
