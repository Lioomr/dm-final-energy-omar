# Smart Energy Consumption Analytics System

## Overview

This project analyzes household electricity consumption using data mining techniques to understand usage patterns, predict consumption, detect abnormal spikes, and support energy-saving decisions.

The project is motivated by household energy efficiency needs, especially in contexts where peak electricity demand and load management are important.

## Objectives

- Analyze household energy consumption patterns.
- Predict short-term electricity usage.
- Detect abnormal consumption spikes.
- Classify high vs normal energy usage.
- Present results through a notebook and an interactive dashboard.

## Dataset

- **Name:** Household Electric Power Consumption
- **Source:** UCI / Kaggle
- **Records:** 2M+ in the original public dataset
- **Features:** 9 raw time-series and power-metering features
- **Current working CSV:** The local file in `DataSet/` is analyzed as-is. The notebook and dashboard display its actual date coverage automatically.

## Required Deliverables

The course brief requires a full Jupyter notebook workflow. This project satisfies that requirement with an ordered CRISP-DM notebook set:

1. `notebooks/01_Business_Understanding.ipynb`
2. `notebooks/02_Data_Understanding.ipynb`
3. `notebooks/03_Data_Preparation.ipynb`
4. `notebooks/04_Modelling.ipynb`
5. `notebooks/05_Evaluation.ipynb`
6. `notebooks/06_Deployment.ipynb`

Run the notebooks in order from `01` to `06`. Together, they form the full Jupyter notebook deliverable.

The Python files in `src/` are helper modules used by the Streamlit dashboard and earlier experiments. The final notebook workflow is organized in the six notebooks above.

## Techniques Used

- **Clustering:** K-Means, DBSCAN
- **Regression:** Linear Regression, Ridge Regression, Polynomial Ridge
- **Classification:** Logistic Regression, Random Forest
- **Anomaly Detection:** Isolation Forest, PCA analysis

## Methodology

The project follows the CRISP-DM framework:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment through a Streamlit dashboard

## Project Structure

```text
DataSet/       Raw household power consumption CSV
dashboard/     Streamlit dashboard
notebooks/     Ordered CRISP-DM notebook deliverable
outputs/       Generated files, figures, and logs
reports/       Project reports and presentation files
src/           Reusable Python preparation and modeling code
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebooks:

```bash
jupyter notebook
```

Then run the notebooks in order from `01_Business_Understanding.ipynb` to `06_Deployment.ipynb`.

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

For Streamlit Community Cloud, use this entrypoint:

```text
streamlit_app.py
```

## Author

Omar Abdulaal
