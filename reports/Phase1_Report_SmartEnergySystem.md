# Smart Energy Prediction System | Data Mining Final Project | Phase 1 Report

## Smart Energy Prediction System
### Household Energy Consumption Analysis & Anomaly Detection

**Data Mining Course | Phase 1 Report | April 2026**  
**Team:** Omar Abdulaal

---

# 1. Domain Understanding

## 1.1 Business Problem

Egypt is currently navigating one of its most challenging energy periods in recent history. The combination of regional conflict, rising fuel costs, and surging demand has placed enormous pressure on the national power grid.

The Egyptian government has enacted mandatory load-shedding schedules and launched public campaigns to reduce household energy consumption. In this context, the ability to analyze, predict, and optimize household power usage is not merely an academic exercise — it is a matter of national economic resilience.

Despite these pressures, most households lack visibility into their own energy consumption patterns. They receive a monthly bill but have no insight into which appliances drive peak usage, when consumption is highest, or how much energy is wasted. A data-driven approach can fill this gap.

## 1.2 Why Data Mining?

Data mining is uniquely suited to this domain because energy consumption is a rich, multi-variate, time-series signal.

### Data Mining Enables:
1. Segmenting usage into behavioral clusters
2. Predicting future consumption
3. Detecting anomalies
4. Mining association rules

## 1.3 Research Questions

1. What are the dominant daily and seasonal usage patterns in a household?
2. Can we accurately predict short-term energy consumption?
3. Are there identifiable anomalous consumption events?
4. Which appliance categories contribute most to peak load?

---

# 2. Dataset Selection & Preview

## 2.1 Dataset Overview

| Attribute | Details |
|---|---|
| Dataset Name | Individual Household Electric Power Consumption |
| Source | UCI Machine Learning Repository / Kaggle |
| Collection Period | December 2006 – November 2010 |
| Sampling Rate | 1 measurement per minute |
| Total Records | 2,075,259 rows |
| Features | 9 |
| Missing Values | ~1.25% |
| File Format | CSV |

## 2.2 Feature Description

| Feature | Unit | Description |
|---|---|---|
| global_active_power | kW | Total household active power consumption |
| voltage | V | Minute-averaged supply voltage |
| global_intensity | A | Total current intensity |
| sub_metering_1 | Wh | Kitchen appliances |
| sub_metering_2 | Wh | Laundry appliances |
| sub_metering_3 | Wh | Water heater and AC |

## 2.3 Known Data Quality Issues

1. Missing values
2. Data type mismatch
3. Date/time split columns
4. No weather context
5. Single-household dataset limitation

---

# 3. CRISP-DM Plan

| Phase | Activities |
|---|---|
| Business Understanding | Define objectives and success metrics |
| Data Understanding | Explore distributions and identify outliers |
| Data Preparation | Handle missing values and engineer features |
| Modelling | Apply ML algorithms |
| Evaluation | Compare models using metrics |
| Deployment | Build Streamlit dashboard |

---

# 4. Technique Selection & Rationale

## 4.1 Clustering — K-Means & DBSCAN

Cluster daily consumption profiles to discover behavioral segments.

## 4.2 Regression — Ridge Regression & Polynomial Regression

Predict future energy consumption using historical patterns.

## 4.3 Anomaly Detection — PCA & Isolation Forest

Detect unusual energy usage events and spikes.

## 4.4 Classification — Random Forest

Classify high-consumption vs normal-consumption hours.

---

# 5. GitHub Repository & Environment

## Repository Structure
- /data
- /notebooks
- /dashboard
- /reports
- README.md
- requirements.txt

## Python Environment
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- streamlit

---

# 6. Anticipated Limitations & Mitigations

1. Geographic mismatch
2. Single household limitation
3. No external variables
4. Missing values

---

# 7. References

- UCI Machine Learning Repository
- CRISP-DM methodology papers
- Data Mining research publications
