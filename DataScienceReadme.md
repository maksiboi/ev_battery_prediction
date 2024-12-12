# Battery Monitoring and Prediction: Electric Vehicle Dataset

## Overview

This repository contains the code and data pipeline for analyzing and predicting electric vehicle (EV) metrics based on simulated driving data. The goal is to predict and classify various EV-related parameters such as remaining battery range, energy consumption, and battery range classification. The data used for these predictions comes from simulated trips of electric vehicles under varying conditions (e.g., wind, traffic, and occupancy).

### Project Structure

```
.
|____run.py
|____etl
| |____constants.py
| |____campagins
| | |____campaing.py
| |____transform.py
| |____extract.py
```

- **`run.py`**: The entry point of the ETL pipeline and campaign execution. It orchestrates the extraction, transformation, validation, and campaign execution.
- **`etl/`**: Contains the modules for extraction, transformation, and campaign logic.
  - **`constants.py`**: Holds constants and column mappings for the dataset.
  - **`extract.py`**: Extracts the data from CSV files and adds additional metadata.
  - **`transform.py`**: Transforms the data, renames columns, and applies necessary computations.
  - **`campagins/campaing.py`**: Contains the logic for different campaigns (regression and classification tasks) and saves the data as Parquet files.

## Data Description

The dataset is composed of synthetic EV trip data generated using the **SUMO** traffic simulation software. It includes various factors that affect energy consumption, including driver behavior, traffic conditions, road slope, and weather conditions. The dataset is stored in CSV files, each representing a simulated trip with specific conditions.

Each folder in the dataset corresponds to a unique **origin-destination route**, and the name of each CSV file encodes the traffic factor, occupancy, auxiliary system load, and wind conditions.

### CSV File Format

The data in each CSV file includes the following columns:

- **vehID**: Vehicle ID (e.g., `EV0`, `EV1`, etc.)
- **step**: Simulation step (1-second intervals)
- **acceleration (m/s²)**: Vehicle acceleration
- **actualBatteryCapacity (Wh)**: Battery capacity in watt-hours
- **SoC (%)**: State of charge
- **speed (m/s)**: Vehicle speed in meters per second
- **speedFactor**: Factor applied to vehicle speed in the simulation
- **totalEnergyConsumed (Wh)**: Total energy consumed up to the current step
- **totalEnergyRegenerated (Wh)**: Total energy regenerated up to the current step
- **lon**: Longitude of the vehicle
- **lat**: Latitude of the vehicle
- **alt**: Altitude of the vehicle
- **slope (°)**: Road slope
- **completedDistance (km)**: Distance completed up to the current step
- **mWh**: Energy consumption in mWh
- **remainingRange (km)**: Remaining driving range based on current consumption rate

### Vehicle IDs and Driving Styles

Each `vehID` corresponds to a specific EV model and driving style. For example:

- **EV0**: BMW_i3, defensive driving style
- **EV1**: BMW_i3, normal driving style
- **EV2**: BMW_i3, aggressive driving style
- **EV3**: VW_ID3, defensive driving style
- **EV4**: VW_ID3, normal driving style
- **EV5**: VW_ID3, aggressive driving style
- **EV6**: VW_ID4, defensive driving style
- **EV7**: VW_ID4, normal driving style
- **EV8**: VW_ID4, aggressive driving style

### Data Storage Format

The raw data, after extraction and transformation, is stored in **Parquet** format for efficient processing. The data is split into **training** and **testing** sets (70%/30%), and the Parquet files are stored in separate directories for each campaign.

#### Parquet File Naming Structure

Each campaign generates two files:

- **train_data.parquet**: Training data
- **test_data.parquet**: Testing data

These files are saved in the `data/output_data/{campaign_name}/` directory, where `{campaign_name}` corresponds to the type of analysis or prediction (e.g., `campaign1`, `campaign2`, etc.).

---

## Pipeline Overview

### 1. **Data Extraction** (`extract.py`)

The data extraction step loads all CSV files from the `data/input_data` directory and adds relevant metadata based on the file name. This metadata includes trip information, traffic factors, occupancy, auxiliary load, and wind conditions. The extraction process uses **PySpark** for distributed processing.

### 2. **Data Transformation** (`transform.py`)

The transformation process includes the following:

- **Renaming columns** to more user-friendly names (e.g., `vehID` becomes `vehicle_id`, `SoC(%)` becomes `state_of_charge`).
- **Unit conversion**: The speed is converted from meters per second (m/s) to kilometers per hour (km/h).
- **Data validation**: The data is validated to ensure there are no null values and that values fall within acceptable ranges (e.g., speed should be non-negative, SoC should be between 0 and 100, etc.).

### 3. **Campaigns** (`campagins/campaing.py`)

Once the data is prepared, it is used in different predictive campaigns. The main goal of each campaign is to predict or classify key EV metrics based on the input features. The campaigns include:

#### - **Campaign 1: Remaining Range Prediction (Regression)**

Predict the **remaining battery range** based on several input features. This campaign uses the following columns:

- `vehicle_id`, `acceleration`, `actual_battery_capacity_wh`, `state_of_charge`, `speed`, `total_energy_consumed_wh`, `total_energy_regenerated_wh`, `completed_distance`, `traffic_factor`, `wind`, `remaining_range`

#### - **Campaign 2: Energy Consumption Prediction (Regression)**

Predict the **total energy consumption** or the **average energy consumption** up to that point in time. This campaign uses the following columns:

- `vehicle_id`, `speed`, `acceleration`, `road_slope`, `auxiliaries`, `traffic_factor`, `wind`, `total_energy_consumed_wh`

#### - **Campaign 3: Battery Range Classification (Classification)**

Classify the EV's battery range into three categories:

- Short range (< 50 km)
- Medium range (50-150 km)
- Long range (> 150 km)

This campaign uses the following columns:

- `vehicle_id`, `state_of_charge`, `speed`, `acceleration`, `completed_distance`, `alt`, `road_slope`, `wind`, `traffic_factor`, `occupancy`, `auxiliaries`, `remaining_range`

### 4. **Saving Data as Parquet**

For each campaign, the data is split into **training** and **testing** sets (70%/30%), and these sets are saved as Parquet files in the `data/output_data/{campaign_name}/` directory. The Parquet files are optimized for storage and can be loaded into **PySpark** or **Pandas** (via **PyArrow**) for further analysis.

---

## How to Use

1. **Run the ETL Pipeline**: 

To run the entire pipeline (extract, transform, validate, and save), simply execute the following command:

```bash
chmod +x start.sh
start.sh
```

2. **Load Parquet Data**: 

You can load the resulting Parquet files using **PySpark** or **PyArrow**. Here's an example using **PyArrow**:

```python
import pyarrow.parquet as pq
import pandas as pd
# Load the Parquet file
table = pq.read_table('data/output_data/campaign1/train_data.parquet')

# Convert to Pandas DataFrame
df = table.to_pandas()
```

---

## Notes for Data Scientists

- The data columns have been renamed to be more intuitive (e.g., `vehID` is now `vehicle_id`, `SoC(%)` is now `state_of_charge`, etc.).
- You can use the saved **Parquet** files for training and testing machine learning models.
- For further preprocessing and feature engineering, you can modify the pipeline or directly use the transformed data.
- The data is already split into training and testing sets, making it ready for machine learning workflows.

