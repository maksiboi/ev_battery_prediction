from pyspark.sql import DataFrame

from pyspark.sql import functions as F


def transform(df: DataFrame) -> DataFrame:
    # Rename the columns to more user-friendly names
    df = (
        df.withColumnRenamed("vehID", "vehicle_id")
        .withColumnRenamed("step", "simulation_step")
        .withColumnRenamed("acceleration(m/s²)", "acceleration")
        .withColumnRenamed("actualBatteryCapacity(Wh)", "actual_battery_capacity_wh")
        .withColumnRenamed("SoC(%)", "state_of_charge")
        .withColumnRenamed("speed(m/s)", "speed")
        .withColumnRenamed("speedFactor", "speed_factor")
        .withColumnRenamed("totalEnergyConsumed(Wh)", "total_energy_consumed_wh")
        .withColumnRenamed("totalEnergyRegenerated(Wh)", "total_energy_regenerated_wh")
        .withColumnRenamed("lon", "lon")
        .withColumnRenamed("lat", "lat")
        .withColumnRenamed("alt", "alt")
        .withColumnRenamed("slope(º)", "road_slope")
        .withColumnRenamed("completedDistance(km)", "completed_distance")
        .withColumnRenamed("mWh", "consumption_average_m_per_wh")
        .withColumnRenamed("remainingRange(km)", "remaining_range")
        .withColumnRenamed("trafficFactor", "traffic_factor")
    )

    # Convert the speed from m/s to km/h
    df = df.withColumn("speed", F.col("speed") * 3.6)

    return df


def validate_data(df):
    # 1. Check for non-null values in mandatory columns
    for column in df.columns:
        df = df.filter(F.col(column).isNotNull())

    # 2. Range checks:
    df = df.filter(
        (F.col("speed") >= 0)  # Speed should be positive
        & (F.col("state_of_charge").between(0, 100))  # SoC should be between 0 and 100
        & (F.col("remaining_range") >= 0)  # Remaining range should be >= 0
        & (
            F.col("road_slope").between(-90, 90)
        )  # Road slope should be between -90 and 90 degrees
    )

    # 3. Ensure total energy consumed is less than or equal to the actual battery capacity
    df = df.filter(
        F.col("total_energy_consumed_wh") <= F.col("actual_battery_capacity_wh")
    )

    # 4. Valid coordinates (latitude and longitude should be within valid ranges)
    df = df.filter(
        (F.col("lat").between(-90, 90))  # Latitude should be between -90 and 90
        & (F.col("lon").between(-180, 180))  # Longitude should be between -180 and 180
    )

    # 5. Check for duplicates based on vehicle_id and simulation_step
    df = df.dropDuplicates(["vehicle_id", "simulation_step", "trip_plan"])

    return df
