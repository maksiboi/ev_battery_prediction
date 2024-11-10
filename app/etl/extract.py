import os
import re
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)


def extract(spark: SparkSession) -> DataFrame:
    # Define the base directory containing all the subdirectories
    base_dir = "data/input_data"

    # DataFrame schema
    schema = StructType(
        [
            StructField("vehID", StringType(), True),
            StructField("step", IntegerType(), True),
            StructField("acceleration(m/s²)", FloatType(), True),
            StructField("actualBatteryCapacity(Wh)", FloatType(), True),
            StructField("SoC(%)", FloatType(), True),
            StructField("speed(m/s)", FloatType(), True),
            StructField("speedFactor", FloatType(), True),
            StructField("totalEnergyConsumed(Wh)", FloatType(), True),
            StructField("totalEnergyRegenerated(Wh)", FloatType(), True),
            StructField("lon", FloatType(), True),
            StructField("lat", FloatType(), True),
            StructField("alt", FloatType(), True),
            StructField("slope(º)", FloatType(), True),
            StructField("completedDistance(km)", FloatType(), True),
            StructField("mWh", FloatType(), True),
            StructField("remainingRange(km)", FloatType(), True),
        ]
    )

    # Get a list of all CSV files with their paths
    csv_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(base_dir)
        for f in filenames
        if f.endswith(".csv")
    ]

    # Regex pattern to extract metadata from the file name
    pattern = re.compile(
        r"(\d+)_(.*?)_(.*?)_(\d+(\.\d+)?)_(\d+)_(\d+)_(-?\d+(\.\d+)?)_output.csv"
    )

    df = None

    for file in csv_files:
        # Extract metadata from the file name
        match = pattern.match(os.path.basename(file))
        if match:
            trip, trafficFactor, occupancy, auxiliaries, wind = (
                f"{match.group(2)}-{match.group(3)}",
                float(match.group(4)),
                int(match.group(6)),
                float(match.group(7)),
                float(match.group(8)),
            )
            # Load the CSV file into a temporary DataFrame
            temp_df = spark.read.csv(file, header=True, schema=schema, sep=";")

            # Add the metadata as new columns to the DataFrame
            temp_df = (
                temp_df.withColumn("trip_plan", lit(trip))
                .withColumn("trafficFactor", lit(trafficFactor))
                .withColumn("occupancy", lit(occupancy))
                .withColumn("auxiliaries", lit(auxiliaries))
                .withColumn("wind", lit(wind))
            )
            # Union the temporary DataFrame with the main DataFrame
        if df is None:
            df = temp_df
        else:
            df = df.union(temp_df)

        # Just for developing purpose
        if df.count() > 100000:
            break

    return df
