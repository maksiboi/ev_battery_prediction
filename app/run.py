from pyspark.sql import SparkSession
from app.etl.extract import extract
from app.etl.transform import transform, validate_data
from pyspark.sql import functions as F


def main():
    spark = SparkSession.builder.appName("Battery Monitoring").getOrCreate()

    # ETL
    df = extract(spark)
    df = transform(df)
    df = validate_data(df)

    df.printSchema()
    df.filter(F.col("vehicle_id") == "EV0").show(1, vertical=True)


if __name__ == "__main__":
    main()
