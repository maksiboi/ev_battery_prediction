from pyspark.sql import SparkSession
from app.etl.extract import extract
from app.etl.transform import transform, validate_data
from app.etl.campagins.campaing import BatteryMonitoringCampaigns  
from pyspark.sql import functions as F


def main():
    spark = SparkSession.builder.appName("Battery Monitoring").getOrCreate()

    # ETL
    df = extract(spark)
    df = transform(df)
    df = validate_data(df)
    
    # Initialize the campaigns class with the DataFrame
    campaigns = BatteryMonitoringCampaigns(df)
    
    # Call the class methods
    campaigns.left_range_campaign()
    campaigns.energy_consumption()
    campaigns.battery_range_classification()


if __name__ == "__main__":
    main()
