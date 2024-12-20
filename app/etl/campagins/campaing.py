from pyspark.sql import DataFrame
import app.etl.constants as constants

class BatteryMonitoringCampaigns:
    def __init__(self, df: DataFrame):
        self.df = df

    def _save_parquet(self, model_df: DataFrame, campaign_name: str) -> None:
        """
        Helper method to split the DataFrame into train and test sets (70%/30%),
        and save them as Parquet files.
        
        :param model_df: The DataFrame to split.
        :param campaign_name: The name of the campaign to generate the folder names.
        """
        # Split the data into 70% train and 30% test
        train_df, test_df = model_df.randomSplit([0.7, 0.3], seed=42)

        # Save the train and test DataFrames as Parquet files
        train_df.write.parquet(f"data/output_data/{campaign_name}/train_data.parquet", mode="overwrite")
        test_df.write.parquet(f"data/output_data/{campaign_name}/test_data.parquet", mode="overwrite")
        
    def left_range_campaign(self) -> None:
        """ Prediction of remaining battery range (regression) """
        model_df = self.df.select(
            constants.DataFrameColumns.VEHICLE_ID.col,
            constants.DataFrameColumns.TRIP_PLAN.col,
            constants.DataFrameColumns.SIMULATION_STEP.col,
            constants.DataFrameColumns.ACCELERATION.col,
            constants.DataFrameColumns.ACTUAL_BATTERY_CAPACITY_WH.col,
            constants.DataFrameColumns.STATE_OF_CHARGE.col,
            constants.DataFrameColumns.SPEED.col,
            constants.DataFrameColumns.TOTAL_ENERGY_CONSUMED_WH.col,
            constants.DataFrameColumns.TOTAL_ENERGY_REGENERATED_WH.col,
            constants.DataFrameColumns.COMPLETED_DISTANCE.col,
            constants.DataFrameColumns.TRAFFIC_FACTOR.col,
            constants.DataFrameColumns.WIND.col,
            constants.DataFrameColumns.REMAINING_RANGE.col
        )

        # Use the helper method to save the train and test data
        self._save_parquet(model_df, "campaign1")

    def energy_consumption(self) -> None:
        """ Prediction of energy consumption, i.e., the average energy consumption (mWh) """
        model_df = self.df.select(
            constants.DataFrameColumns.VEHICLE_ID.col,
            constants.DataFrameColumns.TRIP_PLAN.col,
            constants.DataFrameColumns.SIMULATION_STEP.col,
            constants.DataFrameColumns.SPEED.col,
            constants.DataFrameColumns.ACCELERATION.col,
            constants.DataFrameColumns.ROAD_SLOPE.col,
            constants.DataFrameColumns.AUXILIARIES.col,
            constants.DataFrameColumns.TRAFFIC_FACTOR.col,
            constants.DataFrameColumns.WIND.col,
            constants.DataFrameColumns.TOTAL_ENERGY_CONSUMED_WH.col
        )

        # Use the helper method to save the train and test data
        self._save_parquet(model_df, "campaign2")

    def battery_range_classification(self) -> None: 
        """ Battery range classification (classification) """
        model_df = self.df.select(
            constants.DataFrameColumns.VEHICLE_ID.col,
            constants.DataFrameColumns.TRIP_PLAN.col,
            constants.DataFrameColumns.SIMULATION_STEP.col,
            constants.DataFrameColumns.STATE_OF_CHARGE.col,
            constants.DataFrameColumns.SPEED.col,
            constants.DataFrameColumns.ACCELERATION.col,
            constants.DataFrameColumns.COMPLETED_DISTANCE.col,
            constants.DataFrameColumns.ALT.col,
            constants.DataFrameColumns.ROAD_SLOPE.col,
            constants.DataFrameColumns.WIND.col,
            constants.DataFrameColumns.TRAFFIC_FACTOR.col,
            constants.DataFrameColumns.OCCUPANCY.col,
            constants.DataFrameColumns.AUXILIARIES.col,
            constants.DataFrameColumns.REMAINING_RANGE.col
        )

        # Use the helper method to save the train and test data
        self._save_parquet(model_df, "campaign3")
