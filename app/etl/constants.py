from enum import Enum
from pyspark.sql import functions as F

class DataFrameColumns(Enum):
    VEHICLE_ID = "vehicle_id"
    SIMULATION_STEP = "simulation_step"
    ACCELERATION = "acceleration"
    ACTUAL_BATTERY_CAPACITY_WH = "actual_battery_capacity_wh"
    STATE_OF_CHARGE = "state_of_charge"
    SPEED = "speed"
    SPEED_FACTOR = "speed_factor"
    TOTAL_ENERGY_CONSUMED_WH = "total_energy_consumed_wh"
    TOTAL_ENERGY_REGENERATED_WH = "total_energy_regenerated_wh"
    LON = "lon"
    LAT = "lat"
    ALT = "alt"
    ROAD_SLOPE = "road_slope"
    COMPLETED_DISTANCE = "completed_distance"
    CONSUMPTION_AVERAGE_M_PER_WH = "consumption_average_m_per_wh"
    REMAINING_RANGE = "remaining_range"
    TRIP_PLAN = "trip_plan"
    TRAFFIC_FACTOR = "traffic_factor"
    OCCUPANCY = "occupancy"
    AUXILIARIES = "auxiliaries"
    WIND = "wind"

    @property
    def col(self):
        return F.col(self.value)
