Title
=====

Dataset of Electric Vehicle Synthetic Trips

Description
===========

This dataset contains csv files of electric vehicles simulated trips using SUMO software. There are 21 different routes and for each route, simulations have been done combining different factors that affect electric vehicle energy consumption. This factors are: driver behaviour, occupancy, auxiliary system consumption, traffic and wind condition.

Each folder of the dataset has the name of origin-destination route. The name of each csv file indicates the simulated conditions, for example:

"1_andoain_zumarraga_0.2_1_200_-1.9444.csv" means "1_origin_destination_trafficFactor_occupancy_auxiliaries_wind.csv"

- trafficFactor: traffic factor used to generate traffic in SUMO.
- occupancy: number of occupants appart from the driver.
- auxiliaries: considered auxiliary system load in Watts.
- wind: wind speed in m/s.

Csv files have the following data:

- vehID: vehicle ID 
- step: simulation step 
- acceleration(m/sÂ²): acceleration
- actualBatteryCapacity(Wh): actual battery capacity
- SoC(%): state of charge
- speed(m/s): speed
- speedFactor: speed factor
- totalEnergyConsumed(Wh): total energy consumed up to that step
- totalEnergyRegenerated(Wh): total energy regenerated up to that step
- lon: longitude
- lat: latitude
- alt: altitude (m)
- slope(Âº): road slope
- completedDistance(km): completed distance up to that step
- mWh: consumption average in m/Wh up to that step
- remainingRange(km): remaining range based on consumption average

Each step represents 1 second in time.
Each vehID represents a vehicle model and a driving style:

- EV0: BMW_i3, defensive style
- EV1: BMW_i3, normal style
- EV2: BMW_i3, aggressive style
- EV3: VW_ID3, defensive style
- EV4: VW_ID3, normal style
- EV5: VW_ID3, aggressive style
- EV6: VW_ID4, defensive style
- EV7: VW_ID4, normal style
- EV8: VW_ID4, aggressive style
- EV9: VW_eUp, defensive style
- EV10: VW_eUp, normal style
- EV11: VW_eUp, aggressive style
- EV12: SUV, defensive style
- EV13: SUV, normal style
- EV14: SUV, aggressive style

reference to vehicle models: https://github.com/eclipse/sumo/tree/main/data/emissions/MMPEVEM

Dataset Download
================

- https://opendatasets.vicomtech.org/di23-devst-dataset-of-electric-vehicle-simulated-trips/d913b9ff

License
=======
All datasets on this page are copyright by Vicomtech and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, for noncommercial purposes only. If others modify or adapt the material, they must license the modified material under identical terms.

Citation
========
If you find this useful for you project or research, consider citing it.
