#Making a process in ClearML
from clearml import Task

task = Task.init(project_name="Varroa Prediction Pipeline", task_name="Data Processor")

varroa_path = task.get_parameter("General/varroa_samples_path") or "/data/varroa_sampling.csv"
yard_path = task.get_parameter("General/yard_info_path") or "/data/yard.csv"
weather_path = task.get_parameter("General/weather_data_path") or "/data/weather.csv"


# %% [markdown]
# ## Varroa Infestation Prediction - Data Processor

# %% [markdown]
# #### Authors: Evan Kennedy, Mit Ghandi, Anbuchelvan Kumaravel

# %% [markdown]
# **Library Imports**

# %%
import pandas as pd
import numpy as np
import duckdb

# %% [markdown]
# **Data Imports**

# %%
#Reading in data
varroa_import = pd.read_csv(varroa_path)
yard_import = pd.read_csv(yard_path)
weather_import = pd.read_csv(weather_path)

# %%
#Grabbing desired columns
varroa_samples = varroa_import[['sampling_id', 'date_from', 'date_to', 'varroa_count', 'hive_id', 'yard_id']]
yard = yard_import[['yard_id', 'elevation', 'station_id']]
weather_data = weather_import[['station_id', 'date', 'hour', 'air_temp', 'dew_point', 'pressure', 'wind_dir', 'wind_spd']]

# %%
#Merging yard data into varroa data
varroa_combined = pd.merge(varroa_samples, yard, how='left')

# %% [markdown]
# **Prepping Hive Data**

# %%
#Adding in field that counts the number of hives per yard
varroa_complete = varroa_combined.copy()
varroa_complete['yard_density'] = varroa_complete.groupby('yard_id')['hive_id'].transform('nunique')

# %%
#Cutting out samples not including both dates
varroa_trimmed = varroa_complete.dropna(axis=0, subset=['date_from', 'date_to'])

# %%
#Changing any dates starting with a 000 to start with a 201 to ensure proper conversion
varroa_dated = varroa_trimmed.copy()
varroa_dated['date_from'] = varroa_dated['date_from'].str.replace('^000', '201', regex=True)
varroa_dated['date_to'] = varroa_dated['date_to'].str.replace('^000', '201', regex=True)

#Converting dates to datetime type
varroa_dated['date_from'] = pd.to_datetime(varroa_dated['date_from'])
varroa_dated['date_to'] = pd.to_datetime(varroa_dated['date_to'])

#Adding in sampling length field
varroa_dated['sample_length'] = (varroa_dated['date_to'] - varroa_dated['date_from']).dt.days

# %%
#Revising prior cuts to only care about the starting date, as that determines what weather data best precedes it
varroa_proper_trimmed = varroa_complete.dropna(axis=0, subset=['date_from'])
varroa_proper_trimmed = varroa_proper_trimmed.drop(columns=['date_to'])

# %%
#Changing any dates starting with a 000 to start with a 201 to ensure proper conversion
varroa_proper_dated = varroa_proper_trimmed.copy()
varroa_proper_dated['date_from'] = varroa_proper_dated['date_from'].str.replace('^000', '201', regex=True)

#Converting dates to datetime type
varroa_proper_dated['date_from'] = pd.to_datetime(varroa_proper_dated['date_from'])

# %% [markdown]
# **Prepping Weather Data**

# %%
#Converting -9999 values used in the dataset to indicate missing values into properly identifiable NaN values
weather_proper = weather_data.replace(-9999, np.nan)

#Checking missing value counts
#print(weather_proper.isna().sum())
#Checking non-missing value counts
#print(weather_proper.notna().sum())

# %%
#Combining date and hour fields into a single datetime field
weather_dated = weather_proper.copy()
weather_dated['datetime'] = pd.to_datetime(weather_dated['date'] + ' ' + weather_dated['hour'])
weather_dated = weather_dated.drop(columns=['date','hour'])

# %%
#Handling missing values in [air_temp, dew_point, pressure, wind_dir, wind_spd] by replacing with the 
#temporally closest observation value for that same station
weather_nonmissing = weather_dated.copy()
#Sorting by station_id & datetime to use ffill() & bfill() to replace missing values by the closest observation easily
weather_nonmissing = weather_nonmissing.sort_values(by=['station_id', 'datetime'])
#Using ffill() & bfill() to fill in missing values from the observation w/ the closest datetime, from the same station
fields = ['air_temp', 'dew_point', 'pressure', 'wind_dir', 'wind_spd']
for field in fields:
    weather_nonmissing[field] = weather_nonmissing.groupby('station_id')[field].ffill().bfill()

# %% [markdown]
# **Combining the Datasets**

# %%
# Using DuckDB for speed (Pandas approaches proved prohibitively slow)
fields = ['air_temp', 'dew_point', 'pressure', 'wind_dir', 'wind_spd', 'datetime']

# Generate 30 columns per field using the 30 most recent preceding readings
day_columns = []
for f in fields:
    for i in range(1, 31):
        col = f"""
            MAX(CASE WHEN w.rn = {i} THEN w.{f} END) AS {f}_obs_{i}"""
        day_columns.append(col)

day_cols_str = ', '.join(day_columns)

query = f"""
    WITH ranked_weather AS (
        SELECT
            w.*,
            v.date_from,
            ROW_NUMBER() OVER (
                PARTITION BY w.station_id, v.date_from
                ORDER BY w.datetime DESC
            ) AS rn
        FROM varroa_proper_dated v
        LEFT JOIN weather_nonmissing w
            ON w.station_id = v.station_id
            AND w.datetime < v.date_from
    )
    SELECT
        v.*,
        {day_cols_str}
    FROM varroa_proper_dated v
    LEFT JOIN ranked_weather w
        ON w.station_id = v.station_id
        AND w.date_from = v.date_from
        AND w.rn <= 30
    GROUP BY v.*
    ORDER BY v.sampling_id --guarantees stable row order
"""
combined_data = duckdb.query(query).df()

# %% [markdown]
# The following check & drop are to cover for the possibility of varroa samplings in the dataset lacking weather data preceeding them, a problem that arose in the dataset the pipeline was originally generated for. 
# 
# Ideally, carefully selected input data should not suffer this missingness, in which case the following 2 cells are redundant.

# %%
#print(combined_data.isna().sum().to_string())

# %% [markdown]
# For the original dataset: Additional analysis not included here found that these ~(2/5) of the samples had a valid weather station associated with them in the original dataset, but that the weather data for the dates preceeding them were missing. Given this, we were unable to use these samplings.

# %%
#Dropping rows with missing values
combined_data.dropna(inplace=True)

# %% [markdown]
# **Prepping Data for Modeling**

# %% [markdown]
# Converting all the datetime fields for weather into time difference fields (in hours), then converting 'date_from' into a year, month, & day field. (Necessary so that all fields are usable as integer values)

# %%
combined_date_adjusted = combined_data.copy()

#Datetime fields to time preceding fields
for i in range(1,31):
    combined_date_adjusted["time_diff_obs_" + str(i)] = (
    (combined_date_adjusted["date_from"] - combined_date_adjusted["datetime_obs_" + str(i)].min()).dt.total_seconds() / 3600
    )
    combined_date_adjusted.drop(columns=["datetime_obs_" + str(i)], inplace=True)

#Converting 'date_from' to a year, month, and day field
combined_date_adjusted["obs_year"] = combined_date_adjusted["date_from"].dt.year
combined_date_adjusted["obs_month"] = combined_date_adjusted["date_from"].dt.month
combined_date_adjusted["obs_day"] = combined_date_adjusted["date_from"].dt.day
combined_date_adjusted.drop(columns=["date_from"], inplace=True)

# %% [markdown]
# Converting day & month into cyclical patterns so the model can more easily understand that December precedes January & that the 31st of a month precedes the 1st of the next month

# %%
combined_ready = combined_date_adjusted.copy()

#Month
combined_ready["obs_month_sin"] = np.sin(2 * np.pi * combined_ready["obs_month"] / 12)
combined_ready["obs_month_cos"] = np.cos(2 * np.pi * combined_ready["obs_month"] / 12)

#Day of the Month
combined_ready["obs_day_sin"] = np.sin(2 * np.pi * combined_ready["obs_day"] / 31)
combined_ready["obs_day_cos"] = np.cos(2 * np.pi * combined_ready["obs_day"] / 31)

# %% [markdown]
# For defining 'action needed' on Varroa mites, when sampling with a drop-tray (as is the case for our data), a rate of 30 mites/day is generally seen as the point of action being needed. However, given that our data often lacks the necessary additional date marker to know the sampling period duration, & that even with that information 30 mites/day is far less valuable a metric when the board is not counted daily, we'll instead go with a simpler 'alarm bell' point of 100 mites.
# 
# If desired, this threshold can be changed easily simply by inputting a different value in place of 100 in the following cell

# %% [markdown]
# Preparing Version of the data w/ a binary classification instead of a Varroa Count

# %%
#Converting varroa_count to binary >= 100 mites field
combined_binary = combined_ready.copy()
combined_binary['infested'] = combined_binary['varroa_count'] >= 100
combined_binary.drop(columns=['varroa_count'], inplace=True)

# %% [markdown]
# Exporting Data

# %%
#Saving data used
task.upload_artifact("processed_tensors", artifact_object=combined_binary)


