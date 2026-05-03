#Making a ClearML Task
from clearml import Task

task = Task.init(project_name="Varroa Prediction Pipeline", task_name="Predictor")

#Getting model
upstream_id = task.get_parameter("General/upstream_id")

classifier = Task.get_task(task_id=upstream_id).artifacts["trained_model"].get()

#Getting params
data_path = task.get_parameter("General/weather_data_path") or "/data/weather.csv"
station_id = task.get_parameter("General/station_id") or 112250
yard_density = task.get_parameter("General/yard_density") or 100
yard_elevation = task.get_parameter("General/yard_elevation") or 500
start_date = task.get_parameter("General/start_date") or '2017-07-21'
end_date = task.get_parameter("General/end_date") or '2017-07-28'


# %% [markdown]
# ## Varroa Infestation Prediction - Predictor

# %% [markdown]
# #### Authors: Evan Kennedy, Mit Ghandi, Anbuchelvan Kumaravel

# %% [markdown]
# **Library Imports**

# %%
import pandas as pd
import numpy as np
import duckdb
from xgboost import XGBClassifier

# %% [markdown]
# **Prepping Weather Data**

# %%
#Importing specified weather data
weather_data = pd.read_csv(data_path)

#Converting -9999 values used in the dataset to indicate missing values into properly identifiable NaN values
weather_proper = weather_data.replace(-9999, np.nan)

#Combining date and hour fields into a single datetime field
weather_dated = weather_proper.copy()
weather_dated['datetime'] = pd.to_datetime(weather_dated['date'] + ' ' + weather_dated['hour'])
weather_dated = weather_dated.drop(columns=['date','hour'])

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
# **Building Tensors**

# %%
#Making the base dataframe
model_input = pd.DataFrame(columns = ['date_from','elevation', 'yard_density', 'station_id'])

#Setting up dataframe w/o the weather data, to be joined with the weather data
dates = pd.date_range(start=start_date, end=end_date, freq='D')
for day in dates:
    model_input.loc[len(model_input)] = [day, yard_density, yard_elevation, station_id]

# %%
#Merging the dataframes to form the tensors
fields = ['air_temp', 'dew_point', 'pressure', 'wind_dir', 'wind_spd', 'datetime']

#Generate 30 columns per field using the 30 most recent preceding readings
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
        FROM model_input v
        LEFT JOIN weather_nonmissing w
            ON w.station_id = v.station_id
            AND w.datetime < v.date_from
    )
    SELECT
        v.*,
        {day_cols_str}
    FROM model_input v
    LEFT JOIN ranked_weather w
        ON w.station_id = v.station_id
        AND w.date_from = v.date_from
        AND w.rn <= 30
    GROUP BY v.*
    ORDER BY v.date_from --guarantees stable row order
"""
combined_data = duckdb.query(query).df()

# %%
#Converting all the datetime fields for weather into time difference fields (in hours), 
#then converting 'date_from' into a year, month, & day field. 
#(Necessary so that all fields are usable as integer values)
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

# %%
#Converting day & month into cyclical patterns so the model can more easily understand that 
#December precedes January & that the 31st of a month precedes the 1st of the next month
combined_ready = combined_date_adjusted.copy()

#Month
combined_ready["obs_month_sin"] = np.sin(2 * np.pi * combined_ready["obs_month"] / 12)
combined_ready["obs_month_cos"] = np.cos(2 * np.pi * combined_ready["obs_month"] / 12)

#Day of the Month
combined_ready["obs_day_sin"] = np.sin(2 * np.pi * combined_ready["obs_day"] / 31)
combined_ready["obs_day_cos"] = np.cos(2 * np.pi * combined_ready["obs_day"] / 31)

# %%
#Dropping 'station_id' column
ready_input = combined_ready.drop(columns=['station_id'])
ready_input.insert(0, 'Unnamed: 0', range(len(ready_input)))

# %% [markdown]
# **Using the Model**

# %%
#Making predictions
preds = classifier.predict(ready_input).astype(bool)
probs = classifier.predict_proba(ready_input)

# %%
#Reporting & Logging predictions
outputs = []
dates = pd.date_range(start=start_date, end=end_date, freq='D')
i = 0
for day in dates:
    #Getting prediction
    if preds[i] == False:
        prediction = 'Safe'
    else:
        prediction = 'Infestation'

    #Logging
    outputs = outputs + [{'date': day.strftime("%Y-%m-%d"), 'prediction': prediction, 'infestation probability': (str(probs[i,1]*100) + ' %')}]

    #Printing
    print(f'[{day.strftime("%Y-%m-%d")}]\nPrediction: {prediction}\n Infestation Probability: {probs[i,1]*100} %\n')
    i = i + 1

#Saving predictions
task.upload_artifact("predictions", artifact_object=outputs)


