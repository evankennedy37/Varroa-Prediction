from clearml import Task
from clearml.automation import PipelineController

#Registering predictor as a template task
task = Task.create(project_name="Varroa Prediction Pipeline", task_name="Predictor")
task.set_script(
    repository="https://github.com/evankennedy37/Varroa-Prediction",
    branch="main",
    entry_point="steps/varroa_predictor.py"
)
task.close()

# Register the prediction pipeline
pipe = PipelineController(name="Prediction Pipeline", project="Varroa Prediction Pipeline", version="1.0")

pipe.add_parameter("trained_model_task_id", default="")  # filled in by user after training
pipe.add_parameter("weather_path", default="/data/weather.csv")
pipe.add_parameter("station_id", default=112250)
pipe.add_parameter("yard_density", default=100)
pipe.add_parameter("yard_elevation", default=500)
pipe.add_parameter("start_date", default='2017-07-21')
pipe.add_parameter("end_date", default='2017-07-28')

pipe.add_step(
    name="predictor",
    base_task_project="Varroa Prediction Pipeline",
    base_task_name="Predictor",
    execution_queue="default",
    parameter_override={
        "General/upstream_id": "${pipeline.trained_model_task_id}",
        "General/weather_path": "${pipeline.weather_path}",
        "General/station_id": "${pipeline.station_id}",
        "General/yard_density": "${pipeline.yard_density}",
        "General/yard_elevation": "${pipeline.yard_elevation}",
        "General/start_date": "${pipeline.start_date}",
        "General/end_date": "${pipeline.end_date}",
    }
)

print("Prediction pipeline registered successfully.")