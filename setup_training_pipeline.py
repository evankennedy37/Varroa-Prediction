from clearml import Task
from clearml.automation import PipelineController

#Register pipeline steps as template tasks
steps = [
    ("Data Processor", "steps/varroa_data_processor.py"),
    ("Model Fitter",  "steps/varroa_model_fitter.py"),
]

#Registering steps as template tasks
for task_name, entry_point in steps:
    task = Task.init(project_name="Varroa Prediction Pipeline", task_name=task_name)
    task.set_script(
        repository="https://github.com/evankennedy37/Varroa-Prediction",
        branch="main",
        entry_point=entry_point
    )
    task.close()

#Register the training pipeline
pipe = PipelineController(name="Training Pipeline", project="Varroa Prediction Pipeline", version="1.0")

pipe.add_parameter("varroa_samples_path", default="/data/varroa_sampling.csv")
pipe.add_parameter("yard_info_path", default="/data/yard.csv")
pipe.add_parameter("weather_data_path", default="/data/weather.csv")

pipe.add_step(
    name="data_processor",
    base_task_project="Varroa Prediction Pipeline",
    base_task_name="Data Processor",
    execution_queue="default",
    parameter_override={
        "General/varroa_samples_path": "${pipeline.varroa_samples_path}",
        "General/yard_info_path": "${pipeline.yard_info_path}",
        "General/weather_data_path": "${pipeline.weather_data_path}",
    }
)

pipe.add_step(
    name="model_fitter",
    parents=["data_processor"],
    base_task_project="Varroa Prediction Pipeline",
    base_task_name="Model Fitter",
    execution_queue="default",
    parameter_override={
        "General/upstream_id": "${data_processor.id}",
    }
)

pipe.create(
    project_name="My Project",
    task_name="My Pipeline"
)
print("Training pipeline registered successfully.")