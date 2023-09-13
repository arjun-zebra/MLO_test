import mlflow
import hydra
from hydra import utils
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from pathlib import Path
###################################

# import hydra
# c = []
# hydra.main(config_name="config.yaml")(lambda x:c.append(x))()
# cfg = c[0]
# print(cfg)

@hydra.main(config_path="conf", config_name='config_test.yaml')
def main(cfg):
    print('==================')
    print(cfg)
    print('==================')
    # mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')


    experiment_id = mlflow.create_experiment(
    cfg.experiment_name.A,
    artifact_location=Path.cwd().joinpath("mlruns").joinpath(cfg.experiment_name.A).as_uri(),
    # tags={"version": "v1", "priority": "P1"},
    )
    experiment = mlflow.get_experiment(experiment_id)
    print(experiment)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    print("Creation timestamp: {}".format(experiment.creation_time))
    
    mlflow.set_experiment(experiment.name)
    print(hydra.utils.get_original_cwd())
    print(experiment.artifact_location)

    with mlflow.start_run() as run:
        mlflow.log_params(cfg)
        mlflow.log_artifact(Path.cwd() / '.hydra/config.yaml')
        print(run.info.run_id) # just to show each run is different
        mlflow.end_run()

if __name__ == '__main__':
    main()