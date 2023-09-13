import hydra
import os
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
# from tkinter.tix import Tree
from typing import Sequence
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import time
import ai_core
import ai_core.tools
import ai_core.api
import json
import sys
import datetime as dt
import mlflow

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(HydraConfig.get().runtime.output_dir)
    # hydra.job.override_dirname('Latest')

    ############# DFP specific block
    creation_did = cfg.creation_did
    window_size = int(cfg.window_size)
    init_dir = "/Users/arjun.gullbadhar@zebra.com/experiment_1_TS"
    # init_dir = init_dir.replace("/dbfs","dbfs:")
    runtime_oppath = HydraConfig.get().runtime.output_dir
    runtime_oppath = runtime_oppath.replace("/dbfs","dbfs:")
    print(creation_did)
    print(window_size)

    base_path = "dbfs:/mnt/qa1datamartstdsandbox/qa1datamartstdsandbox-ds-store-std-sandbox-rw"


    # change the input path before executing
    input_df_path = f'{base_path}/ML_Ops_Exp/Input/ANN_Input.parquet'
    forecast_out_time_series = f'{runtime_oppath}/Output_data/TS_Output.parquet'
    ts_model_path = f'{runtime_oppath}/Output_data/model'


    git_root = '/Workspace/Repos/arjun.gullbadhar@zebra.com/MLO_test'
    print(input_df_path)
    print(forecast_out_time_series)

    format_params = {
        "input_df_path":input_df_path,
        "forecast_out_time_series": forecast_out_time_series, 
        "ts_model_path": ts_model_path,
        "creation_did" : creation_did,
        "window_size": window_size,
        "runtime_oppath": runtime_oppath,
        "init_dir": init_dir,
        "git_root": git_root,
        "base_path": base_path
            }

    #############################################################################################
    try:
        experiment_id = mlflow.create_experiment(init_dir)
        mlflow.set_experiment(init_dir)
        print('===============')
        print('Try block')
    except:
        mlflow.set_experiment(init_dir)
        print('===============')
        print('except block')

    with mlflow.start_run(run_name=str(HydraConfig.get().job.num)):
        mlflow.log_params(cfg)
        mlflow.log_param("Output_Parquet_Path",forecast_out_time_series)
        mlflow.end_run()


    
    
    t0 = time.time()
    json_name = f"{git_root}/TS_Testing/ts.json"

    if True:
        # print format_params
        if isinstance(format_params, dict):
            print(json.dumps(format_params, sort_keys=False, indent=4))
        else:
            print(format_params)
        

        # Load stages
        job_json = json_name
        print(job_json)
        print(f"Running job_json: {job_json}")
        stages = ai_core.tools.load_stages(f"{job_json}", format_params, sequence='DFP')


    # Create Spark session

        spark = SparkSession.builder.getOrCreate()
        sparkConf = SparkConf()
        print("Spark session created")
        platform = ai_core.platforms.DatabricksPlatform()

        # Run pipeline
            
        ai_core.api.run(platform=platform, spark=spark, stages_list=stages['DFP']['stages'])

    elapsed_time = time.time() - t0
    print("----- Total time ----")
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
if __name__ == "__main__":
    my_app()
