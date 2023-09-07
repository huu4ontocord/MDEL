from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb


class MyCustomTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False


    def __init__(self, run_name: str):
        self.run_name = run_name
        run = wandb.init(self.run_name)


    def tracker(self):
        return self.run.run


    def store_init_configuration(self, values: dict):
        wandb.config(values)


    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)