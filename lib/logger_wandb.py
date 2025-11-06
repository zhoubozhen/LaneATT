import wandb

def init_wandb(exp_name, cfg):
    wandb.init(
        project="LaneATT-TuSimple",
        name=exp_name,
        config=cfg
    )

def log_metrics(metrics, step):
    wandb.log(metrics, step=step)
