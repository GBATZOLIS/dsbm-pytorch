import hydra
from omegaconf import OmegaConf

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    # Set to False to allow adding new keys
    OmegaConf.set_struct(cfg, False)

    # Manually set whether to resolve interpolations
    resolve_interpolations = False  # Set to True if you want to resolve interpolations

    # Since Hydra changes the working directory, we save the original directory path
    original_dir = hydra.utils.get_original_cwd()

    # Resolve interpolations if required
    if resolve_interpolations:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

    # Check if job configuration keys are included
    job_keys = ["job", "run", "sweep", "job_logging"]
    for key in job_keys:
        if key not in cfg:
            print(f"Warning: '{key}' key is not in the final configuration.")
    
    cfg.run_dir = f'./{cfg.paths.experiments_dir_name}/{cfg.name}/{cfg.run_name}'
    print(cfg.run_dir)

    # Building the full path to the save location
    save_path = f"{original_dir}/conf/experiment/{cfg.run_name}_full_config.yaml"

    # Saving the configuration to a YAML file
    with open(save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    print(f"Configuration saved to {save_path}")

if __name__ == "__main__":
    main()
