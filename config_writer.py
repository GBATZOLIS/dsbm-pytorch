import hydra
from omegaconf import OmegaConf

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    # Manually set whether to resolve interpolations
    resolve_interpolations = False  # Set to True if you want to resolve interpolations

    # Since Hydra changes the working directory, we save the original directory path
    original_dir = hydra.utils.get_original_cwd()

    # Resolve interpolations if required
    if resolve_interpolations:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = OmegaConf.create(cfg)

    # Building the full path to the save location
    save_path = f"{original_dir}/conf/experiment/cifar10_coupled_vae_imf_full_config.yaml"

    # Saving the configuration to a YAML file
    with open(save_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)

    print(f"Configuration saved to {save_path}")

if __name__ == "__main__":
    main()
