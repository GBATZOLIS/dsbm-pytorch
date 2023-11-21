import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    # Hydra 1.1 or earlier: prevent changing the working directory

    # Relative paths to the specific configuration files
    exp1_path = '/store/CIA/gb511/projects/sbvae/code/conf/experiment/cifar10_coupled_vae_imf.yaml'
    exp2_path = '/store/CIA/gb511/projects/sbvae/code/conf/experiment/cifar10_latent_imf.yaml'

    # Loading the experiment configurations
    exp1_cfg = OmegaConf.load(exp1_path)
    exp2_cfg = OmegaConf.load(exp2_path)

    # Function to compare configurations
    same, different = compare_configs(exp1_cfg, exp2_cfg)

    # Printing same and different configurations
    print_config_differences(same, different)

def compare_configs(cfg1, cfg2):
    same, different = {}, {}
    for key, value in cfg1.items():
        if key in cfg2 and cfg1[key] == cfg2[key]:
            same[key] = cfg1[key]
        elif key in cfg2:
            different[key] = {'cfg1': cfg1[key], 'cfg2': cfg2[key]}
    return same, different

def print_config_differences(same, different):
    print("Same Configurations:")
    for key, value in same.items():
        print(f"{key}: {value}")

    print("\nDifferent Configurations:")
    for key, values in different.items():
        print(f"{key}: cfg1={values['cfg1']}, cfg2={values['cfg2']}")

if __name__ == "__main__":
    main()
