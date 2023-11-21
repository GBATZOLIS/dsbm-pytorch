import argparse
import yaml
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def main(obs_config_path, latent_config_path):
    obs_cfg = load_config(obs_config_path)
    latent_cfg = load_config(latent_config_path)
    
    from run_combined_dbdsb import run
    return run(obs_cfg, latent_cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DBDSB method with specified configurations")
    parser.add_argument('--obs-config', type=str, required=True, help='Path to the first YAML configuration file')
    parser.add_argument('--latent-config', type=str, required=True, help='Path to the second YAML configuration file')
    args = parser.parse_args()

    obs_config_path = os.path.join('conf', 'experiment', args.obs_config)
    latent_config_path = os.path.join('conf', 'experiment', args.latent_config)

    main(obs_config_path, latent_config_path)
