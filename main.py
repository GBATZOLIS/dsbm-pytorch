import os
import hydra
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    if cfg.Method == "DSB":
        from run_dsb import run
        return run(cfg)
    elif cfg.Method == "DBDSB":
        from run_dbdsb import run
        return run(cfg)
    elif cfg.Method == "RF":
        from run_rf import run
        return run(cfg)
    elif cfg.Method == "vae":
        from run_vae import run
        return run(cfg)
    else: 
        raise NotImplementedError

if __name__ == "__main__":
    main()