import sys
sys.path.append('.')
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import hydra
import trainers

@hydra.main(version_base="1.2", config_path="../configs", config_name="default")
def main(hydra_cfg):
    trainer = eval(hydra_cfg.trainer)(hydra_cfg)
    trainer.train()

if __name__ == '__main__':
    main()