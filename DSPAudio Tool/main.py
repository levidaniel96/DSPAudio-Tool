import torch
import hydra
from config import create_data_config 
from hydra.core.config_store import ConfigStore
from create_data import create_MyDataset
print("finished import ")
print("PyTorch has version {}".format(torch.__version__))
print("cuda is available? {}" .format(torch.cuda.is_available()))




cs = ConfigStore.instance()
cs.store(name="create_data_config", node=create_data_config) 
@hydra.main(config_path = "conf", config_name = "config")

def main(cfg: create_data_config): 
    create_MyDataset(cfg.paths,cfg.params,cfg.flags)
if __name__ == '__main__':
    print('start')
    main()
    
    