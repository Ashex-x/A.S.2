import data_processing
import __init__
import Predict

import os
import yaml
import torch

def main(self):
    # get config
    rootpath = __init__.PathInit('PathInit').get_root()

    config_path = os.path.join(rootpath, 'Config', 'params.yaml') 
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    split_ratio = config['hyperparameter']['split_ratio']
    num_head = config['hyperparameter']['num_head']
    model_dim = config['hyperparameter']['model_dim']
    max_len = config['hyperparameter']['max_len']
    num_layer = config['hyperparameter']['num_layer']
    pre_len = config['hyperparameter']['pre_len']
    
    traindata, testdata, rangeval, minval = data_processing.LoadData(split_ratio, rootpath).normalize_data()
    #print(traindata[:5], testdata[:5], rangeval, minval)

    Predict.Transformer(num_head, model_dim, max_len, num_layer, pre_len)


if __name__ == "__main__":
    main("")