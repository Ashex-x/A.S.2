from torch.utils.data import DataLoader
import os
import yaml
import torch

import data_processing
import __init__
import Predict
import train

def main():
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
    batch_size = config['hyperparameter']['batch_size']
    learning_rate = config['hyperparameter']['learning_rate']
    epochs = config['hyperparameter']['epochs']
    seq_len = config['hyperparameter']['seq_len']
    pred_len = config['hyperparameter']['pred_len']
    features = config['hyperparameter']['features']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    traindata, testdata = data_processing.LoadData(split_ratio, rootpath, device, features, model_dim).normalize_data()
    # print(traindata[:5], testdata[:5], rangeval, minval)

    _traindata = data_processing.SetDataset(traindata, seq_len, pred_len)
    _testdata = data_processing.SetDataset(testdata, seq_len, pred_len)
    
    traindata_loader = DataLoader(_traindata, batch_size=batch_size, shuffle=True)
    testdata_loader = DataLoader(_testdata, batch_size=batch_size, shuffle=False)

    model = Predict.Transformer(num_head, model_dim, max_len, num_layer, pred_len).to(device)

    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = train.Train_Evaluate.Train(model, traindata_loader, optimizer, criterion)
        test_loss = train.Train_Evaluate.evaluate(model, testdata_loader, criterion)
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}')
    
    torch.save(model.state_dict(), 'weather_transformer.pth')
    print("Training finish.")

if __name__ == "__main__":
    main()