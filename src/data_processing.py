# data_processing.py

import pandas
import torch
import os
from torch.utils.data import Dataset
# from torch.utils.data import Dataset


class LoadData:
  def __init__(self, split_ratio, rootpath, device, features, model_dim):
    self.split_ratio = split_ratio
    self.rootpath = rootpath
    self.device = device
    self.features = features
    self.model_dim = model_dim

  def load_data(self):
    # get the root path
    dataframe = pandas.read_csv(os.path.join(self.rootpath, "Data", "Raw Data", "Train_data.csv"))
    # fetch real features
    dataframe = dataframe.drop(columns=["data", "prcp", "snow", "wdir", "wpgt", "tsun"])
    # transform dataframe to tensor
    data_tensor = torch.tensor(dataframe.values, dtype=torch.float32)
    return data_tensor

  # MinMaxNormalization
  def normalize_data(self):
    self.data_tensor = self.load_data()
    min_vals = self.data_tensor.min(dim=0).values
    max_vals = self.data_tensor.max(dim=0).values
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized_datatensor = (self.data_tensor - min_vals) / range_vals

    # slip data
    nor_traintensor = normalized_datatensor[: int(len(normalized_datatensor) * (1 - self.split_ratio))]
    nor_testtensor = normalized_datatensor[int(len(normalized_datatensor) * (1 - self.split_ratio)) :]

    # save processed data
    abs_savepath = os.path.join(self.rootpath, "Data", "Processed Data")
    savepath_train = os.path.join(abs_savepath, "Processed Train.csv")
    savepath_test = os.path.join(abs_savepath, "Processed Test.csv")
    pandas.DataFrame(nor_traintensor.numpy()).to_csv(savepath_train, index=False, header=False)
    pandas.DataFrame(nor_testtensor.numpy()).to_csv(savepath_test, index=False, header=False)

    nor_traintensor = nor_traintensor.to(self.device)
    nor_testtensor = nor_testtensor.to(self.device)

    lineartrasform = torch.nn.Linear(in_features=self.features, out_features=self.model_dim).to(self.device)

    nor_traintensor = lineartrasform(nor_traintensor)
    nor_testtensor = lineartrasform(nor_testtensor)

    return nor_traintensor, nor_testtensor


class SetDataset(Dataset):
  def __init__(self, data, seq_len, pred_len):
    self.data = data
    self.seq_len = seq_len
    self.pred_len = pred_len

    # print (f"{self.data.shape} {self.data[:5]}")

  def __len__(self):
    return len(self.data) - self.seq_len - self.pred_len + 1

  def __getitem__(self, idx):
    x = self.data[idx : idx + self.seq_len]
    y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
    # print (x.shape)
    # print (y.shape)
    return x, y


# if __name__ == "__main__":
#    LD = LoadData("")
#    data_tensor = LD.load_data()
#    traindata, testdata, range_value = LD.normalize_data(data_tensor)

# data_processing.py
