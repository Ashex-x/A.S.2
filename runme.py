import os
import yaml
import random

import torch
from torch.utils.data import DataLoader

from src.data_processing import LoadData, SetDataset
from src.model import Transformer
from src.train import TrainEvaluate
from src import PathInit


def main():
  # Configuration
  random.seed(220)
  rootpath = PathInit("PathInit").get_root()

  config_path = os.path.join(rootpath, "config", "params.yaml")
  with open(config_path, "r") as file:
    config = yaml.safe_load(file)

  split_ratio = config["hyperparameter"]["split_ratio"]
  num_head = config["hyperparameter"]["num_head"]
  model_dim = config["hyperparameter"]["model_dim"]
  max_len = config["hyperparameter"]["max_len"]
  num_layer = config["hyperparameter"]["num_layer"]
  batch_size = config["hyperparameter"]["batch_size"]
  learning_rate = config["hyperparameter"]["learning_rate"]
  epochs = config["hyperparameter"]["epochs"]
  seq_len = config["hyperparameter"]["seq_len"]
  pred_len = config["hyperparameter"]["pred_len"]
  features = config["hyperparameter"]["features"]

  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = torch.device("cpu")

  train_data, test_data = LoadData(split_ratio, rootpath, device, features, model_dim).normalize_data()
  # print(train_data[:5], test_data[:5], rangeval, minval)

  _train_data = SetDataset(train_data, seq_len, pred_len)
  _test_data = SetDataset(test_data, seq_len, pred_len)

  train_data_loader = DataLoader(_train_data, batch_size=batch_size, shuffle=True)
  test_data_loader = DataLoader(_test_data, batch_size=batch_size, shuffle=False)

  model = Transformer(num_head, model_dim, max_len, num_layer, pred_len).to(device)

  criterion = torch.nn.functional.mse_loss
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  model.train()
  for epoch in range(epochs):  # You should have epoch loop
      total_loss = 0.0
      for inputs, targets in train_data_loader:
          # 1. Clear gradients FIRST
          optimizer.zero_grad()
          
          # 2. Ensure proper device placement
          inputs, targets = inputs.to(device), targets.to(device)
          
          # 3. Forward pass
          outputs = model(inputs)
          
          # 4. Verify shapes match exactly
          print(f'Output shape: {outputs.shape}, Target shape: {targets.shape}')
          if outputs.shape != targets.shape:
              # If you need to select first feature from targets:
              targets = targets[:, :, 0].contiguous()  # Make explicit copy
              
          # 5. Loss calculation
          loss = criterion(outputs, targets)
          
          # 6. Backward pass
          loss.backward()
          
          # 7. Optional: gradient clipping
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          
          # 8. Update weights
          optimizer.step()
          
          total_loss += loss.item()
      
      print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data_loader)}')

  return

  for epoch in range(epochs):
    train_loss = TrainEvaluate.train(model, train_data_loader, optimizer, criterion)
    test_loss = TrainEvaluate.evaluate(model, test_data_loader, criterion)
    print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

  # torch.save(model.state_dict(), 'weather_transformer.pth')
  print("Training finished.")


if __name__ == "__main__":
  main()
