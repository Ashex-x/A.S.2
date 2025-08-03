import torch


class TrainEvaluate:
  def train(model, dataloader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    for inputs, targets in dataloader:
      outputs = model(inputs)
      print(f'{outputs.size()} {targets.size()}')
      
      loss = criterion(outputs, targets[:, :, 0])
      loss.backward()

      print(loss.item())

      optimizer.step()

      optimizer.zero_grad()
      total_loss += loss.item()

    return total_loss / len(dataloader)

  def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
      for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)
