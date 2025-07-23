import torch

class Train_Evaluate:
    def Train(model, dataloader, optimizer, criterion):
        # model.Train()
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print (f'{outputs.size()} {targets.size()}')
            loss = criterion(outputs, targets.squeeze(-1))
            # loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def evaluate(model, dataloader, criterion):
        # model.evaluate()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.squeeze(-1))
                total_loss += loss.item()
        return total_loss / len(dataloader)