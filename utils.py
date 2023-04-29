import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, epochs, train_dataloader, optimizer, loss_fn):
    # initially train the model with all the parameters
    total_data = len(train_dataloader)
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_dataloader):
            pred = model(data.to(device))
            loss = loss_fn(pred, label.to(device))

            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Epoch: {epoch} Loss {i} / {total_data}: {torch.linalg.norm(loss)}")

            optimizer.zero_grad()