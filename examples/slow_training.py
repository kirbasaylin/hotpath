import torch
from torch.utils.data import DataLoader


def train(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters())

    for batch in loader:
        inputs = batch["x"].to(device)
        labels = batch["y"].cuda()
        mask = torch.zeros((inputs.shape[0], inputs.shape[1]))
        copied = torch.tensor(inputs)

        query = model.q(inputs)
        key = model.k(inputs)
        value = model.v(inputs)
        scores = torch.matmul(query, key.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, value)

        loss = ((output * mask.to(device)) - labels + copied).pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.save(model.state_dict(), "checkpoint.pt")
        torch.cuda.empty_cache()
        history = loss.detach().cpu().numpy()
        print(loss.item())


def validate(model, inputs):
    model.eval()
    return model(inputs)
