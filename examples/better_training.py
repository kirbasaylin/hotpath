import torch
from torch.utils.data import DataLoader


def train(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters())

    for step, batch in enumerate(loader):
        inputs = batch["x"].to(device, non_blocking=True)
        labels = batch["y"].to(device, non_blocking=True)
        mask = torch.zeros((inputs.shape[0], inputs.shape[1]), device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(inputs)
            loss = ((output * mask) - labels).pow(2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            pass


def validate(model, inputs):
    model.eval()
    with torch.inference_mode():
        return model(inputs)
