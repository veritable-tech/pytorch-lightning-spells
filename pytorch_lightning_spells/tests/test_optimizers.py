import torch
from pytorch_lightning_spells.optimizers import RAdam


def test_RAdam():
    model = torch.nn.Linear(6, 1)
    optimizer = RAdam(model.parameters())
    inputs = torch.rand(6)
    loss = 1 - model(inputs)
    print(loss)
    for _ in range(3):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_new = 1 - model(inputs)
        print(loss_new)
        assert loss_new <= loss
        loss = loss_new
