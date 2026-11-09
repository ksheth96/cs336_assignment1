import torch
from cs336_basics.optimizers import SGD


def main():
    for lr in (1, 10, 100, 1000):
        # Use the same initialization for each learning rate.
        torch.manual_seed(0)
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        sgd_optimizer = SGD(params=[weights], lr=lr)
        print(f"\nLearning rate: {lr}")
        for t in range(10):
            sgd_optimizer.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss.backward()
            sgd_optimizer.step()


if __name__==main():
    main()

