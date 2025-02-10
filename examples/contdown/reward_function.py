import torch

def reward_func(queries, batch):
    # queries is prompts + responses
    print("Reward function called")
    print(batch)
    return torch.randn(len(queries))
