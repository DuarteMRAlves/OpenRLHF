import torch

def reward_func(queries, prompts):
    # queries is prompts + responses
    print(queries)
    print(prompts)
    answers = queries[len(prompts):]
    print(answers)
    return torch.randn(len(queries))
