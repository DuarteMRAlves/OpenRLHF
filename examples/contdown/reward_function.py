import re
import torch

def reward_func(queries, batch):
    # queries is prompts + responses
    return torch.tensor([_reward_func(q, s) for q, s in zip(queries, batch)])

def _reward_func(query_w_answer, sample):
    format_reward = _format_reward_func(query_w_answer, sample)
    #print(f"Reward: {format_reward}")
    return format_reward


def _format_reward_func(query_w_completion, sample):
    prompt = sample["prompt"]
    completion = query_w_completion[len(prompt):]
    #print(f"Completion: {completion}")
    format_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer><|im_end|>$"
    match = re.match(format_regex, completion, re.DOTALL)
    if match is None or len(match.groups()) != 2:
        return 0.0
    return 1.0
