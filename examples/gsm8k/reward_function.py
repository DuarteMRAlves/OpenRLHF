import re
import torch

from math_verify import parse, verify


def reward_func(queries, batch):
    # queries is prompts + responses
    return torch.tensor([_reward_func(q, s) for q, s in zip(queries, batch)])

def _reward_func(query_w_answer, sample):
    format_reward = _format_reward_func(query_w_answer, sample)
    answer_reward = _answer_reward_func(query_w_answer, sample)
    return 0.2 * format_reward + 0.8 * answer_reward


def _format_reward_func(query_w_completion, sample):
    prompt = sample["prompt"]
    prompt_no_think = prompt[:-len("<think> ")].strip()
    completion = query_w_completion[len(prompt_no_think):].strip()
    format_regex = r"^<think>.*<\/think>\s*<answer>.*<\/answer>$"
    match = re.match(format_regex, completion, re.DOTALL)
    if match is None:
        return 0.0
    return 1.0


def _answer_reward_func(query_w_completion, sample):
    try:
        prompt = sample["prompt"]
        gold = sample["answer"]
        prompt_no_think = prompt[:-len("<think> ")].strip()
        completion = query_w_completion[len(prompt_no_think):].strip()
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        
        answer = match.group(1).strip()

        gold = parse(gold)
        answer = parse(answer)

        if verify(gold, answer):
            return 1.0
        return 0.0
    except Exception as e:
        print("Error in reward function for completion:", completion)
        print(e)
        return 0.0