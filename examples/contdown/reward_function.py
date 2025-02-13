import re
import torch

def reward_func(queries, batch):
    # queries is prompts + responses
    return torch.tensor([_reward_func(q, s) for q, s in zip(queries, batch)])

def _reward_func(query_w_answer, sample):
    format_reward = _format_reward_func(query_w_answer, sample)
    used_reward = _used_numbers_reward(query_w_answer, sample)
    evaluable_reward = _evaluable_reward_func(query_w_answer, sample)
    target_reward = _target_reward_func(query_w_answer, sample)
    return 0.2 * format_reward + 0.2 * used_reward + 0.2 * evaluable_reward + 0.4 * target_reward


def _format_reward_func(query_w_completion, sample):
    prompt = sample["prompt"]
    completion = query_w_completion[len(prompt):]
    format_regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer><|im_end|>$"
    match = re.match(format_regex, completion, re.DOTALL)
    if match is None or len(match.groups()) != 2:
        return 0.0
    return 1.0

def _used_numbers_reward(query_w_completion, sample):
    try:
        prompt = sample["prompt"]
        numbers = sample["nums"]
        completion = query_w_completion[len(prompt):]
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        
        equation = match.group(1).strip()

        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            return 0.0
        return 1.0
    except Exception as e:
        print("Error in reward function for completion:", completion)
        print(e)
        return 0.0
    

def _evaluable_reward_func(query_w_completion, sample):
    try:
        prompt = sample["prompt"]

        completion = query_w_completion[len(prompt):]
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        
        equation = match.group(1).strip()
        
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})

        result = float(result)

        return 1.0
    except Exception as e:
        print("Error in reward function for completion:", completion)
        print(e)
        return 0.0


def _target_reward_func(query_w_completion, sample):
    try:
        prompt = sample["prompt"]
        numbers = sample["nums"]
        target = float(sample["target"])
        
        completion = query_w_completion[len(prompt):]
        match = re.search(r"<answer>(.*?)<\/answer>", completion)
        if match is None:
            return 0.0
        
        equation = match.group(1).strip()

        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        
        # Check if answer only contains valid numbers
        if not all(n in numbers for n in used_numbers):
            return 0.0
        
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        diff = abs(float(result) - target)
        if diff < 1e-5:
            return 1.0
        return 0.0
    except Exception as e:
        print("Error in reward function for completion:", completion)
        print(e)
        return 0.0