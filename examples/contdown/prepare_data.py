import argparse
import datasets

task_description = """You will be given a list of numbers and a target number. Your task is to generate a valid mathematical expression using all the numbers exactly once, and only the basic operations (+, -, ×, ÷) to reach the target number. Parentheses can be used to enforce operation order.

Before producing the final answer, think through the steps inside <think> </think> and then provide the solution inside <answer> </answer>.

Input-Output Examples

Example 1

Input:
Numbers: [4, 2, 6, 5]
Target: 38

Output:
<think>
Start with the largest numbers: 6 × 5 = 30
Now, use the remaining numbers: 4 × 2 = 8
Adding both results: 30 + 8 = 38
</think>
<answer>
(6 × 5) + (4 × 2)
</answer>

Example 2

Input:
Numbers: [25, 50, 8, 3]
Target: 200

Output:
<think>
Multiply 50 × 3 to get 150
Multiply 25 × 2 to get 50
Adding both results: 150 + 50 = 200
</think>
<answer>
(50 × 3) + (25 × 2)
</answer>

Example 3

Input:
Numbers: [7, 2, 10, 5]
Target: 45

Output:
<think>
Multiply 10 × 5 to get 50
Multiply 7 × 2 to get 14
Subtracting both results: 50 - 14 = 45
</think>
<answer>
(10 × 5) - (7 × 2)
</answer>

Rules & Constraints
1.	Use all given numbers exactly once.
2.	Only addition (+), subtraction (-), multiplication (×), and division (÷) are allowed.
3.	Parentheses can be used to ensure proper order of operations.
4.	The result must be exactly the target number.
5.	Think inside <think> </think> and answer inside <answer> </answer>.
"""

# def build_prompt(nums: list[int], target: int):
#     return f"{task_description}\n\nInput:\nNumbers: {nums}\nTarget: {target}\n\nOutput:\n"

def build_messages(nums: list[int], target: int):
    return [
        {"role": "user", "content": f"{task_description}\n\nInput:\nNumbers: {nums}\nTarget: {target}\n\nOutput:\n"}
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    dataset = datasets.load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    def _add_prompt(example):
        example["messages"] = build_messages(example["nums"], example["target"])
        return example

    dataset = dataset.map(_add_prompt)

    print("messages 0:")
    print(dataset[0]["messages"][0]["content"])
    print("end of prompt 0")
    dataset.save_to_disk(args.output_dir)
