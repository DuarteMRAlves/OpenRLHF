import argparse
import random
import datasets
import tqdm

def main():
    args = parse_args()
    random.seed(args.seed)

    examples = [generate_example() for _ in tqdm.trange(args.num_examples)]

    for i in range(20):
        print(examples[i])

    dataset = datasets.Dataset.from_list(examples)
    
    def _add_messages(example):
        example["messages"] = build_messages(example["nums"], example["target"])
        return example

    dataset = dataset.map(_add_messages)

    print("messages 0:")
    print(dataset[0]["messages"][0]["content"])
    print("end of prompt 0")

    dataset.save_to_disk(args.output_dir)


task_description = """You will be given a list of numbers and a target number. Your task is to generate a valid mathematical expression using all the numbers exactly once, and only the basic operations (+, -, ×, ÷) to reach the target number. Parentheses can be used to enforce operation order.

Before producing the final answer, think through the steps inside <think> </think> and then provide the solution inside <answer> </answer>.

Rules & Constraints
1.	Use all given numbers exactly once.
2.	Only addition (+), subtraction (-), multiplication (×), and division (÷) are allowed.
3.	Parentheses can be used to ensure proper order of operations.
4.	The result must be exactly the target number.
5.	Think inside <think> </think> and answer inside <answer> </answer>.
"""

def build_messages(nums: list[int], target: int):
    return [
        {"role": "user", "content": f"{task_description}\nNumbers: {nums}\nTarget: {target}\nLet's think step-by-step inside <think> </think> and then provide the solution inside <answer> </answer>."}
    ]


def generate_example():
    example = propose_example()
    while not accept_example(example):
        example = propose_example()
    return example


def accept_example(example):
    target = example["target"]
    return target >= 0 and target <= 100


def propose_example():
    num_numbers = random.randint(3, 5)

    numbers = [random.randint(1, 50) for _ in range(num_numbers)]

    solution_nodes = [{"number": n, "op": "leaf"} for n in numbers]

    while len(solution_nodes) > 1:

        idx1, idx2 = random.sample(range(len(solution_nodes)), 2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        node2 = solution_nodes.pop(idx2)
        node1 = solution_nodes.pop(idx1)

        num1 = node1["number"]
        num2 = node2["number"]

        valid_op = False
        while not valid_op:
            op = random.choice(["+", "-", "×", "÷"])
            valid_op = (
                op != "÷" or (num2 != 0 and num1 % num2 == 0)
            )

        if op == "+":
            new_number = num1 + num2
        elif op == "-":
            new_number = num1 - num2
        elif op == "×":
            new_number = num1 * num2
        else:
            new_number = num1 // num2
        
        new_node = {"number": new_number, "op": op, "left": node1, "right": node2}
        solution_nodes.append(new_node)

    assert len(solution_nodes) == 1
    solution = solution_nodes[0]

    random.shuffle(numbers)
    return {
        "nums": numbers,
        "target": solution["number"],
        "solution": tree_to_formula(solution),
    }

def tree_to_formula(node):
    """
    Recursively convert a solution tree into a mathematical formula string.

    The tree is expected to be a dictionary with keys:
      - "number": the value at this node,
      - "op": the operation that produced the node ("+", "-", "*", "/", or "leaf").
      - "left" and "right": the child nodes (only when op is not "leaf").

    Args:
      node (dict): The root node of the solution tree.

    Returns:
      str: The mathematical formula represented by the tree.
    """
    # Base case: if the node is a leaf, return its number.
    if node.get("op") == "leaf":
        return str(node.get("number"))
    
    # Recursively process the left and right subtrees.
    left_expr = tree_to_formula(node.get("left"))
    right_expr = tree_to_formula(node.get("right"))
    
    # Return the expression for this node, with parentheses to preserve order.
    return f"({left_expr} {node.get('op')} {right_expr})"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    main()