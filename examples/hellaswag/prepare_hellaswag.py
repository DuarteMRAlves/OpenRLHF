import argparse
from datasets import load_dataset, Dataset

def make_user_prompt(example):
    ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
    lettered_choices = [f"{l}. {c}" for l, c in zip(["A", "B", "C", "D"], example["endings"])]
    choices = "\n".join(lettered_choices)
    user_prompt = f"\"{ctx}\"\n\nWhat should be the continuation of the above sentence?\n\n{choices}\n\nAfter thinking about the reasoning process, respond with the letter (A, B, C, D) corresponding to the correct choice."
    return user_prompt

def prepare_hellaswag_conversational(output_path):
    # Load the GSM8K train dataset
    dataset = load_dataset("Rowan/hellaswag", split="train")

    # Define system message
    system_message = (
        "You are an assistant in a conversation with a user. "
        "The user asks a question, and you solve it. "
        "The should first think about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer should be enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )

    # Format dataset as a conversation
    formatted_data = []
    for example in dataset:
        user_prompt = make_user_prompt(example)
        answer = ["A", "B", "C", "D"][int(example["label"])]

        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        formatted_data.append({
            "messages": conversation,
            "answer": answer
        })

    # Convert to Hugging Face dataset
    hf_dataset = Dataset.from_list(formatted_data)
    print("Record 0")
    print(hf_dataset[0])
    print(hf_dataset[0]["messages"][1]["content"])

    # Save as Hugging Face dataset
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Hellaswag dataset as a conversational dataset")
    parser.add_argument("output_path", type=str, help="Path to save the dataset")
    args = parser.parse_args()

    prepare_hellaswag_conversational(args.output_path)
