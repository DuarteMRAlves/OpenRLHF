import argparse
from datasets import load_dataset, Dataset

def prepare_gsm8k_conversational(output_path):
    # Load the GSM8K train dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # Define system message
    system_message = (
        "You are an assistant in a conversation with a user. "
        "The user asks a question, and you solve it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
    )
    
    # Format dataset as a conversation
    formatted_data = []
    for example in dataset:
        question = example["question"]
        
        conversation = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        formatted_data.append({
            "messages": conversation,
            "answer": example["answer"].split("####")[-1].strip()
        })
    
    # Convert to Hugging Face dataset
    hf_dataset = Dataset.from_list(formatted_data)
    print("Record 0")
    print(hf_dataset[0])
    
    # Save as Hugging Face dataset
    hf_dataset.save_to_disk(output_path)
    print(f"Dataset saved at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GSM8K dataset as a conversational dataset")
    parser.add_argument("output_path", type=str, help="Path to save the dataset")
    args = parser.parse_args()
    
    prepare_gsm8k_conversational(args.output_path)
