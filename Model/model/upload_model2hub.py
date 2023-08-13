import torch
import os
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_model_path", type=str)
    parser.add_argument("--upload_name", type=str)
    parser.add_argument("--HF_key", type=str)

    return parser.parse_args()

def main():
    args = get_args()
    
    peft_model_id = args.peft_model_path
    config = PeftConfig.from_pretrained(peft_model_id)

    print(f"Loading base model: {config.base_model_name_or_path}")    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)

    print(f"Loading PEFT: {peft_model_id}")
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = model.to('cuda')
    
    print(f"Saving LoRA to hub ...")
    model.push_to_hub(f"{args.upload_name}_LoRA",
                  use_auth_token=args.HF_key)
    
    print(f"Running merge_and_unload ...")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    print(f"Saving Full-Model to hub ...")
    model.push_to_hub(f"{args.upload_name}", use_auth_token=args.HF_key, use_temp_dir=True)
    tokenizer.push_to_hub(f"{args.upload_name}", use_auth_token=args.HF_key, use_temp_dir=True)


if __name__ == "__main__" :
    main()