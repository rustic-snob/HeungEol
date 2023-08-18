import os
import argparse
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
import wandb

from tqdm.auto import tqdm
from llm_model.models import Model
from utils import utils, data_controller
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from shutil import copyfile
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftConfig

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """---Setting---"""
    # argsparse 이용해서 실험명 가져오기
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str)
    args = parser.parse_args()
    # args.exp_name이 None이면 assert False라서 에러 발생 시키기
    assert args.exp_name is not None, "실험명을 입력해주세요."
    # config 파일 불러오기
    with open('./config/use_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
    # 실험 결과 파일 생성 및 폴더명 가져오기
    folder_name, save_path = utils.get_folder_name(CFG, args)
    copyfile('./config/use_config.yaml',f"{save_path}/config.yaml")
    pl.seed_everything(CFG['seed'])
    # wandb 설정
    wandb_logger = wandb.init(
        name=folder_name, project="HeungEol", entity=CFG['wandb']['id'], dir=save_path)
    wandb_logger = WandbLogger()
    wandb_logger.experiment.config.update(CFG)

    """---Train---"""
    # 데이터 로더와 모델 가져오기
    if 'HeungEol' not in CFG['train']['model_name']:
        tokenizer = AutoTokenizer.from_pretrained(CFG['train']['model_name'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(PeftConfig.from_pretrained(CFG['train']['model_name']).base_model_name_or_path)    
    
    tokenizer.pad_token = tokenizer.eos_token
    
    dataloader = data_controller.Dataloader(tokenizer, CFG)
    
    #Already Wrapped by PEFT
    if 'HeungEol' in CFG['train']['model_name']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        
        config = PeftConfig.from_pretrained(CFG['train']['model_name'])
        
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map={"":0},
            low_cpu_mem_usage=True)
        
        base_model.gradient_checkpointing_enable()
        base_model = prepare_model_for_kbit_training(base_model)
        
        LM = PeftModel.from_pretrained(base_model, CFG['train']['model_name'])
        
        for name, param in LM.named_parameters():
            if "lora" in name or "Lora" in name:
             param.requires_grad = True
    
    #QLoRA
    elif CFG['adapt']['peft'] == 'QLoRA':
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        
        LM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=CFG['train']['model_name'],
            quantization_config=bnb_config,
            device_map={"":0},
            low_cpu_mem_usage=True)
        
        LM.gradient_checkpointing_enable()
        LM = prepare_model_for_kbit_training(LM)
        
        peft_config = LoraConfig(r=CFG['adapt']['r'], 
            lora_alpha=CFG['adapt']['alpha'], 
            target_modules=["query_key_value"], 
            lora_dropout=CFG['adapt']['dropout'], 
            bias="none", 
            task_type="CAUSAL_LM")
        
        LM = get_peft_model(LM, peft_config)
    
    #LoRAFull
    elif CFG['adapt']['peft'] == 'LoRAFull':
        
        LM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=CFG['train']['model_name'],
            low_cpu_mem_usage=True).to(device=f"cuda", non_blocking=True)
        
    #LoRAfp16
    elif CFG['adapt']['peft'] == 'LoRAfp16':
        
        LM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=CFG['train']['model_name'],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(device=f"cuda", non_blocking=True)
        
    #Full-Finetuning
    else:
        
        LM = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=CFG['train']['model_name'],
        low_cpu_mem_usage=True).to(device=f"cuda", non_blocking=True)
    
    utils.print_trainable_parameters(LM)
    
    LM.resize_token_embeddings(len(tokenizer))
    model = Model(LM, tokenizer, CFG)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    
    # check point
    checkpoint = utils.LoRACheckpoint(monitor='val_loss',
        save_top_k=3,
        dirpath = f"{save_path}/checkpoints",
        mode = 'min')
    
    callbacks.append(checkpoint)
    
    # Trainer
    trainer = pl.Trainer(accelerator='gpu',
                         precision="16-mixed" if CFG['train']['halfprecision'] else 32,
                         accumulate_grad_batches=CFG['train']['gradient_accumulation'],
                         max_epochs=CFG['train']['epoch'],
                         default_root_dir=save_path,
                         log_every_n_steps=1,
                         val_check_interval=0.25,           # 1 epoch 당 valid loss 4번 체크: 학습여부 빠르게 체크
                         logger=wandb_logger,
                         callbacks=callbacks,
                         enable_checkpointing=False
                         )
    """---fit---"""
    model.LM.config.use_cache = False
    trainer.fit(model=model, datamodule=dataloader)