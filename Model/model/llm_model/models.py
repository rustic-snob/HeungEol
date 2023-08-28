import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle

from . import lr_schedule_controller

class Model(pl.LightningModule):
    def __init__(self, LM, tokenizer, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        # 사용할 모델을 호출
        self.LM = LM                            # Language Model
        self.tokenizer = tokenizer              # Tokenizer
        
        self.optim = torch.optim.AdamW
        

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def training_step(self, batch, batch_idx):
        x, structures = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            labels=x['labels']
        )
        base_loss = outputs['loss']
        logits = outputs['logits']
        labels = x['labels']
        starting_output = x['starting_output']
        
        if self.CFG['train']['align_loss']:
            
            structures = [list(map(int, structure.split(' / '))) for structure in structures]
            align_loss = torch.tensor(0.0, device=self.device)
            aligned = 0
            
            for idx, gt_structure in enumerate(structures):
                 
                all_new_tokens = logits[idx][labels[idx].tolist().count(-100)-1:-1].argmax(dim=-1)
                cur_output_tokens = all_new_tokens[all_new_tokens != self.tokenizer.pad_token_id]

                pred_structure = self._compute_output_structure(self.tokenizer.decode(cur_output_tokens))
                align_info = self._compute_align_loss(gt_structure, pred_structure)
                
                align_loss += align_info[1]  / self.CFG['train']['batch_size']
                aligned += align_info[0]
                
            loss = base_loss + align_loss
                  
            self.log("align_loss", align_loss)
            
        elif self.CFG['train']['better_align_loss']:
            better_align_loss = self._compute_better_align_loss(logits[:,:-1,:], labels[:,1])
            
            loss = base_loss + better_align_loss
            
            self.log("better_align_loss", better_align_loss)
            
        else:
            loss = base_loss
        
        if self.CFG['train']['repeat_penalty']:
            ul_loss = self._unlikelihood_loss(logits[:,:-1,:], labels[:,1], starting_output)
            
            loss += ul_loss
            
            self.log("ul_loss", ul_loss)

        self.log("base_loss", base_loss)
        self.log("train_loss", loss)

        return loss
    
    # 이하 데이터셋 구축 이후 상세 구현 예정

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            labels=x['labels']
        )
        loss = outputs['loss']
        
        self.log("val_loss", loss)  
        # breakpoint()
        
        # generated_tokens = ''.join([self.tokenizer.convert_ids_to_tokens(sample) for sample in torch.argmax(outputs['logits'], dim=-1)])
        # reference_tokens = ''.join(y)
        
        # self.log("generated_tokens", generated_tokens)
        # self.log("reference_tokens", reference_tokens)
        
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        gened_list = []
        with torch.no_grad():
            for i in x:      
                gened = self.LM.generate(**self.tokenizer(i, return_tensors='pt', return_token_type_ids=False), 
                                        max_new_tokens=256,
                                        early_stopping=True,
                                        do_sample=True,
                                        eos_token_id=2)
                gened_list.append(self.tokenizer.decode(gened[0]))
            
        return gened_list
        
    def _compute_individual_loss(self, logits, labels):
        labels = labels.to(logits.device)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        individual_token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        individual_losses = torch.tensor([x.sum() /(label!=-100).sum() for x, label in zip(individual_token_losses.view_as(shift_labels), shift_labels)])
        
        return individual_losses

    def _compute_output_structure(self, output):
        chunks = output.split('/') 
        return [len(chunk.replace(' ', '')) for chunk in chunks]
    
    def _compute_align_loss(self, desired_structure, output_structure):
        
        flag = True
        
        scale = self.CFG['train']['align_loss_scale']
        
        desired = torch.tensor(desired_structure, dtype=torch.float32, device=self.device)
        output = torch.tensor(output_structure, dtype=torch.float32, device=self.device)
        
        # If under_generated, it is equivalent to generate 0 syllable for gt k syllables 
        if len(desired) < len(output):
            # stack 0 to the desired tensor to have the same length with output
            padding = torch.zeros(len(output) - len(desired), dtype=torch.float32, device=self.device)
            desired = torch.cat((desired, padding))
            
            flag = False
        
        # If over_generated, it is equivalent to generate k syllables for gt 0 syllables    
        elif len(output) < len(desired):
            # stack 0 to the output tensor to have the same length with desired
            padding = torch.zeros(len(desired) - len(output), dtype=torch.float32, device=self.device)
            output = torch.cat((output, padding))
            
            flag = False
            
        # If lengths match, compute MSE loss
        loss = torch.nn.functional.mse_loss(desired, output) * scale
        return (flag, loss)

    def _compute_better_align_loss(self, logits, labels):
        """
        Compute a penalty based on misaligned ' / ' and syllables.

        Args:
        - logits (torch.Tensor): The logits from the model.                         (bsz, seq_len-1, vocab_size)
        - labels (torch.Tensor): The ground truth labels.                           (bsz, seq_len-1)
        - scale (float): The penalty's scale to apply for each misalignment.

        Returns:
        - torch.Tensor: The computed penalty.
        """
        
        # Get slash token id set
        slash_tokens = ['/', ' /', ' / ', '/ ']
        
        slash_token_ids = set([self.tokenizer.encode(token, add_special_tokens=False)[0] for token in slash_tokens if len(self.tokenizer.encode(token, add_special_tokens=False)) == 1])
        
        # Get the predicted tokens from logits
        pred_tokens = logits.argmax(dim=-1)                                              # (bsz,seq_len-1)

        pred_slash_mask = gt_slash_mask = torch.zeros_like(labels, dtype=torch.float32)  # (bsz,seq_len-1)
        
        # Create masks
        for slash_token_id in slash_token_ids:
            pred_slash_mask |= (pred_tokens == slash_token_id)
            gt_slash_mask |= (labels == slash_token_id)

        # Find misalignments
        misalignments = (pred_slash_mask ^ gt_slash_mask) & (labels != -100)

        # Compute the penalty
        misalignment_penalty = misalignments.float().sum() * self.CFG['train']['align_loss_scale']

        return misalignment_penalty

    def _unlikelihood_loss(self, logits, labels, starting_output):
        """
        Compute the unlikelihood loss for repetition penalty.

        Args:
        - logits (torch.Tensor): The logits from the model.                         (bsz, seq_len-1, vocab_size)
        - labels (torch.Tensor): The ground truth labels.                           (bsz, seq_len-1)
        - starting_output (list): position where the output(after prompt) starts.   (bsz,)

        Returns:
        - torch.Tensor: The computed unlikelihood loss.
        """
        
        # Get the predicted probabilities from logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Compute the binary mask M indicating the start of each phrase in the sequence
        slash_tokens = ['/', ' /', ' / ', '/ ']
        slash_token_ids = set([self.tokenizer.encode(token, add_special_tokens=False)[0] for token in slash_tokens if len(self.tokenizer.encode(token, add_special_tokens=False)) == 1])
        
        
        M = torch.zeros_like(labels, dtype=torch.float32)               # (bsz, seq_len-1)
        for slash_token_id in slash_token_ids:
            M |= (labels == slash_token_id)
            
        M = torch.roll(M, shifts=1, dims=1)
        M[:, 0] = False
        
        ul_loss = 0.0
        for x in range(M.shape[0]):
            for y in range(M.shape[1]):
                if M[x, y] == 1 and labels[x, y] != -100:
                    Ci = self._compute_Ci(labels[x].tolist(), M[x].tolist(), y, starting_output[x])
                    for c in Ci:
                        ul_loss -= torch.log(1 - probs[x, y, c])

        return ul_loss * self.CFG['train']['align_loss_scale']

    def _compute_Ci(self, D, M, i, sp):
        """
        Compute the set Ci for the i-th position.

        Args:
        - D (list): The desired ground truth sequence of tokens.
        - M (list): A binary mask indicating the start of each phrase in the sequence.
        - i (int): The position for which we want to compute Ci.
        - sp (int): starting point of current D

        Returns:
        - set: The set Ci.
        """
        
        # Identify tokens up to the i-th position that are the start of a new phrase/syllable
        start_tokens = [D[j] for j in range(sp-1, i) if M[j] == 1]
        
        # Convert the list of start tokens to a set to ensure uniqueness
        Ci = set(start_tokens)
        
        # Remove the i-th token from the set
        Ci.discard(D[i])
        
        return Ci
    
    def configure_optimizers(self): 
        optimizer = self.optim(self.parameters(), lr=self.CFG['train']['LR']['lr'])

        if self.CFG['train']['LR']['name'] == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer,
                lr_lambda=lambda epoch: 0.95 ** epoch,
                last_epoch=-1,
                verbose=False)
        elif self.CFG['train']['LR']['name'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=5,
                gamma=0.3,
                verbose=True)
        elif self.CFG['train']['LR']['name'] == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.CFG['train']['LR']['lr'] / self.CFG['train']['LR']['base'], 
                                                          max_lr=self.CFG['train']['LR']['lr'] / self.CFG['train']['LR']['max'], 
                                                          step_size_up=self.CFG['train']['LR']['step_up'], 
                                                          step_size_down=self.CFG['train']['LR']['step_down'], cycle_momentum=False, mode='exp_range')
        elif self.CFG['train']['LR']['name'] == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5,verbose=True)
        elif self.CFG['train']['LR']['name'] == 'WarmupConstantLR':
            scheduler = lr_schedule_controller.WarmupConstantLR(optimizer, warmup_steps=self.CFG['train']['LR']['warmupconstantLR_step'])
        elif self.CFG['train']['LR']['name'] == 'WarmupDecayLR':
            scheduler = lr_schedule_controller.WarmupDecayLR(optimizer, warmup_steps=self.CFG['train']['LR']['warmupdecayLR_warmup'], total_steps=self.CFG['train']['LR']['warmupdecayLR_total'])

        
        lr_scheduler = {
            'scheduler': scheduler,
            'interval' : self.CFG['train']['LR']['interval'],
            'name': self.CFG['train']['LR']['name']
        }

        return [optimizer], [lr_scheduler]