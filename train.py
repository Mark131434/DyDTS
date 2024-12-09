import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import argparse
from dataclasses import dataclass
import pickle
import glob
from torch.cuda.amp import autocast, GradScaler

@dataclass
class DialogueExample:
    dialogue_id: str
    sentences: List[str]
    embeddings: np.ndarray
    core_indices: List[int] 

@dataclass
class TrainingSample:
    anchor_idx: int
    positive_indices: List[int]
    hard_negative_indices: List[int]
    regular_negative_indices: List[int]

def load_pkl_data(data_dir: str) -> List[Dict]:
    processed_data = []
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No pkl files found in {data_dir}")
    
    for pkl_file in tqdm(pkl_files, desc="Loading pkl files"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    processed_data.extend(data)
                else:
                    processed_data.append(data)
        except Exception as e:
            logging.error(f"Error loading {pkl_file}: {str(e)}")
            continue
            
    logging.info(f"Loaded {len(processed_data)} samples from {len(pkl_files)} files")
    return processed_data

class DialogueDataset(Dataset):
    def __init__(self, processed_data: List[Dict], tokenizer: BertTokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for data in processed_data:
            example = data['example']
            training_samples = data['training_samples']
            
            for sample in training_samples:
                anchor_sent = sample.anchor_idx

                pos_sent = sample.positive_indices

                for neg_sent in sample.hard_negative_indices:
                    self.samples.append({
                        'anchor': anchor_sent,
                        'positive': pos_sent,
                        'negative': neg_sent,
                        'neg_type': 'hard'
                    })
                
                for neg_sent in sample.regular_negative_indices:
                    self.samples.append({
                        'anchor': anchor_sent,
                        'positive': pos_sent,
                        'negative': neg_sent,
                        'neg_type': 'regular'
                    })
   
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        anchor_encoding = self.tokenizer(
            sample['anchor'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            sample['positive'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        negative_encoding = self.tokenizer(
            sample['negative'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
            'neg_type': 1.0 if sample['neg_type'] == 'hard' else 0.5  # hard negative have high value
        }

    @staticmethod
    def collect_fn(batch):
        return {
            'anchor_input_ids': torch.stack([item['anchor_input_ids'] for item in batch]),
            'anchor_attention_mask': torch.stack([item['anchor_attention_mask'] for item in batch]),
            'positive_input_ids': torch.stack([item['positive_input_ids'] for item in batch]),
            'positive_attention_mask': torch.stack([item['positive_attention_mask'] for item in batch]),
            'negative_input_ids': torch.stack([item['negative_input_ids'] for item in batch]),
            'negative_attention_mask': torch.stack([item['negative_attention_mask'] for item in batch]),
            'neg_type': torch.tensor([item['neg_type'] for item in batch])
        }

class DialogueBertModel(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', margin: float = 0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.margin = margin
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
    def get_similarity(self, sent1_emb: torch.Tensor, sent2_emb: torch.Tensor, sigma=1.0) -> torch.Tensor:
        return torch.cosine_similarity(sent1_emb,sent2_emb)

class Trainer:
    def __init__(self, 
                 model: DialogueBertModel,
                 tokenizer: BertTokenizer,
                 args):
        """
        init trainer
        
        Args:
            model: BERT MODEL
            tokenizer: BERT TOKENIZER
            args: PARAMETERS
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.scaler = GradScaler(enabled=(not args.no_amp))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_training(self, train_dataset: Dataset):

        train_sampler = RandomSampler(train_dataset)
        
        self.train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            collate_fn=DialogueDataset.collect_fn
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        
        total_steps = len(self.train_dataloader) * self.args.num_epochs
        num_warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        
        return total_steps

    def save_checkpoint(self, path: str, epoch: int, global_step: int, loss: float):
        """save checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def train(self, train_dataset: Dataset):
        """train model"""
        total_steps = self.setup_training(train_dataset)
        global_step = 0
        best_loss = float('inf')
        

        if self.args.resume and os.path.exists(self.args.resume):
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint['global_step']
            print(f"Resumed from checkpoint: {self.args.resume}")
            self.logger.info(f"Resumed from checkpoint: {self.args.resume}")

        for epoch in range(self.args.num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, 
                                               desc=f"Training Epoch {epoch + 1}")):
                with autocast(enabled=(not self.args.no_amp)):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
    
                    anchor_emb = self.model(batch['anchor_input_ids'],
                                            batch['anchor_attention_mask'])
                    positive_emb = self.model(batch['positive_input_ids'],
                                            batch['positive_attention_mask'])
                    negative_emb = self.model(batch['negative_input_ids'],
                                            batch['negative_attention_mask'])
                    
            
                    pos_sim = self.model.get_similarity(anchor_emb, positive_emb)
                    neg_sim = self.model.get_similarity(anchor_emb, negative_emb)

                    diff = self.model.margin - pos_sim + neg_sim
                    clamped_diff = torch.clamp(diff, min=0.0)
                    loss = torch.mean(batch['neg_type'] * clamped_diff)

         
                self.scaler.scale(loss).backward()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                epoch_loss += loss.item()
                global_step += 1
                
              
                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self.save_checkpoint(
                        os.path.join(self.args.output_dir, f'checkpoint-{global_step}'),
                        epoch,
                        global_step,
                        loss.item()
                    )
            
       
            avg_loss = epoch_loss / len(self.train_dataloader)
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}, "
                             f"Average Loss: {avg_loss:.4f}")
            
        
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(
                    os.path.join(self.args.output_dir, 'best_model'),
                    epoch,
                    global_step,
                    avg_loss
                )

def parse_args():
    parser = argparse.ArgumentParser()
    
 
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing the pkl data files")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length for BERT input")
    
   
    parser.add_argument("--model_name", type=str, default="sup-simcse-bert-base-uncased",
                      help="Pretrained BERT model name")
    parser.add_argument("--margin", type=float, default=0.5,
                      help="Margin for triplet loss")
    
   
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Initial learning rate")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                      help="Epsilon for Adam optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Ratio of warmup steps")
    parser.add_argument("--num_epochs", type=int, default=1,
                      help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping")
 
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model checkpoints")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save checkpoint every X updates steps (0 to disable)")
    
 
    parser.add_argument("--resume", type=str, default=None,
                      help="Path to checkpoint for resuming training")
    
   
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for initialization")
    parser.add_argument("--no_amp", action="store_true",
                      help="Disable automatic mixed precision training")
    
    args = parser.parse_args()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def main():

    args = parse_args()
    
  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
 
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = DialogueBertModel(model_name=args.model_name, margin=args.margin)
    
 
    processed_data = load_pkl_data(args.data_dir)
    train_dataset = DialogueDataset(
        processed_data=processed_data,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    

    trainer = Trainer(model, tokenizer,args)
    
  
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()