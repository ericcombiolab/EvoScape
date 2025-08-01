import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import roc_auc_score, average_precision_score
from model import FitnessPredictor
from data import FitnessDataset

def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def load_data(rank,world_size,data_path,batch_size):
    df = pd.read_csv('{}'.format(data_path))
    raw_seq_arr_train = df.loc[:,'raw_seq'].values
    mut_seq_arr_train = df.loc[:,'mut_seq'].values
    judge_same = (raw_seq_arr_train == mut_seq_arr_train)
    if sum(judge_same) > 0:
        print('There are {} samples with raw_seq == mut_seq'.format(sum(judge_same)))
        raise Exception('There are samples with raw_seq == mut_seq')
    score_train = df.loc[:,'label'].values
    train_dataset = FitnessDataset(raw_seq_arr_train,mut_seq_arr_train,score_train)
    sampler = DistributedSampler(train_dataset,num_replicas=world_size,rank=rank,shuffle=True)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    return sampler, loader, len(df)

def train_model(rank,world_size,num_layers,num_heads,block_embed_dim,batch_size,ratio,training_epoch,data_path,esmfold_path,saved_model_prefix):
    setup(rank,world_size)
    
    train_sampler, train_loader, num_samples = load_data(rank,world_size,data_path,batch_size)
    
    device = torch.device('cuda:{}'.format(rank))
    
    model = FitnessPredictor(num_layers,num_heads,block_embed_dim,esmfold_path).to(device)
    model = DDP(model, device_ids=[device],find_unused_parameters=False)
    
    lr_params = 1e-3  # Generic learning rate for other parameters
    # Create parameter groups
    param_groups = [
        {"params": [p for n, p in model.named_parameters()], "lr": lr_params}
    ]

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([ratio,1]).to(device))
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(param_groups, betas= (0.9,0.98), weight_decay=0.01)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas= (0.9,0.98), weight_decay=0.01)
    
    
    start_epoch = 0
    end_epoch = training_epoch
    num_epochs = end_epoch - start_epoch
    accumulation_batch_size = 60 # accumulate gradients with a batch size of accumulation_batch_size
    accumulation_steps = accumulation_batch_size // (batch_size)
    
    batches_per_epoch = int(np.ceil(num_samples/world_size/batch_size))
    steps_per_epoch = batches_per_epoch // accumulation_steps
    # Add one more step per epoch if there are leftover batches
    if batches_per_epoch % accumulation_steps != 0:
        steps_per_epoch += 1
    num_training_steps = steps_per_epoch * num_epochs
    
    num_warmup_steps = min(int(0.05*num_training_steps),500)
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.1, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    model.train()
    optimizer.zero_grad()
    for epoch in range(start_epoch+1, end_epoch+1):
        train_sampler.set_epoch(epoch)
        train_preds = []
        train_labels = []
        total_loss = 0
        for batch_idx, (batch_raw_seqs, batch_mut_seqs, batch_mask_raw, batch_mask_mut, batch_labels, batch_raw_mutations, batch_mut_mutations) in enumerate(train_loader):
            batch_raw_seqs, batch_mut_seqs = batch_raw_seqs.to(device), batch_mut_seqs.to(device)
            batch_mask_raw, batch_mask_mut = batch_mask_raw.to(device), batch_mask_mut.to(device)
            batch_labels,batch_raw_mutations,batch_mut_mutations = batch_labels.to(device),batch_raw_mutations.to(device),batch_mut_mutations.to(device)
            _, output = model(batch_raw_seqs,batch_mut_seqs,batch_mask_raw,batch_mask_mut,batch_raw_mutations,batch_mut_mutations)
            batch_labels = batch_labels.to(torch.long)
            
            loss1 = torch.mean(criterion(output, batch_labels))
            loss = (loss1) / accumulation_steps * 30
            total_loss += loss.item()
            loss.backward()
            
            if (batch_idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            output = output[:,1].cpu().detach().numpy()
            batch_labels = batch_labels.float().cpu().numpy()
            
            train_preds.extend(output)
            train_labels.extend(batch_labels)
        

        train_auroc = roc_auc_score(train_labels, train_preds)
        train_auprc = average_precision_score(train_labels, train_preds)
        mean_loss = total_loss/len(train_loader)
        print(f'Epoch {epoch}, Mean loss: {mean_loss:.4f}, Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}',flush=True)
        
        if (batch_idx+1) % accumulation_steps != 0 and (batch_idx+1) > 0:  # Handle leftover batches
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        model_path = '{}_epoch{}.pt'.format(saved_model_prefix,epoch)
        if dist.get_rank() == 0:
            print('Saving model to {}'.format(model_path))
            torch.save(model.module.state_dict(), model_path)
    
    cleanup()

def main():
    # User-defined inputs
    ##############################################################################################
    # ratio is the ratio of positive samples to negative samples in the training set
    # os.environ['CUDA_VISIBLE_DEVICES'] sets the GPUs to be used for training
    # data_path is the path to the training dataset
    # esmfold_path is the path to the finetuned ESMFold model
    # saved_model_prefix is the prefix for the saved model checkpoints, the saved model will be named as {model_prefix}_epoch{epoch}.pt
    # batch_size is the batch size for training
    # training_epoch is the number of epochs for training
    
    ratio = 0.035 # the ratio of positive samples to negative samples in the training set
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    data_path = '../data/evoscape_train_test/train/train.csv'
    esmfold_path = '../esmfold_finetuned.pt'
    saved_model_prefix = '../evoscape'
    batch_size = 2
    training_epoch = 15
    ##############################################################################################
    
    num_layers = 1 # number of Transformer encoder layers added after ESM model
    num_heads = 8
    block_embed_dim = 128
    world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train_model, args=(world_size,num_layers,num_heads,block_embed_dim,batch_size,ratio,training_epoch,data_path,esmfold_path,saved_model_prefix), nprocs=world_size, join=True)

    

if __name__ == '__main__':
    main()
    
            
            
    


