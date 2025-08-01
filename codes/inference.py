import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import FitnessPredictor
from data import FitnessDataset

def run_infer(data_path,esmfold_path,state_dict_path,device_id):
    df = pd.read_csv('{}'.format(data_path))
    raw_seq_arr_test = df.loc[:,'raw_seq'].values
    mut_seq_arr_test = df.loc[:,'mut_seq'].values
    label_test = [-1]*len(raw_seq_arr_test)  # Placeholder labels, as we are not using them in inference
    
    judge_same = (raw_seq_arr_test == mut_seq_arr_test)
    if sum(judge_same) > 0:
        print('There are {} samples with raw_seq == mut_seq'.format(sum(judge_same)))
        raise Exception('There are samples with raw_seq == mut_seq')
    
    dataset = FitnessDataset(raw_seq_arr_test,mut_seq_arr_test,label_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    num_layers = 1 # number of Transformer encoder layers added after ESM model
    num_heads = 8
    block_embed_dim = 128
    model = FitnessPredictor(num_layers,num_heads,block_embed_dim,esmfold_path)
    
    state_dict = torch.load(state_dict_path,map_location=device)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    output_list = []
    with torch.no_grad():
        for i, (batch_raw_seq,batch_mut_seq,batch_mask_raw,batch_mask_mut,batch_label,batch_raw_mutations,batch_mut_mutations) in enumerate(loader):
            batch_raw_seq,batch_mut_seq,batch_mask_raw,batch_mask_mut,batch_label = batch_raw_seq.to(device),batch_mut_seq.to(device),batch_mask_raw.to(device),batch_mask_mut.to(device),batch_label.to(device)
            batch_raw_mutations, batch_mut_mutations = batch_raw_mutations.to(device), batch_mut_mutations.to(device)

            output, _ = model(batch_raw_seq,batch_mut_seq,batch_mask_raw,batch_mask_mut,batch_raw_mutations,batch_mut_mutations)
            
            output = output[:,1].cpu().detach().numpy()
            output_list.extend(output)

    return output_list

if __name__ == '__main__':
    # User-defined inputs
    #######################################################
    # data_path is the path to the test dataset
    # esmfold_path is the path to the finetuned ESMFold model
    # state_dict_path is the path to the saved model during the training stage
    # output_path is the path to the output file
    batch_size = 1
    device_id = 0
    data_path = '../data/sample_demo/test_H1N1_HA.csv'
    esmfold_path = '../esmfold_finetuned.pt'
    state_dict_path = '../evoscape_weights.pt'
    output_path = '../output.csv'
    #######################################################
    
    output_list = run_infer(data_path,esmfold_path,state_dict_path,device_id)
    df_output = pd.DataFrame({'score': output_list})
    df_output.to_csv(output_path, index=False)
    

    
        
    
    
    


