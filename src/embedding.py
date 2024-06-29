import h5py
import torch
from esm import Alphabet, FastaBatchedDataset, pretrained
from transformers import T5EncoderModel, T5Tokenizer
import tqdm
from src.data import *
from src.utils import *
import os
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "cpu"
    dtype=torch.bfloat16
else:
    device = "cpu"
    dtype=torch.bfloat16

def embed_esm1b(embed_dataloader, out_file):
    model, _ = pretrained.load_model_and_alphabet("esm1b_t33_650M_UR50S")
    model.eval().to(device)
    embed_h5 = h5py.File(out_file, "w")
    try:
        with torch.autocast(device_type=device,dtype=dtype):
            with torch.no_grad():
                for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
                    embed = model(toks.to(device), repr_layers=[33])["representations"][33].float().cpu().numpy()
                    for j in range(len(labels)):
                        # removing start and end tokens
                        embed_h5[labels[j]] = embed[j, 1:1+lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")
    

def embed_prott5(embed_dataloader, out_file):
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model.eval().to(device)
    embed_h5 = h5py.File(out_file, "w")
    try:
        with torch.autocast(device_type=device,dtype=dtype):
            with torch.no_grad():
                for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
                    embed = model(input_ids=torch.tensor(toks['input_ids'], device=device),
                    attention_mask=torch.tensor(toks['attention_mask'], 
                        device=device)).last_hidden_state.float().cpu().numpy()
                    for j in range(len(labels)):
                        # removing end tokens
                        embed_h5[labels[j]] = embed[j, :lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")

def embed_onehot(embed_dataloader, alphabet, out_file):
    # Get the number of different tokens in the alphabet
    num_classes = len(alphabet)
    embed_h5 = h5py.File(out_file, "w")
    try:
        for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
            embed = torch.nn.functional.one_hot(toks.to(device), num_classes=num_classes).float().cpu().numpy()
            for j in range(len(labels)):
                embed_h5[labels[j]] = embed[j, 1:1+lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")

def embed_seq_blosum(token_list, alphabet, blosum):
    reverse_alphabet = {v: k for k, v in alphabet.items()}
    seq_embedding = []

    for token in token_list:
        # Get the corresponding key from the dictionary
        key = reverse_alphabet.get(token, None)
        if key:
            # Check if the key exists in the 'AA' column of the DataFrame
            if key in blosum['AA'].values:
                # Get the rows where the key exists in the 'AA' column
                row = blosum[blosum['AA'] == key]
                # Iterate through each row and print the values of the other 20 columns
                letter_embedding = row[blosum.columns[1:]].values
            else:
                # If the key does not exist, return an array of 20 zeros
                letter_embedding = [0] * 20
        else:
            # If the key is None, return an array of 20 zeros
            letter_embedding = [0] * 20
        # Append the values to the result arrays list
        seq_embedding.append(letter_embedding)

    return seq_embedding

def embed_blosum(embed_dataloader, alphabet, out_file):
    embed_h5 = h5py.File(out_file, "w")
    try:
        for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
            embed = []
            raw_blosum = pd.read_csv('data_files/BLOSUM.csv')
            for tok in toks:
                embedded_seq = embed_seq_blosum(tok, alphabet, raw_blosum)
                embed.append(embedded_seq)
            embed = torch.Tensor(embed).float().cpu().numpy()
            for j in range(len(labels)):
                embed_h5[labels[j]] = embed[j, 1:1+lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")

def embed_esm2(embed_dataloader, out_file):
    model, _ = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval().to(device)
    embed_h5 = h5py.File(out_file, "w")
    try:
        with torch.autocast(device_type=device, dtype=dtype):
            with torch.no_grad():
                for i, (toks, lengths, np_mask, labels) in tqdm.tqdm(enumerate(embed_dataloader)):
                    embed = model(toks.to(device), repr_layers=[33])["representations"][33].float().cpu().numpy()
                    for j in range(len(labels)):
                        # removing start and end tokens
                        embed_h5[labels[j]] = embed[j, 1:1+lengths[j]].astype(np.float16)
        embed_h5.close()
    except:
        os.system(f"rm {out_file}")
        raise Exception("Failed to create embeddings")

def generate_embeddings(model_attrs: ModelAttributes):
    fasta_dict = read_fasta(EMBEDDINGS[model_attrs.model_type]["source_fasta"])
    test_df = pd.DataFrame(fasta_dict.items(), columns=['ACC', 'Sequence'])
    embed_dataset = FastaBatchedDatasetTorch(test_df)
    embed_batches = embed_dataset.get_batch_indices(8196, extra_toks_per_seq=1)

    if model_attrs.model_type == FAST:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(model_attrs.alphabet), batch_sampler=embed_batches)
        embed_esm1b(embed_dataloader, EMBEDDINGS[model_attrs.model_type]["embeds"])
    elif model_attrs.model_type == ACCURATE:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverterProtT5(model_attrs.alphabet), batch_sampler=embed_batches)
        embed_prott5(embed_dataloader, EMBEDDINGS[model_attrs.model_type]["embeds"])
    elif model_attrs.model_type == ONEHOT:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(model_attrs.alphabet), batch_sampler=embed_batches)
        embed_onehot(embed_dataloader, model_attrs.alphabet.to_dict(), EMBEDDINGS[model_attrs.model_type]["embeds"])
    elif model_attrs.model_type == BLOSUM:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(model_attrs.alphabet), batch_sampler=embed_batches)
        embed_blosum(embed_dataloader, model_attrs.alphabet.to_dict(), EMBEDDINGS[model_attrs.model_type]["embeds"])
    elif model_attrs.model_type == FAST2:
        embed_dataloader = torch.utils.data.DataLoader(embed_dataset, collate_fn=BatchConverter(model_attrs.alphabet), batch_sampler=embed_batches)
        embed_esm2(embed_dataloader, EMBEDDINGS[model_attrs.model_type]["embeds"])
    else:
        raise Exception("wrong model type provided expected Fast,Accurate got", model_attrs.model_type)
    
