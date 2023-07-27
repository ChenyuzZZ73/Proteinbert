#import torch
#from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer

texta ='PDGRNAAAKAFDLITPTVRKGCCSNPACILNNPNQCG'
textb = '3D-structure'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokensa = tokenizer.tokenize(texta,padding='max_length',max_length=50,add_special_tokens=True)
tokensb = tokenizer(textb,padding='max_length',max_length=50,add_special_tokens=True)
#print(tokensa)
#print(tokensb)



class Dataset(Dataset):
    """Custom Dataset for loading CelebA face images"""
    def __init__(self):
        df = pd.read_csv("data.csv",sep=",", index_col=0)
        self.x_1 = df["Organism"].values
        self.x_2 = df["Keywords"].values
        self.y = df["Sequence"].values


    def __getitem__(self, index):
        seq = self.y[index]
        label = self.x_1 +';' +self.x_2
        label = label[index]
        return seq, label

    def __len__(self):
        return self.y.shape[0]

trainset = Dataset()
dataloader = DataLoader(dataset=trainset,
                          batch_size=2,
                          shuffle=True,
                          num_workers=4)

if __name__ == "__main__":
    #seq = 'MIVYGLYKSPFGPITVAKNEKGFVMLDFCDCAERSSLDNDYFTDFFYKLDLYFEGKKVDLTEPVDFKPFNEFRIRVFKEVMRIKWGEVRTYKQVADAVKTSPRAVGTALSKNNVLLIIPCHRVIGEKSLGGYSRGVELKRKLLELEGIDVAKFIEK'
    seq = "Pyrococcus horikoshii (strain ATCC 700860 / DSM 12428 / JCM 9974 / NBRC 100139 / OT-3);3D-structure;Acyltransferase;Transferase"
    print(tokenizer.encode(seq,return_tensors='pt'))
    for i, (seqs, labels) in enumerate(dataloader):
        #tokenizer(textb, padding='max_length', max_length=50, add_special_tokens=True)
        print(labels)
        print(seqs)
        print(tokenizer.encode(labels, padding='max_length',add_special_tokens=True,return_tensors='pt'))
        #print(tokenizer(labels, add_special_tokens=True))


