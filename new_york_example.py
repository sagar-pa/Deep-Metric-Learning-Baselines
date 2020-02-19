import os, sys, numpy as np, argparse, imp, datetime, time, pickle as pkl, random, json
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import matplotlib
matplotlib.use('agg')

from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms
import netlib as netlib
from tqdm import tqdm
import datasets as data
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= str(0)


"""============================================================================"""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True
np.random.seed(1); random.seed(1)
torch.manual_seed(1); torch.cuda.manual_seed(1); torch.cuda.manual_seed_all(1)


"""============================================================================"""
##################### NETWORK SETUP ##################
device = torch.device('cuda')
#Depending on the choice opt.arch, networkselect() returns the respective network model
parser = argparse.ArgumentParser()
parser.add_argument('--embed_dim',    default=128,         type=int,   help='Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')
parser.add_argument('--arch',         default='resnet50',  type=str,   help='Network backend choice: resnet50, googlenet.')
parser.add_argument('--not_pretrained',                    action='store_true', help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
opt = parser.parse_args()
model      = netlib.networkselect(opt)
model.load_state_dict(torch.load('dmt_example.pt'))
_          = model.to(device)


def ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img


data_source = os.getcwd()+ '/Datasets/new_york/'
data_file = data_source + 'new_york_url_tst_2.txt'
with open (data_file, 'r') as f:
    lines = [line.strip() for line in f]
dataloader = {}
for line in lines:
    id, url1, url2 = line.split()
    dataloader[id] = data_source+'images/'+id+'.jpg'
eval_dataset        = data.BaseTripletDataset(dataloader, opt, is_validation=True, samples_per_class=1)

eval_set = torch.utils.data.DataLoader(eval_dataset, batch_size=112, num_workers=8, shuffle=False, pin_memory=True, drop_last=False)
eval_iter = tqdm(eval_set)
feature_coll = []

model.eval()
torch.cuda.empty_cache()
with torch.no_grad():
    for idx, input in enumerate(eval_iter):
        out = model(input.to(device))
        feature_coll.extend(out.cpu().detach().numpy().tolist())
for idx, id in enumerate(dataloader.keys()):
    repres = data_source + 'dmt_features/' + id + '.npy'
    np.save(repres, feature_coll[idx])