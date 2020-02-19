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
parser.add_argument('--lr',                default=0.00001,  type=float, help='Learning Rate for network parameters.')
parser.add_argument('--fc_lr_mul',         default=0,        type=float, help='OPTIONAL: Multiply the embedding layer learning rate by this value. If set to 0, the embedding layer shares the same learning rate.')
parser.add_argument('--n_epochs',          default=70,       type=int,   help='Number of training epochs.')
parser.add_argument('--kernels',           default=8,        type=int,   help='Number of workers for pytorch dataloader.')
parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--samples_per_class', default=4,        type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
parser.add_argument('--seed',              default=1,        type=int,   help='Random seed for reproducibility.')
parser.add_argument('--scheduler',         default='step',   type=str,   help='Type of learning rate scheduling. Currently: step & exp.')
parser.add_argument('--gamma',             default=0.3,      type=float, help='Learning rate reduction after tau epochs.')
parser.add_argument('--decay',             default=0.0004,   type=float, help='Weight decay for optimizer.')
parser.add_argument('--tau',               default=[30,55],nargs='+',type=int,help='Stepsize(s) before reducing learning rate.')

##### Loss-specific Settings
parser.add_argument('--loss',         default='marginloss', type=str,   help='loss options: marginloss, triplet, npair, proxynca')
parser.add_argument('--sampling',     default='distance',   type=str,   help='For triplet-based losses: Modes of Sampling: random, semihard, distance.')
### MarginLoss
parser.add_argument('--margin',       default=0.2,          type=float, help='TRIPLET/MARGIN: Margin for Triplet-based Losses')
parser.add_argument('--beta_lr',      default=0.0005,       type=float, help='MARGIN: Learning Rate for class margin parameters in MarginLoss')
parser.add_argument('--beta',         default=1.2,          type=float, help='MARGIN: Initial Class Margin Parameter in Margin Loss')
parser.add_argument('--nu',           default=0,            type=float, help='MARGIN: Regularisation value on betas in Margin Loss.')
parser.add_argument('--beta_constant',                      action='store_true', help='MARGIN: Use constant, un-trained beta.')
### ProxyNCA
parser.add_argument('--proxy_lr',     default=0.00001,     type=float, help='PROXYNCA: Learning Rate for Proxies in ProxyNCALoss.')
### NPair L2 Penalty
parser.add_argument('--l2npair',      default=0.02,        type=float, help='NPAIR: Penalty-value for non-normalized N-PAIR embeddings.')

##### Evaluation Settings
parser.add_argument('--k_vals',       nargs='+', default=[1,2,4,8], type=int, help='Recall @ Values.')

##### Network parameters
parser.add_argument('--embed_dim',    default=128,         type=int,   help='Embedding dimensionality of the network. Note: in literature, dim=128 is used for ResNet50 and dim=512 for GoogLeNet.')
parser.add_argument('--arch',         default='resnet50',  type=str,   help='Network backend choice: resnet50, googlenet.')
parser.add_argument('--not_pretrained',                    action='store_true', help='If added, the network will be trained WITHOUT ImageNet-pretrained weights.')
parser.add_argument('--grad_measure',                      action='store_true', help='If added, gradients passed from embedding layer to the last conv-layer are stored in each iteration.')
parser.add_argument('--dist_measure',                      action='store_true', help='If added, the ratio between intra- and interclass distances is stored after each epoch.')

##### Setup Parameters
parser.add_argument('--gpu',          default=0,           type=int,   help='GPU-id for GPU to use.')
parser.add_argument('--savename',     default='',          type=str,   help='Save folder name if any special information is to be included.')

### Paths to datasets and storage folder
parser.add_argument('--source_path',  default=os.getcwd()+'/Datasets',         type=str, help='Path to training data.')
parser.add_argument('--save_path',    default=os.getcwd()+'/Training_Results', type=str, help='Where to save everything.')
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
ids = []
sources = []
dataloader = {}
for line in lines:
    id, url1, url2 = line.split()
    ids.append(id)
    sources.append(data_source+'images/'+id+'.jpg')

dataloader[0] = sources
eval_dataset        = data.BaseTripletDataset(dataloader, opt, is_validation=True, samples_per_class=501)

eval_set = torch.utils.data.DataLoader(eval_dataset, batch_size=112, num_workers=8, shuffle=False, pin_memory=True, drop_last=False)
eval_iter = tqdm(eval_set)
feature_coll = []

model.eval()
torch.cuda.empty_cache()
with torch.no_grad():
    for idx, input in enumerate(eval_iter):
        input_img = input[-1]
        out = model(input_img.to(device))
        feature_coll.extend(out.cpu().detach().numpy().tolist())
for idx, id in enumerate(ids):
    repres = data_source + 'dmt_features/' + id + '.npy'
    np.save(repres, feature_coll[idx])