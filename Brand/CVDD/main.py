import torch
import logging
import random
import numpy as np
import pandas as pd

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from training_pipe import CVDD
from data_preprocess import BrandDataset
from utils import print_text_samples


# Prepare logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = 'Brand\\CVDD\\log' + '/log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Parameters
pretrained_model_name = 'bert-base-multilingual-uncased'
load_model = None
device = 'cuda'
attention_size = 150
attention_heads = 3
lr = 0.001
n_epochs = 15
lr_milestones = [5, 10]
lambda_p = 1.0
alpha_scheduler = 'logarithmic'
weight_decay = 0.5e-6

# Prepare data
train_data = pd.read_csv('G:/Code/Python/GitHub/maboo/Brand/brand_data/brand_final.csv').loc[:, ['name', 'brand']]
test_data = pd.read_csv('G:/Code/Python/GitHub/maboo/Brand/資料/test_brand.csv').loc[:, ['name', 'brand']]
valid_brand = [i for i in set(train_data['brand'].to_list())]

for i in range(len(train_data)):
    if train_data['brand'][i] in valid_brand:
        train_data['brand'][i] = 1
    else:
        train_data['brand'][i] = 0

X_train = data[data['brand'] == 1]['name']
y_train = data[data['brand'] == 1]['brand']

X_valid = data[data['brand'] == 0]['name']
y_valid = data[data['brand'] == 0]['brand']

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
train_dataset = BrandDataset(X_train, y_train, tokenizer, 512)
valid_dataset = BrandDataset(X_valid, y_valid, tokenizer, 512)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

cvdd = CVDD(ad_score='context_dist_mean')
cvdd.set_network(pretrained_model_name=pretrained_model_name,
                 attention_size=attention_size,
                 n_attention_heads=attention_heads)

# Load model
load_model = None
if load_model:
    cvdd.load_model(load_model, device=device)

# Train model
cvdd.train(train_loader, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones, lambda_p=lambda_p,
           alpha_scheduler=alpha_scheduler, weight_decay=weight_decay, device=device)

# Test model
cvdd.test(valid_loader, device=device)

# Print most anomalous and most normal test samples
indices, labels, scores, heads = zip(*cvdd.results['test_scores'])
indices, scores = np.array(indices), np.array(scores)
sort_idx = np.argsort(scores).tolist()  # sorted from lowest to highest anomaly score
idx_sorted = indices[sort_idx]
idx_normal = idx_sorted[:50].tolist()
idx_outlier = idx_sorted[-50:].tolist()[::-1]
att_weights = cvdd.test_att_weights
att_weights_sorted = [att_weights[i] for i in sort_idx]
att_weights_normal = att_weights_sorted[:50]
att_weights_outlier = att_weights_sorted[-50:][::-1]
heads_sorted = [heads[i] for i in sort_idx]
heads_normal = heads_sorted[:50]
heads_outlier = heads_sorted[-50:][::-1]

print_text_samples(valid_dataset, BertTokenizer, idx_normal, export_file='Brand\\CVDD\\log' + '/normals',
                   att_heads=heads_normal, weights=att_weights_normal, title='Most normal examples')
print_text_samples(valid_dataset, BertTokenizer, idx_outlier, export_file='Brand\\CVDD\\log' + '/outliers',
                   att_heads=heads_outlier, weights=att_weights_outlier, title='Most anomalous examples')

cvdd.save_results('Brand\\CVDD\\log'+'/results.json')
cvdd.save_model('Brand\\CVDD\\log'+'/model.ckpt')