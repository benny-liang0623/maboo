import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from networks.network import CVDDNet
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

class CVDDTrainer(BaseTrainer):
    
    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int=150, lr_milestones: tuple=(),
                 batch_size: int = 128, lambda_p: float = 0.0, alpha_scheduler: str='hard',
                 weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)
        
        self.lambda_p = lambda_p
        self.c = None
        
        self.train_dists = None
        self.train_att_matrix = None
        self.train_top_words = None

        self.test_dists = None
        self.test_att_matrix = None
        self.test_top_words = None
        self.test_auc = 0.0
        self.test_scores = None
        self.test_att_weights = None
        
        # alpha annealing strategy
        self.alpha_milestones = np.arange(1, 6) * int(n_epochs / 5)  # 5 equidistant milestones over n_epochs
        if alpha_scheduler == 'soft':
            self.alphas = [0.0] * 5
        if alpha_scheduler == 'linear':
            self.alphas = np.linspace(.2, 1, 5)
        if alpha_scheduler == 'logarithmic':
            self.alphas = np.logspace(-4, 0, 5)
        if alpha_scheduler == 'hard':
            self.alphas = [100.0] * 4
        
    def train(self, dataset:BaseADDataset, net: CVDDNet):
        
        net = net.to(self.device)
        attention_heads = net.attention_heads
        
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        
        net.c.data = torch.from_numpy(
            initialize_context_vectors(net, train_loader, self.device)[np.newaxis, :]).to(self.device)
        
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        net.train()
        alpha_i = 0
        for epoch in range(self.n_epochs):
            
            scheduler.step()
            if epoch in self.alpha_milestones:
                net.alpha = float(self.alphas[alpha_i])
                alpha_i += 1
            
            epoch_loss = 0.0
            n_batches = 0
            att_matrix = np.zeros((attention_heads, attention_heads))
            dists_per_head = ()
            
            for data in train_loader:
                _, text_batch, _, _ = data
                text_batch = text_batch.to(self.device)
                
                optimizer.zero_grad()
                
                cosine_dists, context_weights, A = net(text_batch)
                scores = context_weights * cosine_dists
                # scores.shape = (batch_size, attention_heads)
                # A.shape = (batch_size, attention_heads, sentence_length)
                
                # get orthogonality penalty: P = (CCT - I)
                I = torch.eye(attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)
                
                # compute loss
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P
                
                dists_per_head += (cosine_dists.cpu().data.numpy(),)
                
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            self.train_dists = np.concatenate(dists_per_head)
            self.train_att_matrix = att_matrix / n_batches
            self.train_att_matrix = self.train_att_matrix.tolist()
        
        self.c = np.squeeze(net.c.cpu().data.numpy())
        self.c = self.c.tolist()
        
        # self.train_top_words = get_top_words_per_context(train_loader.dataset, )
        
        return net

    def test(self, dataset: BaseADDataset, net, ad_score='context_dist_mean'):
        
        net = net.to(self.device)
        attention_heads = net.attention_heads
        
        _, val_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        
        epoch_loss = 0.0
        n_batches = 0
        att_matrix = np.zeros((attention_heads, attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        att_weights = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                idx, text_batch, label_batch, _ = data
                text_batch, label_batch = text_batch.to(self.device), label_batch.to(self.device)
                
                cosine_dists, context_weights, A = net(text_batch)
                scores = context_weights * cosine_dists
                _, best_att_head = torch.min(scores, dim=1)
                
                I = torch.eye(attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) **2)
                
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P
                
                # Save tuples of (idx, label, score, best_att_head) in a list
                dists_per_head += (cosine_dists.cpu().data.numpy(),)
                ad_scores = torch.mean(cosine_dists, dim=1)
                idx_label_score_head += list(zip(idx,
                                                 label_batch.cpu().data.numpy().tolist(),
                                                 ad_scores.cpu().data.numpy().tolist(),
                                                 best_att_head.cpu().data.numpy().tolist()))
                att_weights += A[range(len(idx)), best_att_head].cpu().data.numpy().tolist()
                
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()
                
                epoch_loss += loss.item()
                n_batches +=1
        
        self.test_dists = np.concatenate(dists_per_head)
        self.test_att_matrix = att_matrix / n_batches
        self.test_att_matrix = self.test_att_matrix.tolist()
        
        self.test_dists = np.concatenate(dists_per_head)
        self.test_att_matrix = att_matrix / n_batches
        self.test_att_matrix = self.test_att_matrix.tolist()
        
        self.test_scores = idx_label_score_head
        self.test_att_weights = att_weights
        
        # Compute AUC
        _, labels, scores, _ = zip(*idx_label_score_head)
        labels = np.array(labels)
        scores = np.array(scores)
        
        if np.sum(labels) > 0:
            best_context = None
            if ad_score == 'context_dist_mean':
                self.test_auc = roc_auc_score(labels, scores)
            if ad_score == 'context_best':
                self.test_auc = 0.0
                for context in range(attention_heads):
                    auc_candidate = roc_auc_score(labels, self.test_dists[:, context])
                    print(auc_candidate)
                    if auc_candidate > self.test_auc:
                        self.test_auc = auc_candidate
                        best_context = context
                    else:
                        pass
        else:
            best_context = None
            self.test_auc = 0.0
        
        print(f'Test Loss: {(epoch_loss/n_batches):.6f}')
        print(f'Test AUC: {(100*self.test_auc):.2f}')
        print('Finished validation')

def initialize_context_vectors(net, train_loader, device):
    
    X = ()
    for data in train_loader:
        _, text, _, _ = data
        text = text.to(device)
        
        # fix here
        X_batch = net.pretrained_model(text)
        # X_batch.shape = (sentence_length, batch_size, embedding_size)
        
        X_batch = torch.mean(X_batch, dim=0)
        X_batch = X_batch / torch.norm(X_batch, p=2, dim=1, keepdims=True).clamp(min=1e-08)
        X_batch[torch.isnan(X_batch)] = 0
        # X_batch.shape = (batch_size, embedding_size)
        
        X += (X_batch.cpu().data.numpy(),)
    
    X = np.concatenate(X)
    attention_heads = net.attention_heads
    
    kmeans = KMeans(n_cluster=attention_heads).fit(X)
    centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)
    
    return centers    