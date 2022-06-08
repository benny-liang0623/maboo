import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from cvdd_net import CVDDNet
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans


class BaseTrainer(ABC):
    
    def __init__(self, lr, n_epochs, lr_milestones, weight_decay, device):
        super().__init__()
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.weight_decay = weight_decay
        self.device = device
        
        self.test_auc = None
        self.test_scores = None
    
    @abstractmethod
    def train(self, dataset, net):
        pass
    
    @abstractmethod
    def test(self, dataset, net):
        pass


class CVDDTrainer(BaseTrainer):
    
    def __init__(self, lr: float=0.001, n_epochs: int=150, lr_milestones: tuple=(),
                 lambda_p: float=0.0, alpha_scheduler: str='hard', weight_decay: float=1e-6, device: str='cuda'):
        super().__init__(lr, n_epochs, lr_milestones, weight_decay, device)
        
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
        
    def train(self, train_loader, net: CVDDNet):
        logger = logging.getLogger()
        
        net = net.to(self.device)
        attention_heads = net.attention_heads
        
        net.c.data = torch.from_numpy(
            initialize_context_vectors(net, train_loader, self.device)[np.newaxis, :]).to(self.device)
        
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        logger.info("Start training")
        net.train()
        alpha_i = 0
        for epoch in range(self.n_epochs):
            
            if epoch in self.alpha_milestones:
                net.alpha = float(self.alphas[alpha_i])
                logger.info(f'  Temperature alpha scheduler: new alpha is {net.alpha}')
                alpha_i += 1
            
            if epoch in self.lr_milestones:
                logger.info(f'  LR scheduler: new learning rate is {scheduler.get_lr()[0]}')
            
            epoch_loss = 0.0
            n_batches = 0
            att_matrix = np.zeros((attention_heads, attention_heads))
            dists_per_head = ()
            
            for data in train_loader:
                ids = data['ids'].to(self.device)
                mask = data['mask'].to(self.device)
                
                optimizer.zero_grad()
                
                cosine_dists, context_weights, A = net(ids, mask)
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
                scheduler.step()
                
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} |'
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            
            self.train_dists = np.concatenate(dists_per_head)
            self.train_att_matrix = att_matrix / n_batches
            self.train_att_matrix = self.train_att_matrix.tolist()
        
        self.c = np.squeeze(net.c.cpu().data.numpy())
        self.c = self.c.tolist()
        
        logger.info('Finish training')
        
        return net

    def test(self, val_loader, net, ad_score='context_dist_mean'):
        logger = logging.getLogger()
        
        net = net.to(self.device)
        attention_heads = net.attention_heads
        
        logger.info('Starting testing')
        epoch_loss = 0.0
        n_batches = 0
        att_matrix = np.zeros((attention_heads, attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        att_weights = []
        net.eval()
        with torch.no_grad():
            for data in val_loader:
                idx = data['index']
                ids = data['ids'].to(self.device)
                mask = data['mask'].to(self.device)
                targets = data['targets'].to(self.device)
                
                cosine_dists, context_weights, A = net(ids, mask)
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
                                                 targets.cpu().data.numpy().tolist(),
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
        
        logger.info(f'Test Loss: {(epoch_loss/n_batches):.6f}')
        logger.info(f'Test AUC: {(100*self.test_auc):.2f}')
        logger.info(f'Test Best Context: {best_context}')
        logger.info('Finished validation')


def initialize_context_vectors(net, train_loader, device):
    """
    Initialize the context vectors from an initial run of k-means++ on simple average sentence embeddings
    Returns
    -------
    centers : ndarray, [n_clusters, n_features]
    """
    logger = logging.getLogger()
    logger.info('Initialize context vectors.')
    # Get vector representations
    X = ()
    for data in train_loader:
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        # text.shape = (sentence_length, batch_size)

        X_batch = net.pretrained_model(ids, mask)
        # X_batch.shape = (sentence_length, batch_size, embedding_size)

        # compute mean and normalize
        X_batch = torch.mean(X_batch, dim=0)
        X_batch = X_batch / torch.norm(X_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
        X_batch[torch.isnan(X_batch)] = 0
        # X_batch.shape = (batch_size, embedding_size)

        X += (X_batch.cpu().data.numpy(),)

    X = np.concatenate(X)
    n_attention_heads = net.attention_heads

    kmeans = KMeans(n_clusters=n_attention_heads).fit(X)
    centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)

    logger.info("Context vectors initialized.")
    
    return centers


class CVDD(object):
    """A class for Context Vector Data Description (CVDD) models."""

    def __init__(self, ad_score='context_dist_mean'):
        """Init CVDD instance."""

        # Anomaly score function
        self.ad_score = ad_score

        # CVDD network: pretrained_model (word embedding or language model) + self-attention module + context vectors
        self.net_name = None
        self.net = None

        self.trainer = None

        self.train_dists = None
        self.train_top_words = None

        self.test_dists = None
        self.test_top_words = None
        self.test_att_weights = None

        self.results = {
            'context_vectors': None,
            'train_att_matrix': None,
            'test_att_matrix': None,
            'test_auc': None,
            'test_scores': None
        }

    def set_network(self, pretrained_model_name, attention_size=150, n_attention_heads=3):
        """Builds the CVDD network composed of a pretrained_model, the self-attention module, and context vectors."""
        self.net = CVDDNet(pretrained_model_name=pretrained_model_name, attention_size=attention_size, attention_heads=n_attention_heads)

    def train(self, dataset: DataLoader, lr: float = 0.001, n_epochs: int = 25, lr_milestones: tuple = (), lambda_p: float = 1.0,
              alpha_scheduler: str = 'logarithmic', weight_decay: float=0.5e-6, device: str='cuda'):
        """Trains the CVDD model on the training data."""
        self.trainer = CVDDTrainer(lr, n_epochs, lr_milestones, lambda_p, alpha_scheduler, weight_decay, device)
        self.net = self.trainer.train(dataset, self.net)

        # Get results
        self.train_dists = self.trainer.train_dists
        self.train_top_words = self.trainer.train_top_words
        self.results['context_vectors'] = self.trainer.c
        self.results['train_att_matrix'] = self.trainer.train_att_matrix

    def test(self, dataset: DataLoader, device: str = 'cuda'):
        """Tests the CVDD model on the test data."""

        if self.trainer is None:
            self.trainer = CVDDTrainer(device)

        self.trainer.test(dataset, self.net, ad_score=self.ad_score)

        # Get results
        self.test_dists = self.trainer.test_dists
        self.test_top_words = self.trainer.test_top_words
        self.test_att_weights = self.trainer.test_att_weights
        self.results['test_att_matrix'] = self.trainer.test_att_matrix
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_scores'] = self.trainer.test_scores

    def save_model(self, export_path):
        """Save CVDD model to export_path."""
        torch.save(self.net.state_dict(), f'/{export_path}/model.ckpt')

    def load_model(self, import_path, device: str = 'cuda'):
        """Load CVDD model from import_path."""
        self.net = torch.load(f'{import_path}',
                              map_location=device)

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)