import torch
import pandas as pd
import os
import itertools
import logging
import json
import itertools
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler, Dataset
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F 
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import sys
sys.path.append('..')
from tqdm import tqdm

class WSIDataset(Dataset):
    def __init__(self, args, wsi_labels, infold_cases, phase=None):
        self.args = args
        self.wsi_labels = wsi_labels
        self.phase = phase

        self.infold_features, self.infold_labels = [], []
        for case_id, slide_id, label in wsi_labels:
            if case_id in infold_cases:
                if args.pretrain in ['ResNet50_ImageNet']:
                    if self.args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']:
                        fea_path = os.path.join(args.feature_dir, args.pretrain, slide_id+'.pt')
                        if os.path.exists(fea_path):
                            self.infold_features.append(fea_path)
                            self.infold_labels.append(label)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                    
    def __len__(self):
        return len(self.infold_features)
        
    def __getitem__(self, index):
        path = self.infold_features[index]
        label = self.infold_labels[index]
        fea = torch.load(path)
        return fea, label, path
   
class DMINMIL:
    def __init__(self, args):
        self.args = args
        assert self.args.model == 'v4'

        if args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']:
            self.train_loader, self.valid_loader, self.test_loader = self.init_data_wsi()
        else:
            raise NotImplementedError

        self.model = self.init_model()
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params / 1e6} M")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.l2_loss = torch.nn.MSELoss(reduction='mean')

        self.counter = 0
        self.patience = 0 # 20
        self.stop_epoch = 0 # 50
        self.best_loss = np.Inf
        self.flag = 1
        self.ckpt_name = os.path.join(self.args.ckpt_dir, 'best_epoch.pth')
        
        self.best_valid_metrics = None 

    def read_wsi_label(self):
        data = pd.read_csv(self.args.label_csv)

        wsi_labels = []
        for i in range(len(data)):
            case_id, slide_id, label = data.loc[i, "case_id"], data.loc[i, "slide_id"], data.loc[i, "label"]

            if self.args.dataset in ['Camelyon16']:
                assert label in ['tumor_tissue', 'normal_tissue']
                label = 0 if label == 'normal_tissue' else 1
            elif self.args.dataset == 'TCGA-NSCLC':
                assert label in ['TCGA-LUSC', 'TCGA-LUAD']
                label = 0 if label == 'TCGA-LUSC' else 1
            elif self.args.dataset == 'TCGA-BRCA':
                assert label in ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma']
                label = 0 if label == 'Infiltrating Ductal Carcinoma' else 1
            elif self.args.dataset == 'TCGA-RCC':
                assert label in ['TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP']
                if label == 'TCGA-KICH':
                    label = 0
                elif label == 'TCGA-KIRC':
                    label = 1
                elif label == 'TCGA-KIRP':
                    label = 2
            else:
                raise NotImplementedError

            wsi_labels.append([case_id, slide_id, label])   

        return wsi_labels

    def read_in_fold_cases(self, fold_csv):
        data = pd.read_csv(fold_csv)
        train_cases, valid_cases, test_cases = data.loc[:, 'train'].dropna(axis=0, how='any').to_list(), data.loc[:, 'val'].dropna(axis=0, how='any').to_list(), data.loc[:, 'test'].dropna(axis=0, how='any').to_list()
        return train_cases, valid_cases, test_cases

    def make_weights_for_balanced_classes_split(self, data_set):
        N = float(len(data_set))           

        classes = {}
        for label in data_set.infold_labels:
            if label not in classes:
                classes[label] = 1
            else:
                classes[label] += 1
                                                                                                    
        weight = [0] * int(N)                                           
        for idx in range(len(data_set)):   
            y = data_set.infold_labels[idx]                       
            weight[idx] = N / classes[y]    
            
        return torch.DoubleTensor(weight)

    def init_data_wsi(self):
        wsi_labels = self.read_wsi_label()

        split_dir = os.path.join(self.args.split_dir, '{}-fold-{}%-label/').format(self.args.fold, int(self.args.label_frac * 100))
        train_cases, valid_cases, test_cases = self.read_in_fold_cases(os.path.join(split_dir + 'splits_{}.csv'.format(self.args.k)))


        train_set = WSIDataset(self.args, wsi_labels, train_cases, 'train')
        valid_set = WSIDataset(self.args, wsi_labels, valid_cases, 'valid')
        test_set = WSIDataset(self.args, wsi_labels, test_cases, 'test')
        
        logging.info("Case/WSI number for trainset in fold-{} = {}/{}".format(self.args.k, len(train_cases), len(train_set)))
        logging.info("Case/WSI number for validset in fold-{} = {}/{}".format(self.args.k, len(valid_cases), len(valid_set)))
        logging.info("Case/WSI number for testset in fold-{} = {}/{}".format(self.args.k, len(test_cases), len(test_set)))

        weights = self.make_weights_for_balanced_classes_split(train_set)

        if self.args.dataset in ['Camelyon16', 'TCGA-NSCLC', 'TCGA-BRCA', 'TCGA-RCC']:
            train_loader = DataLoader(train_set, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights), replacement=True))
            valid_loader = DataLoader(valid_set, batch_size=1, sampler = SequentialSampler(valid_set))
            test_loader = DataLoader(test_set, batch_size=1, sampler = SequentialSampler(test_set))

            return train_loader, valid_loader, test_loader
        else:
            raise NotImplementedError

    def init_model(self): 
        from models.hdmil import KAN_CLAM_MB_v4, SmoothTop1SVM
        model = KAN_CLAM_MB_v4(I=self.args.feature_dim, dropout = True, n_classes = self.args.n_classes, subtyping = self.args.subtyping,
            instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda(self.args.device), k_sample = self.args.k_sample, args=self.args).to(self.args.device)
        return model
    

    def train(self):
        step = 0
        

        for epoch in range(1, self.args.n_epochs + 1):

            avg_train_loss = 0

            self.model.train()


            for i, (fea, label, fea_path) in enumerate(tqdm(self.train_loader)):

                step += 1
                fea, label = fea.to(self.args.device), label.to(self.args.device)

                self.optimizer.zero_grad()


                loss = self.train_inference(fea, label)

                if torch.isnan(loss):
                    continue
                
                avg_train_loss += loss.item()
            
                loss.backward()
                self.optimizer.step()


            avg_train_loss /= (i + 1)
           
            logging.info("In step {} (epoch {}), average train loss = {:.4f}".format(step, epoch, avg_train_loss))
            
            self.valid(epoch)

            if self.flag == -1:
                break

        return self.best_valid_metrics

    def valid(self, epoch):
        avg_loss = 0
        self.model.eval()

        labels, probs = [], []

       
        for i, (fea, label, _) in enumerate(tqdm(self.valid_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label)

            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()

        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)


        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))

        if epoch >= self.stop_epoch:
            if avg_loss < self.best_loss:
                self.counter = 0
                logging.info(f'Validation loss decreased ({self.best_loss:.4f} --> {avg_loss:.4f}).  Saving model ...')
                torch.save(self.model.state_dict(), self.ckpt_name)
                self.best_loss = avg_loss
                self.best_valid_metrics = [avg_loss, auc, acc, precision, recall, f1]
            else:
                self.counter += 1
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.flag = -1

    def test(self):
        avg_loss = 0
        self.model.load_state_dict(torch.load(self.ckpt_name))
        self.model.eval()

        labels, probs = [], []


        for i, (fea, label, _) in enumerate(tqdm(self.test_loader)):
            fea, label = fea.to(self.args.device), label.to(self.args.device)
            with torch.no_grad():
                loss, y_prob = self.test_inference(fea, label)

            labels.append(label.data.cpu().numpy())
            probs.append(y_prob.data.cpu().numpy())
            avg_loss += loss.item()

        avg_loss /= (i + 1)

        labels, probs = np.concatenate(labels, 0), np.concatenate(probs, 0)

        auc = self.cal_AUC(probs, labels, self.args.n_classes)
        acc, acc_log, precision, recall, f1 = self.cal_ACC(probs, labels, self.args.n_classes)

        logging.info("loss = {:.4f}, auc = {:.4f}, acc = {:.4f}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}".\
            format(avg_loss, auc, acc, precision, recall, f1))

        return avg_loss, auc, acc, precision, recall, f1

    def train_inference(self, fea, label):      

        bag_logit, _, results_dict, M, select_logit, select_M, mask_rate = self.model(fea, label, instance_eval=True)

        cls_loss = 0.7 * self.loss(bag_logit, label) + 0.3 * results_dict['instance_loss'].mean()
        kl_loss = self.kl_loss(F.log_softmax(select_logit), F.log_softmax(bag_logit))
        dis_loss = self.l2_loss(select_M, M)
        rate_loss = self.l2_loss(mask_rate, torch.Tensor([self.args.mask_ratio]).to(self.args.device))

        loss = cls_loss + 0.5 * kl_loss + 0.5 * dis_loss + 2 * rate_loss

        return loss
 
    
    def test_inference(self, fea, label):

        bag_logit = self.model(fea)
       

        loss = self.loss(bag_logit, label)
        y_prob = F.softmax(bag_logit, dim=1)
      
        return loss, y_prob

    def cal_AUC(self, probs, labels, nclasses):
        '''
            probs(softmaxed): ndarray, [N, nclass] 
            labels(inte number): ndarray, [N, 1] 
        '''
        if nclasses == 2:
            auc_score = roc_auc_score(labels, probs[:, 1])
        else:
            auc_score = roc_auc_score(labels, probs, multi_class='ovr')

        return auc_score
    

    def cal_ACC(self, probs, labels, nclasses):
        '''
            probs(softmaxed): ndarray, [N, nclass] 
            labels(inte number): ndarray, [N, 1] 
        '''
        log = [{"count": 0, "correct": 0} for i in range(nclasses)]
        pred_hat = np.argmax(probs, 1)
        labels = labels.astype(np.int32)

        if nclasses == 2:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels, pred_hat, average='binary')
            recall = recall_score(labels, pred_hat, average='binary')
            f1 = f1_score(labels, pred_hat, average='binary')

            return acc_score, log, precision, recall, f1

        else:
            acc_score = accuracy_score(labels, pred_hat)
            precision = precision_score(labels,pred_hat,average='macro')
            recall = recall_score(labels,pred_hat,average='macro')
            f1 = f1_score(labels, pred_hat, labels=list(range(self.args.n_classes)), average='macro')

            return acc_score, log, precision, recall, f1