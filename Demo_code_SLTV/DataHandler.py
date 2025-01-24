import pickle
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
device = f"cuda:{args.gpu}" if t.cuda.is_available() else "cpu"
import os
class DataHandler:
    def __init__(self):

        if args.data == 'G1':
            predir = './data/G1/'
        elif args.data == 'G2':
            predir = './data/G2/'
        elif args.data == 'G3':
            predir = './data/G3/'

        self.predir = predir

    def LoadData(self):
        if args.mode == 'pre':
            if os.path.exists(self.predir + 'p_train_embedding.pkl'):
                with open(self.predir + 'p_train_embedding.pkl', 'rb') as f:
                    train_data = pickle.load(f)
                self.train_source_embedding = train_data['embed'].to(device)
                s_user_labels = train_data['labels']
                with open(self.predir + 'p_eval_embedding.pkl', 'rb') as f:
                    eval_data = pickle.load(f)
                self.test_target_embedding = eval_data['embed'].to(device)
                tst_user_labels = eval_data['labels']

            else:
                with open(self.predir + 'p_train.pkl', 'rb') as f:
                    source_dict = pickle.load(f)
                with open(self.predir + 'p_eval.pkl', 'rb') as f:
                    test_dict = pickle.load(f)
                train_dict = {}
                eval_dict = {}
                self.train_source_embedding, s_user_labels = self.load_data(source_dict)
                self.test_target_embedding, tst_user_labels = self.load_data(test_dict)
                train_dict['embed'] = self.train_source_embedding
                train_dict['labels'] = s_user_labels
                eval_dict['embed'] = self.test_target_embedding
                eval_dict['labels'] = tst_user_labels
                with open(self.predir + 'p_train_embedding.pkl', 'wb') as f:
                    pickle.dump(train_dict, f)
                with open(self.predir + 'p_eval_embedding.pkl', 'wb') as f:
                    pickle.dump(eval_dict, f)

            strnData = RetainTrnData(s_user_labels)
            self.sourceLoader = dataloader.DataLoader(strnData, batch_size=args.batch, shuffle=True, num_workers=0)
            testData = RetainTrnData(tst_user_labels)
            self.testLoader = dataloader.DataLoader(testData, batch_size=args.batch, shuffle=False, num_workers=0)
        else:
            if os.path.exists(self.predir + 'f_train_embedding.pkl'):
                with open(self.predir + 'f_train_embedding.pkl', 'rb') as f:
                    train_data = pickle.load(f)
                self.train_target_embedding = train_data['embed'].to(device)
                t_user_labels = train_data['labels']
                ##
                with open(self.predir + 'f_test_embedding.pkl', 'rb') as f:
                    test_data = pickle.load(f)
                self.test_target_embedding = test_data['embed'].to(device)
                tst_user_labels = test_data['labels']
                ##
                with open(self.predir + 'p_train_embedding.pkl', 'rb') as f:
                    s_train_data = pickle.load(f)
                self.train_source_embedding = s_train_data['embed'].to(device)
                s_user_labels = s_train_data['labels']
            else:
                with open(self.predir+'p_train.pkl','rb') as f:
                    source_dict=pickle.load(f)
                with open(self.predir+'f_train.pkl','rb') as f:
                    target_dict=pickle.load(f)
                with open(self.predir+'f_test.pkl', 'rb') as f:
                    test_dict=pickle.load(f)
                train_dict = {}
                eval_dict = {}
                self.train_source_embedding,s_user_labels=self.load_data(source_dict)
                self.train_target_embedding, t_user_labels = self.load_data(target_dict)
                self.test_target_embedding, tst_user_labels = self.load_data(test_dict)

                train_dict['embed'] = self.train_target_embedding
                train_dict['labels'] = t_user_labels
                eval_dict['embed'] = self.test_target_embedding
                eval_dict['labels'] = tst_user_labels
                with open(self.predir + 'f_train_embedding.pkl', 'wb') as f:
                    pickle.dump(train_dict, f)
                with open(self.predir + 'f_test_embedding.pkl', 'wb') as f:
                    pickle.dump(eval_dict, f)
            strnData = RetainTrnData(s_user_labels)
            self.sourceLoader = dataloader.DataLoader(strnData, batch_size=args.batch, shuffle=True, num_workers=0)
            ttrnData = RetainTrnData(t_user_labels)
            self.targetLoader = dataloader.DataLoader(ttrnData, batch_size=args.batch, shuffle=True, num_workers=0)
            testData = RetainTrnData(tst_user_labels)
            self.testLoader = dataloader.DataLoader(testData, batch_size=args.batch, shuffle=False, num_workers=0)

    def load_data(self,feature_dict):
        config_id_embedding = feature_dict['config_id_embedding']
        config_id_embedding = t.FloatTensor(list(config_id_embedding.apply(str_to_list).values)).to(device)
        dense_embeddings_com = feature_dict['dense_embeddings_com']
        dense_embeddings_com = t.FloatTensor(list(dense_embeddings_com.apply(str_to_list).values)).to(device)
        dense_embeddings_spe = feature_dict['dense_embeddings_spe']
        dense_embeddings_spe = t.FloatTensor(list(dense_embeddings_spe.apply(str_to_list).values)).to(device)
        sparse_embeddings_com = feature_dict['sparse_embeddings_com']
        sparse_embeddings_com = t.FloatTensor(list(sparse_embeddings_com.apply(str_to_list).values)).to(device)
        sparse_embeddings_spe = feature_dict['sparse_embeddings_spe']
        sparse_embeddings_spe = t.FloatTensor(list(sparse_embeddings_spe.apply(str_to_list).values)).to(device)
        dapan_dense_emb_pooling = feature_dict['dapan_dense_emb_pooling']
        dapan_dense_emb_pooling = t.FloatTensor(list(dapan_dense_emb_pooling.apply(str_to_list).values)).to(device)
        sparse_embeddings_dapan = feature_dict['sparse_embeddings_dapan']
        sparse_embeddings_dapan = t.FloatTensor(list(sparse_embeddings_dapan.apply(str_to_list).values)).to(device)
        seq_embeddings_outputs = feature_dict['seq_embeddings_outputs']
        seq_embeddings_outputs = t.FloatTensor(list(seq_embeddings_outputs.apply(str_to_list).values)).to(device)
        train_embedding = torch.concat([config_id_embedding , dense_embeddings_com , dense_embeddings_spe, sparse_embeddings_com, sparse_embeddings_spe,dapan_dense_emb_pooling,sparse_embeddings_dapan, seq_embeddings_outputs], dim=-1).to(device)
        if args.data == 'G1':
            feature_dict["ltv3"] = feature_dict["ltv3"] - feature_dict["ltv_6h"]
            feature_dict["ltv3"] = feature_dict["ltv3"].apply(lambda x: max(x, 0))
            feature_dict["ltv3"] = feature_dict["ltv3"].apply(lambda x: min(x, 5000))
            user_labels = np.array(feature_dict["ltv3"])
        if args.data == 'G2':
            feature_dict["ltv7"] = feature_dict["ltv7"].apply(lambda x: max(x, 0))
            feature_dict["ltv7"] = feature_dict["ltv7"].apply(lambda x: min(x, 5000))
            user_labels = np.array(feature_dict["ltv7"])
        if args.data == 'G3':
            if 'ltv3' in feature_dict.columns:
                feature_dict["ltv3"] = feature_dict["ltv3"].apply(lambda x: max(x, 0))
                feature_dict["ltv3"] = feature_dict["ltv3"].apply(lambda x: min(x, 5000))
                user_labels = np.array(feature_dict["ltv3"])
            else:
                feature_dict["ltv7"] = feature_dict["ltv7"].apply(lambda x: max(x, 0))
                feature_dict["ltv7"] = feature_dict["ltv7"].apply(lambda x: min(x, 5000))
                user_labels = np.array(feature_dict["ltv7"])
        return train_embedding,user_labels


def str_to_list(s):
    s = s.strip('[]')
    return [float(x) for x in s.split(',')]

class RetainTrnData(data.Dataset):
    def __init__(self, y_data):
        self.y_data = y_data
        self.x_data = np.arange(0, len(y_data))

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]



