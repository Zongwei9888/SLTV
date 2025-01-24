import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from DataHandler import DataHandler
from Model import SSL_Finetune_LTVModel,SSL_LTVModel
import pandas as pd

from params import args
from sklearn import metrics

from Utils.TimeLogger import log
import logging
import os
import sys
import Utils.TimeLogger as logger
from Utils.loss import ziln_loss,ziln_pred,js_div
import Utils.ltv_util as ltv_utils
import datetime
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


class Coach:
    def __init__(self, handler):
        self.handler = handler

    def pre_train_source(self):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/'
        log_file = f'ssl_pretrain_'+ args.data
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        args.save_path = log_file

        com_feature = self.handler.train_source_embedding
        tst_feature = self.handler.test_target_embedding
        input_dim = com_feature.shape[1]
        source_expert = SSL_LTVModel(input_dim).to(device)
        source_expert_optimizer = torch.optim.Adam(source_expert.parameters(), lr=args.lr)
        flag_loss = 2000000
        flag = 0
        recent_auc = 0
        best_auc = 0
        best_epoch =0
        for epoch in tqdm(range(1000)):
            ep_loss = 0
            CL_loss = 0
            Zlin_loss = 0
            trnLoader = self.handler.sourceLoader
            source_expert.train()
            for i, tem in enumerate(trnLoader):
                ancs, label = tem
                ancs = ancs.long().to(device)
                label = label.float().to(device)
                batch_emb = com_feature[ancs]
                source_expert_optimizer.zero_grad()
                source_out,cl_loss = source_expert(batch_emb)
                # cl_loss = cl_loss
                n_loss = ziln_loss(label, source_out)
                loss = cl_loss+n_loss
                loss.backward()
                source_expert_optimizer.step()
                ep_loss += loss.item()
                CL_loss += cl_loss.item()
                Zlin_loss+=n_loss.item()
            logging.info('Epoch: %d, Loss: %.4f Cl_loss: %.4f,Ziln_loss: %.4f  ' % (epoch, ep_loss,CL_loss,Zlin_loss))
            if epoch % 1 == 0:
                result = {
                    args.task_names: [],
                    "probs": [],
                    "is_pay": [],
                    "p" + args.task_names: [],
                }
                with torch.no_grad():
                    source_expert.eval()
                    testLoader = self.handler.testLoader
                    for i, tem in enumerate(testLoader):
                        ancs, label = tem
                        ancs = ancs.long().to(device)
                        label = np.array(label)
                        batch_test_emb = tst_feature[ancs]
                        t_target_out = source_expert.get_test(batch_test_emb)
                        probs, pltv = ziln_pred(t_target_out)
                        result[args.task_names].append(label)
                        result["probs"].append(probs.cpu().numpy().flatten())
                        result["p" + args.task_names].append(pltv.cpu().numpy().flatten())
                        result["is_pay"].append((label > 0).astype(int))

                for key in result:
                    result[key] = np.concatenate(result[key], axis=0)
                result = pd.DataFrame(result)
                metric_result = get_metric(result, args.task_names)
                logging.info('AUC:%.4f, Gini:%.4f  ' % (metric_result['pay_auc'], metric_result['gini']))
                recent_auc = metric_result['pay_auc']
            if recent_auc > best_auc:
                torch.save(source_expert.state_dict(), './Models/'+ args.data+'_source_ssl_model.pkl')
                best_auc = recent_auc
                best_epoch = epoch
                flag = 0

            else:
                flag += 1
            logging.info('epoch:%d, best_auc:%.4f  ' % (best_epoch, best_auc))
            if flag > 20:
                break
    def fine_tune_target(self):
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        log_save = './History/'
        log_file = f'_ssl_fine_tune_{args.lr}_{args.batch}'
        fname = f'{log_file}.txt'
        fh = logging.FileHandler(os.path.join(log_save, fname))
        fh.setFormatter(logging.Formatter(log_format))
        logger = logging.getLogger()
        logger.addHandler(fh)
        logger.info(args)
        logger.info('================')
        args.save_path = log_file

        source_feature = self.handler.train_source_embedding
        target_feature = self.handler.train_target_embedding
        tst_feature = self.handler.test_target_embedding

        input_dim = source_feature.shape[1]

        source_model = SSL_LTVModel(input_dim).to(device)
        target_model = SSL_Finetune_LTVModel(input_dim).to(device)
        source_model_dict = torch.load('./Models/'+ args.data+'_source_ssl_model.pkl')
        source_model.load_state_dict(source_model_dict)
        target_model_dict = target_model.state_dict()

        pretrained_dict = {}
        for k, _ in target_model_dict.items():
            if k in source_model_dict:
                pretrained_dict[k] = source_model_dict[k]
            elif 'p_layer' in k:
                pretrained_dict[k] = source_model_dict['p_layer.' + k.split('.')[1]]
            elif 'mu_layer' in k:
                pretrained_dict[k] = source_model_dict['mu_layer.' + k.split('.')[1]]
            elif 'sigma_layer' in k:
                pretrained_dict[k] = source_model_dict['sigma_layer.' + k.split('.')[1]]

        target_model_dict.update(pretrained_dict)
        target_model.load_state_dict(target_model_dict)
        target_optimizer = torch.optim.Adam(target_model.parameters(), lr=args.lr)

        source_model.eval()
        target_model.train()

        for p in source_model.parameters():
            p.requires_grad = False

        for p in target_model.parameters():
            p.requires_grad = True
        recent_auc = 0
        best_auc = 0
        best_epoch = 0
        flag = 0
        criterion = nn.MSELoss()
        for epoch in tqdm(range(1000)):
            ep_loss = 0
            zlin_loss = 0
            dis_loss = 0
            pltv_loss = 0
            sourceLoader = self.handler.sourceLoader
            targetLoader = self.handler.targetLoader
            testLoader = self.handler.testLoader
            combined_loaders = zip(targetLoader, sourceLoader)
            batch = 0

            for (t_ancs,t_labels),(s_ancs,s_labels) in combined_loaders:
                batch+=1
                t_ancs = t_ancs.long().to(device)
                s_ancs = s_ancs.long().to(device)
                t_labels = t_labels.float().to(device)
                s_labels = s_labels.float().to(device)
                if len(t_ancs)<len(s_ancs):
                    s_ancs = s_ancs[:len(t_ancs)]
                    s_labels = s_labels[:len(t_ancs)]
                if len(t_ancs)>len(s_ancs):
                    t_ancs = t_ancs[:len(s_ancs)]
                    t_labels = t_labels[:len(s_ancs)]
                target_optimizer.zero_grad()

                batch_size = t_ancs.shape[0]
                batch_source_emb = source_feature[s_ancs]
                batch_target_emb = target_feature[t_ancs]


                source_out = source_model.get_scores(batch_source_emb)
                concat_emb = torch.cat((batch_source_emb, batch_target_emb), 0)
                joint_feature, t_source_out, t_target_out = target_model(concat_emb,t_ancs.shape[0])
                #JS Loss
                feat_loss = js_div(joint_feature[:batch_size, :], joint_feature[batch_size:, :])
                predict_loss_t_target = ziln_loss(t_labels, t_target_out)
                predict_loss_t_source = ziln_loss(s_labels, t_source_out)
                predict_loss = predict_loss_t_target+predict_loss_t_source
                # predict_loss = predict_loss_t_target
                loss = feat_loss + predict_loss
                loss.backward()
                target_optimizer.step()
                ep_loss += loss.item()
                zlin_loss += predict_loss.item()
                dis_loss += feat_loss.item()
                # pltv_loss+=l1_loss.item()
            logging.info('Epoch: %d, Loss: %.4f zlin_loss: %.4f,dis_loss: %.4f, pltv_loss:%.4f  ' % (epoch, ep_loss, zlin_loss,dis_loss,pltv_loss))
            if epoch % 1 == 0:
                result = {
                    args.task_names: [],
                    "probs": [],
                    "is_pay": [],
                    "p" + args.task_names: [],
                }
                with torch.no_grad():
                    for i, tem in enumerate(testLoader):
                            ancs, label = tem
                            ancs = ancs.long().to(device)
                            label = np.array(label)
                            batch_test_emb = tst_feature[ancs]
                            t_target_out = target_model.get_scores(batch_test_emb)
                            probs,pltv = ziln_pred(t_target_out)
                            result[args.task_names].append(label)
                            result["probs"].append(probs.cpu().numpy().flatten())
                            result["p" + args.task_names].append(pltv.cpu().numpy().flatten())
                            result["is_pay"].append((label > 0).astype(int))

                for key in result:
                    result[key] = np.concatenate(result[key], axis=0)
                result = pd.DataFrame(result)
                metric_result = get_metric(result, args.task_names)
                logging.info('AUC:%.4f, Gini:%.4f  ' % (metric_result['pay_auc'], metric_result['gini']))
                recent_auc = metric_result['pay_auc']
            if recent_auc > best_auc:
                torch.save(target_model.state_dict(), './Models/'+ args.data+'_target_ssl_model.pkl')
                best_auc = recent_auc
                best_epoch = epoch
                flag = 0

            else:
                flag += 1
            logging.info('best_epoch:%d, best_auc:%.4f  ' % (best_epoch, best_auc))
            if flag > 20:
                break


def get_metric(result, task):
    """计算label_mean、pred_mean、gini、auc"""
    # print(f"user_count:{len(result)}")
    result = result.sort_values(by='probs', ascending=False)
    gain = pd.DataFrame({
        'lorenz': ltv_utils.cumulative_true(result[task], result[task]),
        "model": ltv_utils.cumulative_true(result[task], result["p" + task])
    })
    df_gini = ltv_utils.gini_from_gain(gain[["lorenz", "model"]])
    # print("gini:", df_gini["normalized"][1])
    pay_auc = metrics.roc_auc_score(result["is_pay"], result["probs"])
    metric_result = {
        'label_mean': result[task].mean(),
        'pred_mean': result["p" + task].mean(),
        "gini": df_gini["normalized"][1],
        'pay_auc': pay_auc,
    }

    # top—recall
    # print("rank by pltv:")
    # df_topk = cal_pn_recall(result, task, 'p' + task)
    # print(df_topk)
    return metric_result
def cal_pn_recall(result, task, rank_by):
    """计算准召率"""
    tasksort = result.sort_values(by=task, ascending=False)
    ptasksort = result.sort_values(by=rank_by, ascending=False)

    df_topk = pd.DataFrame(columns=['top', 'return_n', 'pn', 'recall', 'hit', 'bias'])
    tt1 = np.arange(0.001, 0.01, 0.001)
    tt2 = np.arange(0.01, 0.2, 0.01)
    tt = np.concatenate((tt1, tt2))
    for i, t in enumerate(tt):
        return_n = int(t * len(ptasksort['p'+task]))
        df_filtered = ptasksort.iloc[:return_n]
        df_true_filtered = tasksort.iloc[:return_n]

        correct_n = (df_filtered[task] > 0).sum()
        pn = correct_n / len(df_filtered)
        recall = correct_n / (ptasksort[task]>0).sum()

        hit = df_true_filtered['vopenid'].isin(df_filtered['vopenid'])
        hitrate = hit.sum() / len(df_filtered)

        pred_mean = df_filtered[task].mean()
        true_mean = df_true_filtered[task].mean()

        # 打印结果
        tmp = pd.DataFrame({
            'top': t,
            'return_n': len(df_filtered),
            # 'correct_n': correct_n,
            'pn': pn,
            'recall': recall,
            'hit': hitrate,
            'bias': (pred_mean - true_mean) / true_mean,
        }, index=[i])
        df_topk = pd.concat([df_topk, tmp], ignore_index=True)

    return df_topk
if __name__ == '__main__':
    # pre_train_source()
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    coach = Coach(handler)
    coach.fine_tune_target()