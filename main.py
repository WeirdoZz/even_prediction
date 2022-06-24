import argparse
import numpy as np
import random
from dataset import NYT, SpecifiedDataset
from Interaction import *
import logging
import datetime
from time import time
from eval_metrics import dcg_k, precision_at_k, ndcg_k
from model.KEPRL import KEPRL
import torch


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_testsample(val_data,num_events):
    """
    生成测试序列，即标签中的第一个事件和随机采样的100个事件组成的101的序列

    :param val_data: 数据interaction
    :param num_events: 总的事件数量
    :return:
    """

    all_sample=[]
    for target in val_data.sequence.targets:
        test_sample=[]
        for i in range(1):
            one_sample=[]
            # 将验证序列的标签的前i个商品放到onesample中
            one_sample += [target[i]]
            # 从所有商品的id中去掉第一个商品的id
            other = list(range(1, num_events))
            other.remove(target[i])
            # 再随机抽取100个负样本加到后面去
            neg = random.sample(other, 10000)
            # neg=other
            one_sample+=neg
            test_sample.append(one_sample)
        test_sample=np.stack(test_sample)
        all_sample.append(test_sample)
    all_sample=np.stack(all_sample)
    return all_sample


def my_evaluate_keprl(model, val_data, config):

    num_chains=len(val_data.sequence.sequences)
    chain_indexes=np.arange(num_chains)
    batch_size = 1024
    num_batches = int(num_chains / batch_size) + 1
    sequences=val_data.sequence.sequences
    target_sequences=val_data.sequence.targets

    dcg_score=0

    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size

        if batch == num_batches - 1:
            if start < num_chains:
                end = num_chains
            else:
                break

        batch_chain_index=chain_indexes[start:end]
        batch_sequences=sequences[batch_chain_index]
        # batch_valid_sequences = np.atleast_2d(batch_sequences)
        batch_len=val_data.sequence.length[batch_chain_index]
        targets=target_sequences[batch_chain_index]


        batch_sequences=torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
        batch_len=torch.from_numpy(batch_len).type(torch.LongTensor).to(device)

        sample_index=[]
        with torch.no_grad():
            prediction_prob,h,batch_kg= model(batch_sequences, batch_len)
            prediction_prob=prediction_prob.squeeze()
            m = torch.distributions.categorical.Categorical(prediction_prob)
            # 从这个分布中随机抽取一个索引（按照分布来的）
            sample = m.sample()
            sample_index.append(sample)
            # sampling_embedding = model.item_embeddings(sample)
            # sampling_embedding = sampling_embedding.unsqueeze(1)

            for i in range(config.T-1):
                prediction_prob,h,batch_kg=model.forward_after(sample,h,batch_kg)
                prediction_prob = prediction_prob.squeeze()
                m = torch.distributions.categorical.Categorical(prediction_prob)
                # 从这个分布中随机抽取一个索引（按照分布来的）
                sample = m.sample()
                sample_index.append(sample)
                # sampling_embedding = model.item_embeddings(sample)
                # sampling_embedding = sampling_embedding.unsqueeze(1)

        sample_index=torch.stack(sample_index,dim=1)
        dcg_score+=dcg_k(targets,sample_index,2)

    return dcg_score/num_batches

def evaluate_keprl(model,val_data,num_events):

    num_chains=len(val_data.sequence.sequences)
    chain_indexes=torch.arange(num_chains).to(device)
    batch_size=1024
    num_batchs=int(num_chains/batch_size)+1

    pred_list=None

    # 验证集的序列和对应的序列长度
    valid_sequences=torch.from_numpy(val_data.sequence.sequences).type(torch.LongTensor).to(device)
    valid_seq_len=torch.from_numpy(val_data.sequence.length).type(torch.LongTensor).to(device)


    # 生成测试用的样本
    all_sample =generate_testsample(val_data,num_events)

    for batch in range(num_batchs):
        start=batch*batch_size
        end=start+batch_size

        if batch == num_batchs - 1:
            if start < num_chains:
                end = num_chains
            else:
                break

        batch_chain_index = chain_indexes[start:end]
        batch_valid_sequences=valid_sequences[batch_chain_index]
        batch_valid_len=valid_seq_len[batch_chain_index]

        # batch_valid_len = torch.from_numpy(batch_valid_len).type(torch.LongTensor).to(device)
        # batch_valid_sequences = torch.from_numpy(batch_valid_sequences).type(torch.LongTensor).to(device)

        prediction_score ,_,_= model(batch_valid_sequences, batch_valid_len)
        rating_pred = prediction_score
        rating_pred = rating_pred.cpu().data.numpy().copy()

        if batch==0:
            pred_list=rating_pred
        else:
            pred_list = np.append(pred_list, rating_pred, axis=0)

    all_top10=[]
    for i in range(1):
        oneloc_top10=[]
        for pred,sample in zip(pred_list[:,i,:],all_sample[:,i,:]):
            # 将101个商品的概率取出并取负数
            each_sample = -pred[sample]
            # 获取概率最大的10个商品的索引
            top10index = np.argsort(each_sample)[:10]
            # 从101个商品中获取这10个商品
            top10item = sample[top10index]
            oneloc_top10.append(top10item)
        oneloc_top10=np.stack(oneloc_top10)
        all_top10.append(oneloc_top10)
    all_top10=np.stack(all_top10,axis=1)
    pred_list=all_top10

    precision,ndcg=[],[]
    k = 10
    for i in range(1):
        pred = pred_list[:, i, :]
        # hit10
        precision.append(precision_at_k(val_data.sequence.targets, pred, i))
        ndcg.append(ndcg_k(val_data.sequence.targets, pred, k, i))

    return precision, ndcg


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop


def train_keprl(train_data,dev_data,test_data,config,kg_map,num_events):
    # 获取训练集的序列的相关数据
    sequences_torch = torch.from_numpy(train_data.sequence.sequences).type(torch.LongTensor).to(device)
    targets_torch = torch.from_numpy(train_data.sequence.targets).type(torch.LongTensor).to(device)
    chain_torch = torch.from_numpy(train_data.sequence.chain_ids).type(torch.LongTensor).to(device)
    trainlen_torch = torch.from_numpy(train_data.sequence.length).type(torch.LongTensor).to(device)
    tarlen_torch = torch.from_numpy(train_data.sequence.tarlen).type(torch.LongTensor).to(device)

    n_train=len(sequences_torch)
    logger.info("训练集的数据条数:{}".format(n_train))

    # 知识表示转换成tensor表示，放到gpu上
    kg_map = torch.from_numpy(kg_map).type(torch.FloatTensor).to(device)
    kg_map.requires_grad = False

    seq_model = KEPRL(num_events, config, device, kg_map).to(device)
    optimizer = torch.optim.Adam(seq_model.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    # 两个计算损失的地方,最终的损失是这两个损失按照权重相加
    lamda = 5
    CEloss = torch.nn.CrossEntropyLoss()
    margin = 0.0
    MRLoss = torch.nn.MarginRankingLoss(margin=margin).to(device)

    record_indexes = torch.arange(n_train).to(device)  # 训练集的索引
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1

    max_dcg_score=0
    max_score_epoch=0

    # 控制提前结束训练的参数
    stopping_step = 0
    cur_best_pre_0 = 0
    should_stop = False

    for epoch in range(config.n_iter):
        t1=time()
        loss=0
        seq_model.train()

        # 将索引打乱用于之后的随机分batch
        torch.randperm(record_indexes.nelement())

        epoch_reward=0.0  # 当前周期的奖励
        epoch_loss=0.0  # 当前周期的损失

        for batch in range(num_batches):
            start=batch*batch_size
            end=start+batch_size

            if batch==num_batches-1:
                if start<n_train:
                    end=n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]  # 当前batch的对应的数据的索引(mask)

            # 获取当前batch对应的数据
            batch_chains = chain_torch[batch_record_index]
            batch_sequences = sequences_torch[batch_record_index]
            batch_targets = targets_torch[batch_record_index]
            trainlen = trainlen_torch[batch_record_index]
            tarlen = tarlen_torch[batch_record_index]

            # 全部转成tensor格式
            # tarlen = torch.from_numpy(tarlen).type(torch.LongTensor).to(device)
            # trainlen = torch.from_numpy(trainlen).type(torch.LongTensor).to(device)
            # batch_chains = torch.from_numpy(batch_chains).type(torch.LongTensor).to(device)
            # batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            # batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)

            # 真实标签
            events_to_predict = batch_targets

            # 构建一个真实标签的类似one_hot的编码
            pred_one_hot = torch.zeros((len(batch_chains), num_events))
            # 获取当前batch的标签(本可以使用batch_targets的,但是上面已经转成tensor了,为了不影响后续,就重新获取一次)
            # batch_tar = targets_torch[batch_record_index]
            # 给每一个真实标签加上一点值
            for i, tar in enumerate(batch_targets):
                pred_one_hot[i][tar] = 0.2
            # pred_one_hot = torch.from_numpy(pred_one_hot).type(torch.FloatTensor).to(device)
            pred_one_hot=pred_one_hot.to(device)

            prediction_score, origin, batch_targets, Reward, dist_sort = seq_model.RL_train(batch_sequences,
                                                                                            events_to_predict,
                                                                                            pred_one_hot, trainlen,
                                                                                            tarlen)
            target = torch.ones((len(prediction_score))).unsqueeze(1).to(device)

            # 最小的seq奖励对应的kg奖励和最大的seq奖励对应的kg奖励
            min_reward = dist_sort[0, :].unsqueeze(1)
            max_reward = dist_sort[-1, :].unsqueeze(1)
            # 求一个margin ranking 损失
            mrloss = MRLoss(max_reward, min_reward, target)

            # 将预测概率去掉stack多出来的那一维
            origin = origin.view(prediction_score.shape[0] * prediction_score.shape[1], -1)
            # 获取第一个预测的事件，平铺开
            target = batch_targets.view(batch_targets.shape[0] * batch_targets.shape[1])
            # 讲对应的奖励平铺开，奖励的维度应该和target的维度是一样的
            reward = Reward.view(Reward.shape[0] * Reward.shape[1]).to(device)

            # 将所有事件链的预测应用到每个事件链上，得到一个2048*2048的张量
            prob = torch.index_select(origin, 1, target)
            # 取其对角就是该事件链的下一个最大概率的事件发生的概率
            prob = torch.diagonal(prob, 0)
            # 获取rl损失
            RLloss = -torch.mean(torch.mul(reward, torch.log(prob)))
            loss = RLloss + lamda * mrloss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= num_batches
        t2 = time()
        output_str = "训练周期 %d [%.1f s]  loss=%.4f" % (epoch + 1, t2 - t1, epoch_loss)
        logger.info(output_str)


        seq_model.eval()
        """
        两种验证方法，一种看预测的全部序列和对应的真实序列的dcg得分，一种看预测序列中的第一个的精度
        """
        # dcg_score= my_evaluate_keprl(seq_model, dev_data, config)
        # logger.info(f"验证周期：{epoch + 1}，dcg得分：{dcg_score}")
        # if dcg_score > max_dcg_score:
        #     max_dcg_score=dcg_score
        #     max_score_epoch=epoch
        # if max_score_epoch-epoch==100:
        #     logger.info(f"最优验证精度在{max_score_epoch}达到，dcg得分为：{max_dcg_score}")
        #     break

        precision, ndcg = evaluate_keprl(seq_model, dev_data, num_events)
        logger.info('精度'.join(str(e) for e in precision))
        logger.info('ndcg得分'.join(str(e) for e in ndcg))
        logger.info("验证时间:{}".format(time() - t2))
        cur_best_pre_0, stopping_step, should_stop = early_stopping(precision[0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                    flag_step=10)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
    logger.info("\n")
    logger.info("\n")


if __name__=="__main__":

    print(f"使用{device}计算..")
    parser = argparse.ArgumentParser()

    """
    L:最大的序列长度
    T:预测事件的长度
    """

    parser.add_argument('--L', type=int, default=9)
    parser.add_argument('--T', type=int, default=2)

    # train arguments
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument("--dataset",type=str,default="NYT")

    # model dependent arguments
    parser.add_argument('--d', type=int, default=50)

    config = parser.parse_args()

    if config.dataset=="NYT":
        dataset=SpecifiedDataset.NYT()
    elif config.dataset=="company_event":
        dataset=SpecifiedDataset.CompanyEvent()

    train_set,dev_set,test_set,num_events,kg_map=dataset.generatet_dataset()

    train_data=Sequence(train_set)
    dev_data = Sequence(dev_set)
    test_data = Sequence(test_set)

    train_data.to_train_sequence(config.L, config.T)
    dev_data.to_val_sequence(config.L, config.T)
    test_data.to_val_sequence(config.L, config.T)

    logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info(config)

    train_keprl(train_data, dev_data, test_data, config, kg_map,num_events)