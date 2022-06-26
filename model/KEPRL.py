import torch
from torch import nn
from model.dunamicGRU import DynamicGRU
import torch.nn.functional as F
from eval_metrics import *


class KEPRL(nn.Module):

    def __init__(self,num_events,model_args,device,kg_map):
        super(KEPRL, self).__init__()

        self.args = model_args
        self.device = device
        self.lamda = 10

        dims = self.args.d

        self.kg_map = kg_map
        # 对事件的id做embedding的层，这里还是无关知识的，只是用于将id做一个embedding
        self.item_embeddings = nn.Embedding(num_events, dims).to(device)
        self.DP = nn.Dropout(0.5)
        self.enc = DynamicGRU(input_dim=dims, output_dim=dims, bidirectional=False)
        # 这个mlp的输入就是gru的输出的隐状态和池化后的喜好表示和预测未来的喜好表示
        self.mlp_history = nn.Linear(50, 50)

        # 这个mlp的输入就是前面三种表示的拼接，然后通过fc输出一个可能选取的商品的概率
        self.mlp = nn.Linear(dims, dims * 2)
        self.fc = nn.Linear(dims * 2, num_events)

        self.BN = nn.BatchNorm1d(50, affine=False)
        # 计算余弦相似度的层
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self,batch_sequences,len):
        """

        :param batch_sequences: 多个batch的用户购买序列(b,l,f)
        :param len: 每个序列中的batch的个数
        :return:
        """

        probs=[]
        # 对输入的序列进行embedding
        input = self.item_embeddings(batch_sequences)
        # 进GRU获取到序列级的知识表示
        out_enc, h = self.enc(input, len)

        # 对事件的知识嵌入向量做一个批量归一化
        # kg_map = self.BN(self.kg_map)
        # kg_map = kg_map.detach()
        # batch_kg = self.get_kg(batch_sequences, len, kg_map)

        mlp_in = h.squeeze()#torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        out=F.softmax(out,dim=-1)
        probs.append(out)
        return torch.stack(probs, dim=1),h#,batch_kg

    def forward_after(self,batch_sequences,h,batch_kg):
        """
        用于预测完第一个之后预测之后的事件用

        :param batch_sequences: 上次预测的事件的id
        :param h: 上次预测完gru输出的隐状态
        :param batch_kg: 本次预测最初的事件序列
        :return:
        """

        probs = []
        # 对输入的序列进行embedding
        input = self.item_embeddings(batch_sequences)
        input=input.unsqueeze(1)
        # 进GRU获取到序列级的知识表示
        out_enc, h = self.enc(input, h,one=True)
        mlp_in = torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)

        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)

        out = self.fc(mlp_hidden)
        out = F.softmax(out,dim=-1)
        probs.append(out)
        return torch.stack(probs, dim=1),h,batch_kg


    def get_kg(self, batch_sequence, len, kg_map):
        """
        获取事件链中事件嵌入的平均池化以获取当前的喜好表示

        :param batch_sequence: 事件链
        :param len: 每个序列包含的batch数量
        :param kg_map: 商品的嵌入向量表
        :return: 平均池化之后的batch sequence，会降一维吧
        """
        batch_kg = []
        # 下面就是做一个平均池化，获取当前的喜好
        for i, seq in enumerate(batch_sequence):
            # sequence中是内部id也就是0，1，2，3，因此需要用kg_map将他映射成嵌入向量

            seq_kg = kg_map[seq]
            seq_kg_avg = torch.sum(seq_kg, dim=0)
            seq_kg_avg = torch.div(seq_kg_avg, len[i])
            batch_kg.append(seq_kg_avg)

        # 将池化之后的数据再次拼接起来
        batch_kg = torch.stack(batch_kg)
        return batch_kg

    def RL_train(self,batch_sequences,events_to_predict,pred_one_hot,train_len,target_len):
        """
        RL训练的一个过程

        :param batch_sequences: 一个batch中的选到的事件链的序列
        :param events_to_predict: 需要预测的真实序列
        :param pred_one_hot: 真实序列编码的类似one-hot信息
        :param train_len: batch_sequences中的每个序列的长度
        :param target_len: 真实序列中每个序列的长度
        :return: 预测的第一个的带真实标签的概率分布,第一个的不带真实标签的概率分布,第一个的sample集合,总奖励,按照seq奖励高低排序的kg奖励(包含三次)
        """

        probs = []  # 加上了真实标签信息的概率
        probs_origin = []  # 模型原始的mlp输出的概率
        each_sample = []
        Rewards = []

        # 对序列id做一个embedding,无关知识,形状应该是[[[...],[...],...],...],0维的长度是当前batch的用户数量
        input = self.item_embeddings(batch_sequences)
        # 使用gru编码序列层面的表示
        out_enc, h = self.enc(input, train_len)

        # 根据知识图获取当前的喜好知识表示
        # kg_map = self.BN(self.kg_map)
        # batch_kg = self.get_kg(batch_sequences, train_len, kg_map)

        # 根据当前喜好获取未来的喜好，并且做一个cat
        mlp_in = h.squeeze()#torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)

        # 再用一个mlp对上面的结果做一个映射，最终fc的输出的维度是总商品数量
        mlp_hidden = self.mlp(mlp_in)
        mlp_hidden = torch.tanh(mlp_hidden)
        out_fc = self.fc(mlp_hidden)

        # 做一个softmax，获取可能的商品概率
        out_distribution = F.softmax(out_fc, dim=1)
        # out_distribution = self.softmax(out_fc)
        probs_origin.append(out_distribution)
        out_distribution = out_distribution * 0.8
        # 将真实的标签的信息添加到预测的概率上
        out_distribution = torch.add(out_distribution, pred_one_hot)
        probs.append(out_distribution)

        m = torch.distributions.categorical.Categorical(out_distribution)
        # 从这个分布中随机抽取一个索引（按照分布来的）
        sample1 = m.sample()
        each_sample.append(sample1)

        # 对于该action我们获取奖励
        Reward = self.generateReward(sample1, self.args.T - 1, 3, events_to_predict,pred_one_hot, h ,target_len)
        Rewards.append(Reward)

        probs = torch.stack(probs, dim=1)
        probs_origin = torch.stack(probs_origin, dim=1)

        return probs, probs_origin, torch.stack(each_sample, dim=1), torch.stack(Rewards, dim=1)

    def generateReward(self, sample1, path_len, path_num, items_to_predict,pred_ont_hot, h_origin, target_len):
        """

        :param sample1: action，是一个事件的id
        :param path_len: target_len 减去1，因为已经有了一个预测值了 就是传进来的sample1，所以少预测一个
        :param path_num: 预测几次,取总和的平均值作为代表
        :param items_to_predict: 真实标签
        :param pred_ont_hot: 真实标签的独热编码（对应位置的值是0.2）
        :param h_origin: gru输出的h
        :param batch_kg: 当前的知识未来事件表示
        :param kg_map: 知识图
        :param target_len: 每一条标签的长度
        :return: 最终的奖励值,按照seq奖励的高低排序的kg奖励
        """

        # 获取未来的事件表示
        # history_kg = self.mlp_history(batch_kg)

        Reward = []
        dist = []
        dist_replay = []

        for paths in range(path_num):
            # (1,2048,50)
            h = h_origin
            # 保存最终所推荐的那几个商品的id
            indexes = []
            indexes.append(sample1)

            # 和h相配合放入到GRU中进行预测下一个推荐用的
            dec_inp_index = sample1
            # 将id做一个embedding。与知识无关
            dec_inp = self.item_embeddings(dec_inp_index)
            # (2048,1,50)
            dec_inp = dec_inp.unsqueeze(1)

            # 获取真实的知识图的平均池化
            # ground_kg = self.get_kg(items_to_predict, target_len, kg_map)

            for i in range(path_len):
                # 用预测的第一时刻放入gru以预测之后时刻(1,2048,50)
                out_enc, h = self.enc(dec_inp, h, one=True)

                # 获取结合了三种表示的状态表示
                mlp_in = h.squeeze()#torch.cat([h.squeeze(), batch_kg, self.mlp_history(batch_kg)], dim=1)
                # 获取三种状态表示的
                mlp_hidden = self.mlp(mlp_in)
                mlp_hidden = torch.tanh(mlp_hidden)
                out_fc = self.fc(mlp_hidden)

                # out_distribution = self.softmax(out_fc)
                out_distribution = F.softmax(out_fc, dim=1)
                out_distribution =out_distribution* 0.8

                # 加上真实标签辅助训练
                out_distribution = torch.add(out_distribution, pred_ont_hot)

                # 往后预测一个
                m = torch.distributions.categorical.Categorical(out_distribution)
                sample2 = m.sample()

                dec_inp = self.item_embeddings(sample2)
                dec_inp = dec_inp.unsqueeze(1)
                indexes.append(sample2)

            # 此时indexes中存储的是config.T-1个预测的商品id
            indexes = torch.stack(indexes, dim=1)
            # 获取这三个事件对应的当前可能性的知识表示
            # episode_kg = self.get_kg(indexes, torch.Tensor([path_len + 1] * len(indexes)), kg_map)

            # 获取我们预测的事件的平均池化，与ground_truth求一个距离
            # dist.append(self.cos(episode_kg, ground_kg))
            # dist_replay.append(self.cos(episode_kg, history_kg))

            Reward.append(bleu_each(items_to_predict, indexes))

        Reward = torch.FloatTensor(Reward).to(self.device)
        # 将多次的预测取平均
        # dist = torch.stack(dist, dim=0)
        # dist = torch.mean(dist, dim=0)

        # dist_replay = torch.stack(dist_replay, dim=0)
        # dist_sort = self.compare_kgReward(Reward, dist_replay)

        # 多次的平均奖励值
        Reward = torch.mean(Reward,dim=0)
        # Reward = Reward + self.lamda * dist
        # dist_sort = dist_sort.detach()
        return Reward#, dist_sort

    def compare_kgReward(self, reward, dist):
        """
        将dist按照reward的每列最大值的索引进行重新排个序

        :param reward: 奖励值
        :param dist: 每个预测商品和target的余弦相似度
        :return: 排好序的dist
        """
        logit_reward, indice = reward.sort(dim=0)
        dist_sort = dist.gather(dim=0, index=indice)
        return dist_sort