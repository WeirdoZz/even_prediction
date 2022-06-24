import numpy as np
import torch


class Sequence:

    def __init__(self,event_sequence):
        chain_ids,event_ids=[],[]
        for chain_id,event_seq in enumerate(event_sequence):
            for event_id in event_seq:
                chain_ids.append(chain_id)
                event_ids.append(event_id)
            # if chain_id==10000:
            #     break

        # 是一个[1,1,1,1,...,5,5,5,5,]这样的列表，其中每个值就是对应的id，出现的次数就是序列的长度
        chain_ids=np.asarray(chain_ids)
        # 是一个[2,7,4,,8,...,12,334,..]的列表，其中每个值是商品id，和上面这个user_id列表是长度相同的
        event_ids=np.asarray(event_ids)

        self.chain_ids=chain_ids
        self.event_ids=event_ids

        self.sequence=None

    def __len__(self):
        return len(self.chain_ids)

    def to_train_sequence(self, sequence_len=9, target_len=2):

        # 先根据用户id进行一个排序,按照升序返回对应的索引
        # 相当于对用户的顺序做了一个限制而已
        sort_indices=np.lexsort((self.chain_ids,))

        chain_ids=self.chain_ids[sort_indices]
        event_ids=self.event_ids[sort_indices]

        # 按照上面的顺序获取所有事件链id，并且获取其第一次出现的索引，和每个事件链出现的次数（也就是事件链的序列长度)
        chain_ids, indices, counts = np.unique(chain_ids,return_index=True,return_counts=True)

        seq_chain = []
        sequences = []
        sequences_length = []
        sequences_targets = []
        sequences_targetlen = []

        for i,chain_id in enumerate(chain_ids):
            start_idx=indices[i]
            try:
                stop_idx=indices[i+1]
            except:
                stop_idx=None

            # 当前事件链的全部事件
            one_sequence=event_ids[start_idx:stop_idx]

            # 防止遇到不同长度的事件链，需要确保最终输入到GRU中的事件链的长度是相同的，需要填充0
            # final_seq=np.pad(one_sequence[-sequence_len:],(0,sequence_len-len(one_sequence[-sequence_len+target_len:])),"constant")

            for train_len in range(len(one_sequence)-1):
                # 获取该序列中的训练子序列
                sub_seq = one_sequence[0:train_len + 1]
                # 需要填充的数量
                num_paddings = sequence_len - train_len - 1
                sub_seq = np.pad(sub_seq, (0, num_paddings), 'constant')

                # 标签子序列
                target_sub = one_sequence[train_len + 1:train_len + 1 + target_len]
                # 获取标签序列填充前的真实长度
                sequences_targetlen.append(len(target_sub))
                target_sub = np.pad(target_sub, (0, target_len - len(target_sub)), 'constant')

                seq_chain.append(chain_id)
                sequences.append(sub_seq)
                sequences_length.append(train_len + 1)
                sequences_targets.append(target_sub)


        sequence_chains=np.array(seq_chain)
        sequences=np.array(sequences)
        sequences_length=np.array(sequences_length)
        sequences_targets=np.array(sequences_targets)
        sequences_targetlen=np.array(sequences_targetlen)

        self.sequence=SequenceInteraction(sequence_chains,sequences,sequences_targets,sequences_length,sequences_targetlen)

    def to_val_sequence(self,sequence_len=9,target_len=2):
        # 先根据用户id进行一个排序,按照升序返回对应的索引
        # 相当于对用户的顺序做了一个限制而已
        sort_indices = np.lexsort((self.chain_ids,))

        chain_ids = self.chain_ids[sort_indices]
        event_ids = self.event_ids[sort_indices]

        # 按照上面的顺序获取所有事件链id，并且获取其第一次出现的索引，和每个事件链出现的次数（也就是事件链的序列长度)
        chain_ids, indices, counts = np.unique(chain_ids, return_index=True, return_counts=True)

        seq_chain = []
        sequences = []
        sequences_length = []
        sequences_targets = []
        sequences_targetlen = []

        for i, chain_id in enumerate(chain_ids):
            start_idx = indices[i]
            try:
                stop_idx = indices[i + 1]
            except:
                stop_idx = None

            # 当前事件链的全部事件
            one_sequence = event_ids[start_idx:stop_idx]

            # 防止遇到不同长度的事件链，需要确保最终输入到GRU中的事件链的长度是相同的，需要填充0
            final_seq=np.pad(one_sequence[-sequence_len:-target_len],(0,sequence_len-target_len-len(one_sequence[-sequence_len+target_len:])),"constant")
            final_tar=one_sequence[-target_len:]

            seq_chain.append(chain_id)
            sequences.append(final_seq)
            sequences_length.append(len(one_sequence)-target_len)
            sequences_targets.append(final_tar)
            sequences_targetlen.append(target_len)

        sequence_chains = np.array(seq_chain)
        sequences = np.array(sequences)
        sequences_length = np.array(sequences_length)
        sequences_targets = np.array(sequences_targets)
        sequences_targetlen = np.array(sequences_targetlen)

        self.sequence = SequenceInteraction(sequence_chains, sequences, sequences_targets, sequences_length,
                                            sequences_targetlen)

class SequenceInteraction:
    def __init__(self,chain_ids,sequences,targets=None,length=None,tar_len=None):
        self.chain_ids = chain_ids
        self.sequences = sequences
        self.targets = targets
        self.length = length
        self.tarlen = tar_len

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]