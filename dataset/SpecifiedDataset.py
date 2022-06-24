from dataset.Dataset import Dataset



class NYT(Dataset):

    def __init__(self):
        self.dir_path="dataset/NYT/"
        self.verb_mapping_file="corpus_verb_mapping.pkl"
        self.train_sequence_file="train_sequence.pkl"
        self.dev_sequence_file = "dev_sequence.pkl"
        self.test_sequence_file = "test_sequence.pkl"
        self.kg_file="event_embedding.npy"

        self.num_events=23474

    def generatet_dataset(self,index_shift=1):
        """
        生成需要的数据集

        :param index_shift: 控制商品id增加多少的
        :return: 训练集（[[...],...]），验证集 ([[...],...])测试集([[...],...])，事件总数(包含0表示的填充事件)，知识图映射([[...],...])
        """

        verb_mapping=self.load_pickle(self.dir_path+self.verb_mapping_file)
        train_sequence=self.load_pickle(self.dir_path+self.train_sequence_file)
        dev_sequence=self.load_pickle(self.dir_path+self.dev_sequence_file)
        test_sequence=self.load_pickle(self.dir_path+self.test_sequence_file)
        kg_mapping=self.load_kg(self.dir_path+self.kg_file)

        assert self.num_events+1==len(kg_mapping)

        train_sequence=self.data_index_shift(train_sequence,index_shift)
        dev_sequence=self.data_index_shift(dev_sequence,index_shift)
        test_sequence=self.data_index_shift(test_sequence,index_shift)

        return train_sequence,dev_sequence,test_sequence,self.num_events+1,kg_mapping

class CompanyEvent(Dataset):

    def __init__(self):
        self.dir_path="dataset/company_event/"
        self.verb_mapping_file="corpus_verb_mapping.pkl"
        self.train_sequence_file="train_sequence.pkl"
        self.dev_sequence_file = "dev_sequence.pkl"
        self.test_sequence_file = "test_sequence.pkl"
        self.kg_file="event_embedding.npy"

        self.num_events=219

    def generatet_dataset(self,index_shift=1):
        """
        生成需要的数据集

        :param index_shift: 控制商品id增加多少的
        :return: 训练集（[[...],...]），验证集 ([[...],...])测试集([[...],...])，事件总数(包含0表示的填充事件)，知识图映射([[...],...])
        """

        verb_mapping=self.load_pickle(self.dir_path+self.verb_mapping_file)
        train_sequence=self.load_pickle(self.dir_path+self.train_sequence_file)
        dev_sequence=self.load_pickle(self.dir_path+self.dev_sequence_file)
        test_sequence=self.load_pickle(self.dir_path+self.test_sequence_file)
        kg_mapping=self.load_kg(self.dir_path+self.kg_file)

        assert self.num_events+1==len(kg_mapping)

        train_sequence=self.data_index_shift(train_sequence,index_shift)
        dev_sequence=self.data_index_shift(dev_sequence,index_shift)
        test_sequence=self.data_index_shift(test_sequence,index_shift)

        return train_sequence,dev_sequence,test_sequence,self.num_events+1,kg_mapping
