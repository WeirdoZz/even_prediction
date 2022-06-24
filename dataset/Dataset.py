import pickle
import numpy as np


class Dataset:

    def load_pickle(self, name):
        """
        加载相关数据集的原始文件

        :param name: 数据集的所在位置
        :return: 返回加载后的文件
        """
        with open(name, "rb") as f:
            return pickle.load(f, encoding="latin1")

    def load_kg(self,name):
        """
        加载当前数据集中事件的对应知识表示

        :param name: 知识表示所在的位置
        :return: 返回形状为[[...],...]的矩阵，每行表示对应id的事件的知识表示
        """

        kg=np.load(name)
        """
        后面需要使用0进行事件填充，所以就需要将事件对应的id都加1
        """
        zero_feature=np.zeros((1,50))

        feature_matrix=np.concatenate((zero_feature,kg),axis=0)
        return feature_matrix

    def data_index_shift(self,lists,increase_by=1):
        """
        将lists中每个序列中每一个事件id都加上increase_by

        :param lists: 所有事件序列的信息列表
        :param increase_by: 将事件id增加多少
        :return: 返回增加id后的信息列表
        """

        for seq in lists:
            for i,item_id in enumerate(seq):
                seq[i]=item_id+increase_by

        return lists