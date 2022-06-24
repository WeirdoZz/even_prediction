import networkx as nx
from pprint import pprint as pt
import matplotlib.pyplot as plt
import re
from random import sample
import random
import pickle

num_chains = 0


class GraphNode(object):
    '''
    树的节点
    '''

    def __init__(self, name, value):
        # 节点的名称
        self.name = name
        # 节点中保存的值
        self.value = value
        # 节点的子节点
        self.children = []


class DAG(object):

    def __init__(self, graph_root_name='root', graph_root_value=None):
        self.graph = GraphNode(graph_root_name, graph_root_value)
        self.if_node_exist = False
        self.search_result_parent = None
        self.search_result_children = []

    def add(self, node, parent_name=None):
        '''
        增加节点
        '''
        if parent_name == None:
            # 如果parent为None，则默认其父节点为root节点
            root_children = self.graph.children
            root_children.append(node)
            self.graph.children = root_children
            # print('Add node:%s sucessfully!' % node.name)
            # print('*' * 30)
        else:
            # 否则检查增加的节点的父节点是否存在
            self.if_node_exist = False
            self.if_node_exist_recursion(
                self.graph, parent_name, search=False)
            if self.if_node_exist:
                # 若父节点存在
                self.add_recursion(parent_name, node, self.graph)
                # print('Add node:%s sucessfully!' % node.name)
                # print('*' * 30)
            # else:
            #     # 若父节点不存在
            #     print("Error: Parent node %s doesn't exist!" % parent_name)
            #     print('*' * 30)

    def search(self, node):
        '''
        检索节点
        打印出其父节点的name以及其下一层所有子节点的name
        '''
        self.if_node_exist = False
        self.if_node_exist_recursion(
            self.graph, node, search=True)
        if self.if_node_exist:
            # 若需要检索的节点存在，返回其父节点以及所有子节点
            print("%s's parent:" % node.name)
            pt(self.search_result_parent)
            print("%s's children:" % node.name)
            pt(self.search_result_children)
            print('*' * 30)
        else:
            # 若检索的节点不存在树中
            print("Error: Node %s doesn't exist!" % node.name)
            print('*' * 30)

    def show_tree(self):
        '''
        利用networkx转换成图结构，方便结合matplotlib将树画出来
        '''
        G = nx.Graph()
        self.to_graph_recursion(self.graph, G)
        nx.draw(G, with_labels=True)
        plt.show()

    def to_graph_recursion(self, graph, G):
        '''
        将节点加入到图中
        '''
        G.add_node(graph.name)
        for child in graph.children:
            G.add_nodes_from([graph.name, child.name])
            G.add_edge(graph.name, child.name)
            self.to_graph_recursion(child, G)

    def if_node_exist_recursion(self, graph, node_name, search=False):
        """
        递归判断节点是否已经在子图中了

        :param graph: 需要判断是否存在node节点的树
        :param node: 需要判断的节点的名称
        :param search: 当检索到该节点时是否返回该节点的父节点和所有子节点
        :return:
        """
        if node_name == self.graph.name:
            self.if_node_exist = True
        if self.if_node_exist:
            return 1
        for child in graph.children:
            if child.name == node_name:
                self.if_node_exist = True
                if search == True:
                    self.search_result_parent = graph.name
                    for cchild in child.children:
                        self.search_result_children.append(cchild.name)
                break
            else:
                self.if_node_exist_recursion(child, node_name, search)

    def add_recursion(self, parent_name, node, tree):
        '''
        增加节点时使用的递归函数
        '''
        # 如果添加的地方就是tree下面的话，就直接添加
        if parent_name == tree.name:
            tree.children.append(node)
            return 1
        # 否则需要找到添加的位置
        for child in tree.children:
            if child.name == parent_name:
                children_list = child.children
                children_list.append(node)
                child.children = children_list
            else:
                self.add_recursion(parent_name, node, child)

    def DFS(self, graph, all_list, cur_list):
        global num_chains
        if len(graph.children) == 0:
            cur_list.append(graph.value[1])
            if len(cur_list) >= 9:
                all_list.append(cur_list.copy())
            num_chains += 1
            # print(f"完成了第{num_chains}个事件链，长度为{len(cur_list)}")
            cur_list.clear()

        else:
            for child in graph.children:
                cur_list.append(child.value[0])
                self.DFS(child, all_list, cur_list)


def preprocess(data_name):
    src_dst_list = []

    with open(data_name) as lines:
        # 获取第一条数据的时间戳
        first_line = next(lines)
        first_line = re.split(r"\s*\t+", first_line.strip())
        last_time = int(first_line[3])

        cur_time_list = []
        for idx, line in enumerate(lines):
            line_data = re.split(r"\s*\t+", line.strip())

            src = int(line_data[0])
            dst = int(line_data[2])
            timestamp = int(line_data[3])

            if timestamp == last_time:
                cur_time_list.append((src, dst))
            else:
                src_dst_list.append(cur_time_list)
                cur_time_list = []
                last_time = timestamp

    # print("按时间划分的src-dst列表构建完毕，第一组为\n", src_dst_list[0])
    return src_dst_list


def construct_DAG(src_dst_list):
    print("正在构建有向无环图")
    dag = DAG()
    # 遍历每一层的全部的二元组列表
    for layer, tuple_list in enumerate(src_dst_list):
        # 遍历当前层的二元组
        for item in tuple_list:
            # 节点的名称为 "层数_下一个时间的id"
            node = GraphNode(str(layer + 1) + "_" + str(item[1]), item)
            # 如果是挂在根节点下面的节点
            if layer == 0:
                dag.add(node)
            else:
                # 否则的话应当挂在上一层对应的节点下面
                dag.add(node, str(layer) + "_" + str(item[0]))
        print(f"第{layer + 1}层的节点添加完毕")
        if layer == 9:
            break
    print("有向无环图构建完毕")
    return dag


if __name__ == "__main__":
    train_txt = "train.txt"
    valid_txt = "valid.txt"
    test_txt = "test.txt"
    dataset_names = ["./GDELT/", "./ICEWS14_forecasting/", "./ICEWS15_forecasting/", "./ICEWS18_forecasting/", "./WIKI/",
                     "./YAGO/"]
    for dataset_name in dataset_names:
        for data in [train_txt, valid_txt, test_txt]:
            print(f"正在处理{dataset_name+data}")
            src_dst_list = preprocess(dataset_name + data)
            dag = construct_DAG(src_dst_list)
            all_list, cur_list = [], []
            dag.DFS(dag.graph, all_list, cur_list)
            # print(f"{dataset_name + data}的事件链已经处理完毕，其前五组值为{all_list[:5]}")

            # 保留上面的list 从中抽取出对应的序列
            sequence_list = []
            for sequence in all_list:  # sample(all_list, len(all_list) // 2)
                cur_sequence = []
                start = random.randint(0, len(sequence)-9)
                for value in sequence[start:start + 9]:
                    cur_sequence.append(value)
                sequence_list.append(cur_sequence)

            with open(dataset_name + data.split(".")[0] + "_sequence.pkl", "wb") as f:
                pickle.dump(sequence_list, f)
            print(sequence_list[:5])
