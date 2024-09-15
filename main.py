import numpy as np
from rdkit import Chem
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random


# SMILES标准化
def standard(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        std_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return std_smiles
    except:
        return None


def extract_molecules(reaction: str):
    reactants = reaction.split('>')[0]
    mols = [standard(m) for m in reactants.split('.') if standard(m) is not None]
    return mols


def get_max_subgraph(graph):
    if len(graph.nodes) == 0:
        return graph
    # 取图中的最大连通分量
    connected_nodes = nx.connected_components(graph)
    subgraph = nx.subgraph(graph, max(connected_nodes, key=len))
    return subgraph


def attack(graph: nx.Graph, remove_num, intentional=False):
    g = graph.copy()
    nodes = list(graph.nodes)
    if not intentional:
        nodes_to_remove = random.sample(nodes, remove_num)
    else:
        nodes_to_remove = sorted(nodes, key=lambda x: graph.degree[x], reverse=True)[:remove_num]
    g.remove_nodes_from(nodes_to_remove)
    return g


def analyze_graph(graph, k=1, draw=False):
    subgraph = get_max_subgraph(graph)

    if draw:
        # 绘图
        plt.clf()
        nx.draw(graph, node_size=8, alpha=0.7)
        plt.savefig('output/graph.png', dpi=700)

        plt.clf()
        nx.draw(subgraph, node_size=8, alpha=0.7)
        plt.savefig('output/subgraph.png', dpi=700)

        # 节点-度分布图
        plt.clf()
        max_degree = max(dict(nx.degree(graph)).values())
        node_degree_distribution = [i for i in nx.degree_histogram(graph)]
        plt.bar(list(range(max_degree + 1)), node_degree_distribution, width=0.8)
        for x, y in zip(list(range(max_degree + 1)), node_degree_distribution):
            plt.text(x, y, y if y > 0 else '', ha='center', va='bottom', fontsize=5)
        plt.xlabel("$k$")
        plt.ylabel("$p_k$")
        plt.savefig('output/node_degree_distribution.png', dpi=700)

    # 其他统计数据
    print('=' * 60)
    print('Graph:', graph)
    print('Maximal connected subgraph size:', len(subgraph.nodes))
    print('Average shortest path length (maximal connected subgraph):', nx.average_shortest_path_length(subgraph))
    print('Average clustering coefficient:', nx.average_clustering(graph))
    print('K-core graph with k=%d:' % k, nx.k_core(graph, k))  # k-core子图


def analyze_attack(graph, intentional: bool, num=20):
    subgraph_size = []
    path_length = []
    node_num = len(graph.nodes)
    g = graph.copy()
    for i in np.linspace(0, 0.99, num=num):
        g = attack(g, remove_num=len(g.nodes) - int((1 - i) * node_num), intentional=intentional)
        subgraph_size.append(len(get_max_subgraph(g).nodes))
        path_length.append(nx.average_shortest_path_length(get_max_subgraph(g)))
    return subgraph_size, path_length


def main(n, random_seed, sample_num):
    """
    :param n: 读取化学反应数据的条数
    :param random_seed: 抽取数据使用的随机数种子
    :param sample_num: 攻击实验时的采样数
    """
    print('Creating graph...')
    G = nx.Graph()
    df = pd.read_csv('USPTO_MIT.csv', header=0)
    df = df.sample(n, random_state=random_seed)  # 随机选取n个化学反应式

    for _, row in df.iterrows():
        molecules = extract_molecules(row['reactions'])
        G.add_nodes_from(molecules)  # 分子作为节点

        edges = set()
        for i in range(len(molecules)):
            for j in range(i + 1, len(molecules)):
                edges.add((molecules[i], molecules[j]))
        G.add_edges_from(edges)  # 两个可反应分子之间连接一条边
    G.remove_edges_from(nx.selfloop_edges(G))  # 删除自环

    # 分析
    print('Analyzing...')
    analyze_graph(G, k=2, draw=True)

    random_size, random_length = analyze_attack(G, intentional=False, num=sample_num)  # 随机攻击
    intentional_size, intentional_length = analyze_attack(G, intentional=True, num=sample_num)  # 故意攻击
    # 删除比例-子图大小
    plt.clf()
    plt.plot(list(np.linspace(0, 0.99, sample_num)), random_size, label='random')
    plt.plot(list(np.linspace(0, 0.99, sample_num)), intentional_size, label='intentional')
    plt.xlabel('fraction of nodes being removed')
    plt.ylabel('size of subgraph')
    plt.legend()
    plt.savefig('output/attack_graph_size.png', dpi=700)

    # 删除比例-平均路径长度
    plt.clf()
    plt.plot(list(np.linspace(0, 0.99, sample_num)), random_length, label='random')
    plt.plot(list(np.linspace(0, 0.99, sample_num)), intentional_length, label='intentional')
    plt.xlabel('fraction of nodes being removed')
    plt.ylabel('path length of subgraph')
    plt.legend()
    plt.savefig('output/attack_path_length.png', dpi=700)


if __name__ == '__main__':
    main(5000, random_seed=202409, sample_num=100)
