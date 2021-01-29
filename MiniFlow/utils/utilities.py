from functools import reduce
import random
from collections import defaultdict
from MiniFlow.nn.core import Placeholder


def topological(graph):
    graph = graph.copy()
    sorted_node = []
    while graph:
        all_nodes_have_inputs = reduce(lambda a, b: a + b, list(graph.values()))
        all_nodes_have_outputs = list(graph.keys())
        all_nodes_only_have_ouputs_no_inputs = set(all_nodes_have_outputs) - set(all_nodes_have_inputs)

        if all_nodes_only_have_ouputs_no_inputs:
            node = random.choice(list(all_nodes_only_have_ouputs_no_inputs))
            sorted_node.append(node)
            if len(graph) == 1:
                sorted_node += graph[node]
            graph.pop(node)

        else:
            raise TypeError("This graph has circle, which cannot get topoligical order!")
    return sorted_node


def convert_feed_dict_to_graph(feed_dict):
    computing_graph = defaultdict(list)
    nodes = [n for n in feed_dict]
    while nodes:
        n = nodes.pop(0)
        if n in computing_graph: continue
        if isinstance(n, Placeholder):
            n.value = feed_dict[n]
        for m in n.outputs:
            nodes.append(m)
            computing_graph[n].append(m)

    return computing_graph


def forward(sorted_nodes):
    for node in sorted_nodes:
        node.forward()


def backward(sorted_nodes):
    for node in sorted_nodes[::-1]:
        node.backward()


def forward_and_backward(sorted_nodes):
    forward(sorted_nodes)
    backward(sorted_nodes)


def topological_sort_feed_dict(feed_dict):
    graph = convert_feed_dict_to_graph(feed_dict)
    return topological(graph)


def optimize(trainables, learning_rate=1e-1):
    for node in trainables:
        node.value += -1 * node.gradients[node] * learning_rate
