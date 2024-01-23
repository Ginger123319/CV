# -*- encoding: utf-8 -*-
import time
import numpy as np
from sklearn import tree
from sklearn.tree import *
from sklearn.ensemble import *
from dc_model_repo.base import ChartData

support_tree_model = (DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor)
support_trees_model = (RandomForestClassifier, RandomForestRegressor,
                       GradientBoostingClassifier, GradientBoostingRegressor,
                       ExtraTreesClassifier, ExtraTreesRegressor)


def is_tree_model(o):
    from dc_model_repo.util.sklearn_util import get_best_estimator_if_cv
    o = get_best_estimator_if_cv(o)
    return isinstance(o, support_tree_model) | isinstance(o, support_trees_model)


class Node:
    def __init__(self, id, name=None, node_label=None, l_edge='', r_edge=''):
        self.id = id
        self.name = name
        self.node_label = node_label
        self.l_edge = l_edge
        self.r_edge = r_edge


def build_trees_visual_data(trees_estimator, persist_path, feature_names, class_names=None):
    attachments = []
    att_path = '/attachments/'
    import os
    p = os.path.join(persist_path, 'attachments')
    os.makedirs(p, exist_ok=True)

    if isinstance(trees_estimator, support_tree_model):
        tree_att_path = att_path + 'tree_' + str(len(attachments) + 1) + '.txt'
        attachments.append(build_visual_data(trees_estimator, persist_path, tree_att_path, feature_names=feature_names, class_names=class_names))
    elif isinstance(trees_estimator, support_trees_model) & hasattr(trees_estimator, 'estimators_'):
        estimators = trees_estimator.estimators_
        for e in estimators:
            e = e[0] if isinstance(e, np.ndarray) else e
            tree_att_path = att_path + 'tree_' + str(len(attachments) + 1) + '.txt'
            attachments.append(build_visual_data(e, persist_path, tree_att_path, feature_names=feature_names, class_names=class_names))
    if len(attachments) > 0:
        return ChartData('tree', 'tree', data=None, attachments=attachments)
    else:
        return None


def build_visual_data(tree_estimator, persist_tree_path, att_path, feature_names, class_names=None):
    if is_tree_model(tree_estimator):
        start_time = time.time()
        from dc_model_repo.util import validate_util
        if validate_util.is_non_empty_list(class_names):
            class_names = [str(i) for i in class_names]
        graph_dot = tree.export_graphviz(tree_estimator, feature_names=feature_names,
                                         class_names=class_names, filled=True, impurity=True, proportion=True)

        import pydotplus
        graph = pydotplus.graph_from_dot_data(graph_dot)
        # with open(persist_tree_path + att_path + '.dot', 'w') as f:
        #     f.write(graph_dot)
        # graph.write_svg("classifier.svg")

        # 遍历所有边
        edges = {}
        for edge in graph.get_edges():
            s = edge.get_source()
            if s in edges:
                edges[s] = (edges.get(s)[0], edge.get_destination())
            else:
                edges[s] = (edge.get_destination(),)

        # 遍历所有节点
        nodes = []
        for node in graph.get_nodes()[1:]:
            node_id = node.get_name()
            node_label = node.__dict__.get('obj_dict').get('attributes').get('label')
            if node_label is not None and len(node_label) > 0:
                node_label = node_label[1:-1]
            n = Node(node_id)
            # 判断该node是否是叶子节点
            if node_id in edges:  # 不是叶子节点
                tmp_str = node_label.split('\\n')[0].split(' ')
                if '<=' in tmp_str:
                    i = tmp_str.index('<=')
                    tmp_array = [' '.join(tmp_str[:i]), '<=', ' '.join(tmp_str[i + 1:])]
                    tmp_str = tmp_array
                if '=' in tmp_str:
                    i = tmp_str.index('=')
                    tmp_array = [' '.join(tmp_str[:i]), '=', ' '.join(tmp_str[i + 1:])]
                    tmp_str = tmp_array
                n.name = tmp_str[0]
                if tmp_str[1] == '<=':
                    n.l_edge = ''.join(tmp_str[1:])
                    tmp_str[1] = '>'
                    n.r_edge = ''.join(tmp_str[1:])
                elif tmp_str[1] == '=':
                    n.l_edge = ''.join(tmp_str[1:])
                    tmp_str[1] = '!='
                    n.r_edge = ''.join(tmp_str[1:])
                else:
                    n.l_edge = ''.join(tmp_str)
            else: # 是叶子节点
                tmp_str = node_label.split('\\n')[-1].split(' ')
                n.name = tmp_str[-1]
            nodes.append(n)
        with open(persist_tree_path + att_path, 'w') as f:
            for n in nodes:
                l_child_id = ''
                r_child_id = ''
                if n.id in edges:
                    (l_child_id, r_child_id) = edges.get(n.id)
                l = '{}&{}&{}&{}&{}&{}\n'.format(n.id, n.name, l_child_id, r_child_id, n.l_edge, n.r_edge)
                f.write(l)
        end_time = time.time()
        print('build visual data cost time: {}s'.format(round(end_time - start_time, 3)))
        att_path = att_path[1:] if att_path.startswith('/') else att_path
        return {'path': att_path}
    else:
        return None
