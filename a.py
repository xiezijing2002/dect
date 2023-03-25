import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Optional

import dgl
import joblib as J
import networkx as nx       # 画有向图的工具
import torch
from androguard.core.analysis.analysis import MethodAnalysis
from androguard.core.api_specific_resources import load_permission_mappings
from androguard.misc import AnalyzeAPK
from pygtrie import StringTrie
ATTRIBUTES = ['external', 'entrypoint', 'native', 'public', 'static', 'codesize', 'api', 'user']
package_directory=(os.path.dirname(os.path.abspath("__file__")))
#package_directory = os.path.dirname(os.path.abspath(__file__))
def memoize(function):
    """
    Alternative to @lru_cache which could not be pickled in ray
    :param function: Function to be cached
    :return: Wrapped function
    """
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper
class FeatureExtractors:
    NUM_PERMISSION_GROUPS = 20
    NUM_API_PACKAGES = 226    # API包的个数。部分外部节点会属于安卓API包
    NUM_OPCODE_MAPPINGS = 21        # 将原来256种可能的opcode缩减为21组
# 为内部节点（含定义和实现的函数）建立opcode映射表
    @staticmethod
    def _get_opcode_mapping() -> Dict[str, int]:
        """
        Group opcodes and assign them an ID
        :return: Mapping from opcode group name to their ID
        """
        mapping = {x: i for i, x in enumerate(['nop', 'mov', 'return',
                                               'const', 'monitor', 'check-cast', 'instanceof', 'new',
                                               'fill', 'throw', 'goto/switch', 'cmp', 'if', 'unused',
                                               'arrayop', 'instanceop', 'staticop', 'invoke',
                                               'unaryop', 'binop', 'inline'])}
        mapping['invalid'] = -1
        return mapping

    @staticmethod
    @memoize
    def _get_instruction_type(op_value: int) -> str:
        """
        Get instruction group name from instruction
        :param op_value: Opcode value
        :return: String containing ID of :instr:
        """
        if 0x00 == op_value:
            return 'nop'
        elif 0x01 <= op_value <= 0x0D:
            return 'mov'
        elif 0x0E <= op_value <= 0x11:
            return 'return'
        elif 0x12 <= op_value <= 0x1C:
            return 'const'
        elif 0x1D <= op_value <= 0x1E:
            return 'monitor'
        elif 0x1F == op_value:
            return 'check-cast'
        elif 0x20 == op_value:
            return 'instanceof'
        elif 0x22 <= op_value <= 0x23:
            return 'new'
        elif 0x24 <= op_value <= 0x26:
            return 'fill'
        elif 0x27 == op_value:
            return 'throw'
        elif 0x28 <= op_value <= 0x2C:
            return 'goto/switch'
        elif 0x2D <= op_value <= 0x31:
            return 'cmp'
        elif 0x32 <= op_value <= 0x3D:
            return 'if'
        elif (0x3E <= op_value <= 0x43) or (op_value == 0x73) or (0x79 <= op_value <= 0x7A) or (
                0xE3 <= op_value <= 0xED):
            return 'unused'
        elif (0x44 <= op_value <= 0x51) or (op_value == 0x21):
            return 'arrayop'
        elif (0x52 <= op_value <= 0x5F) or (0xF2 <= op_value <= 0xF7):
            return 'instanceop'
        elif 0x60 <= op_value <= 0x6D:
            return 'staticop'
        elif (0x6E <= op_value <= 0x72) or (0x74 <= op_value <= 0x78) or (0xF0 == op_value) or (
                0xF8 <= op_value <= 0xFB):
            return 'invoke'
        elif 0x7B <= op_value <= 0x8F:
            return 'unaryop'
        elif 0x90 <= op_value <= 0xE2:
            return 'binop'
        elif 0xEE == op_value:
            return 'inline'
        else:
            return 'invalid'

# 将opcode的映射表用21位的稀疏比特张量表示
    @staticmethod
    def _mapping_to_bitstring(mapping: List[int], max_len) -> torch.Tensor:
        """
        Convert opcode mappings to bitstring
        :param max_len:
        :param mapping: List of IDs of opcode groups (present in an method)
        :return: Binary tensor of length `len(opcode_mapping)` with value 1 at positions specified by :poram mapping:
        """
        size = torch.Size([1, max_len])
        if len(mapping) > 0:
            indices = torch.LongTensor([[0, x] for x in mapping]).t()
            values = torch.LongTensor([1] * len(mapping))
            tensor = torch.sparse.LongTensor(indices, values, size)
        else:
            tensor = torch.sparse.LongTensor(size)
        # Sparse tensor is normal tensor on CPU!
        return tensor.to_dense().squeeze()

    @staticmethod
    def _get_api_trie() -> StringTrie:
        apis = open(Path(package_directory).parent / "metadata" / "api.list").readlines()
        api_list = {x.strip(): i for i, x in enumerate(apis)}
        api_trie = StringTrie(separator='.')
        for k, v in api_list.items():
            api_trie[k] = v
        return api_trie

    @staticmethod
    @memoize
    # 返回15比特的内部节点（方法）特征向量
    def get_api_features(api: MethodAnalysis) -> Optional[torch.Tensor]:
        if not isinstance(api, ExternalMethod):  # 修改此处
            return None
        api_trie = FeatureExtractors._get_api_trie()
        name = str(api.class_name)[1:-1].replace('/', '.')
        _, index = api_trie.longest_prefix(name)
        if index is None:
            indices = []
        else:
            indices = [index]
        feature_vector = FeatureExtractors._mapping_to_bitstring(indices, FeatureExtractors.NUM_API_PACKAGES)
        return feature_vector

    @staticmethod
    @memoize
    # 返回外部节点的独热特征向量
    def get_user_features(user: MethodAnalysis) -> Optional[torch.Tensor]:
        if user.is_external():
            return None
        opcode_mapping = FeatureExtractors._get_opcode_mapping()
        opcode_groups = set()
        for instr in user.get_method().get_instructions():
            instruction_type = FeatureExtractors._get_instruction_type(instr.get_op_value())
            instruction_id = opcode_mapping[instruction_type]
            if instruction_id >= 0:
                opcode_groups.add(instruction_id)
        # 1 subtraction for 'invalid' opcode group
        feature_vector = FeatureExtractors._mapping_to_bitstring(list(opcode_groups), len(opcode_mapping) - 1)
        return torch.LongTensor(feature_vector)
import networkx as nx
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import ExternalMethod

# 覆盖 get_call_graph 函数
def patched_get_call_graph(analysis_object):
    def _add_node(G, method, _entry_points):
        if method not in G.nodes():
            if isinstance(method, ExternalMethod):
                is_external = True
            else:
                is_external = False

            G.add_node(method, external=is_external, entry_points=_entry_points)

    CG = nx.DiGraph()
    entry_points = set()

    for m in analysis_object.get_methods():
        orig_method = m.get_method()
        if m.is_external():
            continue

        if not m.get_xref_to():
            entry_points.add(orig_method)
            _add_node(CG, orig_method, True)

        for other_class, callee, offset in m.get_xref_to():
            _add_node(CG, callee, False)
            CG.add_edge(orig_method, callee, offset=offset)

    return CG
source_file = "D:\APK\APK\d87c43bc79ccb02a1ff3ca40ce6acbac.apk"
file_name = "D:\APK\FCG\d87c43bc79ccb02a1ff3ca40ce6acbac.fcg"
_, _, dx = AnalyzeAPK(source_file)  # androguard提取APK的格式
cg = patched_get_call_graph(dx)

# print(cg.nodes,cg.nodes())

# print(type(cg.nodes),type(cg.nodes())) 都是<class 'networkx.classes.reportviews.NodeView'>
mappings = {}
# 为FCG中的每个节点赋予特征向量（外部：API张量，内部：opcode张量）

# 为FCG中的每个节点赋予特征向量（外部：API张量，内部：opcode张量）
for node in cg.nodes():
    features = {
        "api": torch.zeros(FeatureExtractors.NUM_API_PACKAGES),
        "user": torch.zeros(FeatureExtractors.NUM_OPCODE_MAPPINGS)
    }
    if isinstance(node, ExternalMethod):  # 修改此处
        features["api"] = FeatureExtractors.get_api_features(node)
    else:
        features["user"] = FeatureExtractors.get_user_features(node)
    mappings[node] = features

# print(mappings)
nx.set_node_attributes(cg, mappings)
cg = nx.convert_node_labels_to_integers(cg)
print('cg的类型：',type(cg))
dg = dgl.from_networkx(cg, node_attrs=ATTRIBUTES)
print('dg的类型：',type(dg))
dgl.data.utils.save_graphs(file_name, [dg])
print(f"Processed {source_file}")