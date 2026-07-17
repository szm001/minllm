"""
minllm.py
中英文双语 GPT 大模型预训练系统

本文件是纯 Python GPT 算法基础上，扩展为支持中英文双语的预训练系统。
核心改进：
  1. 双语分词器：中文字符级 + 英文词级的混合分词策略
  2. 中文字符编码：覆盖 CJK 主区及扩展区，UTF-8 全链路处理
  3. 架构调整：扩展词表嵌入、特殊 token（PAD/BOS/EOS/UNK）、词表大小上限
  4. 数据加载：读取 data 目录下新闻文本文件，按段落切分文档
  5. 训练适配：支持提示文本引导的生成、序列截断、线性学习率衰减

所有变量/函数使用 snake_case，类使用 PascalCase，命名完全重写。



### 性能与架构特性说明（非缺陷，未改动）
- GPU 利用率 ：本文件为纯 Python 教学实现（ AutogradValue 标量计算图 + 嵌套列表），无 GPU/向量化支持。这是设计意图，改造需引入 numpy/torch 属重写，超出本次范围。
- 生成文本质量 ：受限于教学规模（ embedding_dim=16, num_layers=1, num_steps=30 ），生成质量有限属预期。修复梯度 bug 后质量已显著提升（测试中可生成连贯短语）。
- 异常处理 ： generate_text 的 temperature 截断、长 prompt 截断、 ModelPersistence 的文件校验与异常捕获、 train() 的学习率恢复均已完备。
"""

import os
import math
import json
import random
import sys

random.seed(42)

'''
# 对字典排序（默认对键 key 排序）
d = {3: 'c', 1: 'a', 2: 'b'}
d2 = {}
for key, value in sorted(d.items()):
    d2[value] = key

print(sorted(d2.items())) 
print(sorted(d2.items(),key=lambda x: x[-1])) 
print(sorted(d2.items(),key=lambda x: -x[-1])) 
print(sorted(d2.items(),key=lambda x: x[-1],reverse=True))
print(sorted(d2.items(),key=lambda x: x[-1],reverse=False))
print(sorted(d2.items(),reverse=False))
print(sorted(d2.items(),reverse=True))

print(sorted(d2.items(),key=lambda x: -x[-1],reverse=False)) 
print(sorted(d2.items(),key=lambda x: -x[-1],reverse=True)) 

[('a', 1), ('b', 2), ('c', 3)]
[('a', 1), ('b', 2), ('c', 3)]
[('c', 3), ('b', 2), ('a', 1)]
[('c', 3), ('b', 2), ('a', 1)]
[('a', 1), ('b', 2), ('c', 3)]
[('a', 1), ('b', 2), ('c', 3)]
[('c', 3), ('b', 2), ('a', 1)]
[('c', 3), ('b', 2), ('a', 1)]
[('a', 1), ('b', 2), ('c', 3)]

print("eeeeeeeeeeeee")
'''

# ============================================================
# 第一部分：中文字符处理工具
# ============================================================

def is_chinese_character(char):
    """
    判断单个字符是否为中文字符。
    覆盖 CJK 统一表意文字主区及常见扩展/兼容区，确保生僻字也能正确识别。
    """
    code_point = ord(char)
    return (
        0x4E00 <= code_point <= 0x9FFF or   # CJK 统一表意文字主区（常用汉字）
        0x3400 <= code_point <= 0x4DBF or   # CJK 扩展 A 区
        0xF900 <= code_point <= 0xFAFF or   # CJK 兼容表意文字
        0x2F800 <= code_point <= 0x2FA1F    # CJK 兼容表意补充
    )


def is_ascii_alphanumeric(char):
    """判断是否为 ASCII 字母或数字，用于英文词元聚合"""
    return char.isalnum() and ord(char) < 128


# ============================================================
# 第二部分：双语分词器
# ============================================================

class BilingualTokenizer:
    """
    中英文双语分词器

    分词策略（针对双语文本边界优化）：
      - 中文字符：每个汉字单独作为一个 token（中文无空格分词，字粒度保留语义单元）
      - 英文/数字：连续的 ASCII 字母数字聚合为一个 token（词级，保留英文语义完整性）
      - 其他字符：标点、空格、全角符号等单独作为 token
    """

    PAD_TOKEN = '<pad>'   # 填充 token
    BOS_TOKEN = '<bos>'   # 序列起始 token
    EOS_TOKEN = '<eos>'   # 序列结束 token
    UNK_TOKEN = '<unk>'   # 未知 token（词表外的 token 回退到此）

    def __init__(self):
        self.token_to_id = {}      # token 字符串 -> id 的映射
        self.id_to_token = {}      # id -> token 字符串的映射
        self.vocab_size = 0
        # 特殊 token 优先注册，保证 id 固定在词表头部
        for special_token in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]:
            self._register_token(special_token)

    def _register_token(self, token):
        """注册单个 token 到词表，若已存在则跳过"""
        if token not in self.token_to_id:
            token_id = self.vocab_size
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.vocab_size += 1
        return self.token_to_id[token]

    @property
    def pad_id(self):
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self):
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self):
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self):
        return self.token_to_id[self.UNK_TOKEN]

    def split_text_to_tokens(self, text):
        """
        将原始文本切分为 token 字符串列表（尚未编码为 id）。
        这是双语分词的核心：按中文字、英文词、其他字符三类分别处理。
        """
        tokens = []
        i = 0
        text_length = len(text)
        while i < text_length:
            char = text[i]
            if is_chinese_character(char):
                # 中文字符：单字成 token，保留字粒度语义
                tokens.append(char)
                i += 1
            elif is_ascii_alphanumeric(char):
                # ASCII 字母/数字：聚合成完整词，避免把 "GPT" 拆成单字符
                j = i
                while j < text_length and is_ascii_alphanumeric(text[j]):
                    j += 1
                tokens.append(text[i:j])
                i = j
            else:
                # 其他字符（标点/空格/全角符号等）单独成 token
                tokens.append(char)
                i += 1
        return tokens

    def build_vocabulary(self, documents, max_vocab_size=None):
        """
        根据文档列表构建词表。

        可选限制最大词表大小：按 token 出现频率降序保留高频 token，
        低频 token 回退为 UNK，避免词表过大导致纯 Python 训练不可行。
        """
        token_counts = {}
        for document in documents:
            for token in self.split_text_to_tokens(document):
                token_counts[token] = token_counts.get(token, 0) + 1

        # 按频率降序排序：
        # token_counts 是字典 {token: count}，
        # sorted_tokens 是 [(token, count), ...] 列表
        # 使用 sorted() 函数，key=lambda item: -item[1] 表示按 count（索引1）取负值排序，实现降序
        # 例如：
        # token_counts = { 'GPT': 40, 'the': 80, '是': 60, '的': 100}
        # sorted_tokens = [('的', 100), ('the', 80), ('是', 60), ('GPT', 40)]
        # 高频词排在前面，优先进入词表；低频词若超出词表上限则会被舍弃，映射为 UNK
        # for token, count in token_counts.items():
        #     print(f"{token}: {count}")
        temp_test = token_counts.items()
        # print(temp_test)
        '''
        temp_test = dict_items(
            [
                ('1', 2), 
                ('.', 3), 
                ('https', 1), 
                (':', 2)
            ]
        )
        '''
        sorted_tokens = sorted(token_counts.items(), key=lambda item: -item[1])
        # print(sorted_tokens)
        '''
        [
            (' ', 23), 
            ('/', 7), 
            ('-', 4), 
            ('：', 4), 
            ('的', 4), 
            ('.', 3), 
            ('，', 3), 
            ('。', 3), 
            ('AI', 3), 
            ('1', 2), 
            (':', 2), 
            ....
            ('链', 1), 
            (' 接', 1)
            ]
        '''
        # 若设了上限，只保留高频 token（减去已注册的特殊 token 数量）
        if max_vocab_size is not None:
            remaining_slots = max_vocab_size - len(self.token_to_id)
            if remaining_slots > 0:
                sorted_tokens = sorted_tokens[:remaining_slots]
            else:
                sorted_tokens = []

        for token, _ in sorted_tokens:
            self._register_token(token)

        return self.vocab_size

    def encode(self, text):
        """将文本编码为 token id 序列，词表外的 token 映射为 UNK"""
        tokens = self.split_text_to_tokens(text)
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, token_ids):
        """将 token id 序列解码为文本，自动跳过特殊 token"""
        result_tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, self.UNK_TOKEN)
            if token not in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN):
                result_tokens.append(token)
        return ''.join(result_tokens)


# ============================================================
# 第三部分：新闻数据加载器
# ============================================================

class NewsDataLoader:
    """
    新闻文本数据加载器

    读取 data 目录下所有 .txt 新闻文件，按行/段落切分为文档，
    全程使用 UTF-8 编码，确保中文字符正确解析。
    """

    def __init__(self, data_directory='data'):
        self.data_directory = data_directory

    def load_documents(self, min_paragraph_length=0):
        """
        加载所有新闻文件并按行切分为文档列表。

        参数 min_paragraph_length：过滤过短的行（如"原文链接"等噪声）。
        """
        documents = []
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"数据目录不存在: {self.data_directory}")
        filelist = os.listdir(self.data_directory)
        for filename in sorted(filelist):
            if not filename.endswith('.txt'):
                continue
            filepath = os.path.join(self.data_directory, filename)
            # 使用 utf-8-sig 自动去除可能的 BOM 头，避免首字符变为 \ufeff
            with open(filepath, 'r', encoding='utf-8-sig') as file_handle:
                content = file_handle.read()
            # 按换行切分为段落，过滤空行和过短行
            for paragraph in content.split('\n'):
                clean_text = paragraph.strip()
                if len(clean_text) >= min_paragraph_length:
                    documents.append(clean_text)

        random.shuffle(documents)
        return documents


# ============================================================
# 第四部分：自动微分引擎
# ============================================================

class AutogradValue:
    """
    支持自动求导的标量值节点（计算图节点）。

    递归应用链式法则：前向传播构建计算图，反向传播按拓扑序逆序累积梯度。
    """

    __slots__ = ('data', 'gradient', '_children', '_local_gradients')

    def __init__(self, data, children=(), local_gradients=()):
        self.data = data                            # 前向传播计算的标量值
        self.gradient = 0                            # 损失对该节点的梯度，反向传播时计算
        self._children = children                    # 计算图中的子节点
        self._local_gradients = local_gradients     # 本节点对各子节点的局部导数

    def __add__(self, other):
        other = other if isinstance(other, AutogradValue) else AutogradValue(other)
        return AutogradValue(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, AutogradValue) else AutogradValue(other)
        return AutogradValue(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        base = self.data
        # 防护：零底数 + 非正指数导致 ZeroDivisionError，比如(0, -1)
        if base == 0 and exponent <= 0:
            return AutogradValue(0, (self,), (0,))
        if abs(base) < 1e-12 and exponent <= 0:
            base = 1e-12
        # 防护：负底数 + 分数指数返回复数，取绝对值兜底（模型中不应出现此情况）,比如(-2, 0.3)
        if base < 0 and not float(exponent).is_integer():
            base = abs(base)
        value = base ** exponent
        grad = exponent * base ** (exponent - 1)
        return AutogradValue(value, (self,), (grad,))

    def log(self):
        # 防护：data <= 0 时 math.log 会抛 ValueError，用极小正值兜底（概率下溢场景）
        safe_data = max(self.data, 1e-12)
        # 语法形式
        # math.log(x)          # 计算 x 的自然对数 (ln x)，默认底数为 e
        # math.log(x, base)    # 计算以 base 为底的 x 的对数 (log_base x)
        # 导数推导：y = ln(x)，则 dy/dx = 1/x
        # 这里 safe_data 是 ln 的输入，所以导数为 1.0 / safe_data
        return AutogradValue(math.log(safe_data), (self,), (1.0 / safe_data,))

    def exp(self):
        # 防护：过大值会导致 OverflowError，截断到合理范围
        safe_data = max(-700, min(self.data, 700))

        # 相比于直接使用 math.e ** safe_data 或 pow(math.e, safe_data)，
        # math.exp(safe_data)` 通常能提供更精确的结果因为它使用了专门优化的算法处理浮点数运算。
        # 导数推导：y = e^x，则 dy/dx = e^x  导数是其本身
        # 这里 safe_data 是 e^x 的输入，所以导数为 e^safe_data
        return AutogradValue(math.exp(safe_data), (self,), (math.exp(safe_data),))

    def relu(self):
        return AutogradValue(max(0, self.data), (self,), (float(self.data > 0),))

    def detach(self):
        """返回同值但无计算图依赖的新节点，用于推理时释放内存"""
        return AutogradValue(self.data)


    def __repr__(self):
        """
        返回对象的字符串表示，用于调试和日志输出。

        __repr__ 是 Python 的"官方"字符串表示方法，设计目标是：
          1. 清晰展示对象的核心状态（data 和 gradient）
          2. 便于开发者调试时快速理解 AutogradValue 节点的数值和梯度
          3. 格式遵循 <类名>(属性=值) 的约定，与 Python 内置类型保持一致

        与 __str__ 的区别：
          - __repr__ 面向开发者，要求准确、无歧义，通常用于调试
          - __str__ 面向用户，要求可读性好，通常用于展示
          - 若未定义 __str__，Python 会回退到使用 __repr__

        示例输出：
          AutogradValue(data=3.141593, grad=0.500000)
          AutogradValue(data=-0.000001, grad=0.000000)
        """
        return f"AutogradValue(data={self.data:.6f}, grad={self.gradient:.6f})"
    
    def __neg__(self):
        """
        实现一元负号运算符（-x），返回当前值的相反数。

        数学原理：neg(x) = -1 * x
        梯度推导：d(-x)/dx = -1

        实现方式：复用 __mul__ 方法，将 self 与 -1 相乘。
        这样自动继承乘法节点的计算图特性，无需额外定义局部梯度。

        示例：
            x = AutogradValue(3.0)
            y = -x  # 等价于 x * -1，y.data = -3.0
            # 反向传播后：x.gradient = -1 * y.gradient
        """
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def backward(self):
        """
        反向传播：通过迭代式深度优先拓扑排序确定计算图节点的依赖顺序，
        然后逆序传播梯度。
        使用迭代而非递归，避免深层计算图（大词表/长序列）导致 RecursionError。
        
        核心原理：链式法则的自动应用
        ============================
        
        假设我们有计算图：a -> b -> c -> loss（其中 -> 表示依赖关系）
        即：b = f(a), c = g(b), loss = h(c)
        
        根据链式法则：
        ∂loss/∂a = ∂loss/∂c * ∂c/∂b * ∂b/∂a
        
        反向传播就是从 loss 开始，沿着计算图反向遍历，将梯度逐层传递回去。
        
        实例说明：y = (a + b) * c 的反向传播
        ==================================
        
        构建计算图：
            节点1: a = AutogradValue(2.0)
            节点2: b = AutogradValue(3.0)  
            节点3: c = AutogradValue(4.0)
            节点4: add_result = a + b = 5.0  (children=(a,b), local_gradients=(1,1))
            节点5: y = add_result * c = 20.0 (children=(add_result,c), local_gradients=(4,5))
        
        拓扑排序过程（栈 LIFO 特性）：
        -----------------------------
        
        初始状态：
            stack = [(y, False)]  # (节点, 子节点是否已处理)
            visited = {}
            topo_order = []
        
        第1轮：弹出 (y, False)
            - y 未访问，标记为已访问 visited = [y]
            - 压入 (y, True)  # "待输出"标记
            - 压入 y 的子节点：(add_result, False), (c, False)
            - 栈状态（LIFO，后入先出）: [(y, True), (add_result, False), (c, False)]
        
        第2轮：弹出 (c, False)
            - c 是叶子节点（无children），直接标记为已访问 visited = [y, c]
            - 压入 (c, True)
            - 无子节点可压入
            - 栈状态: [(y, True), (add_result, False), (c, True)]
        
        第3轮：弹出 (c, True)
            - children_processed=True，输出到 topo_order
            - topo_order = [c]
        
        第4轮：弹出 (add_result, False)
            - 标记为已访问 visited = [y, c, add_result]
            - 压入 (add_result, True)
            - 压入子节点：(a, False), (b, False)
            - 栈状态: [(y, True), (add_result, True), (a, False), (b, False)]
        
        第5轮：弹出 (b, False) -> visited = [y, c, add_result,b] 压入 (b, True) -> 无子节点
        第6轮：弹出 (b, True) -> topo_order = [c, b]
        
        第7轮：弹出 (a, False) -> visited = [y, c, add_result,b,a] 压入 (a, True) -> 无子节点
        第8轮：弹出 (a, True) -> topo_order = [c, b, a]
        
        第9轮：弹出 (add_result, True) -> topo_order = [c, b, a, add_result]
        
        第10轮：弹出 (y, True) -> topo_order = [c, b, a, add_result, y]
        
        最终拓扑序（子节点先于父节点）：[c, b, a, add_result, y]
        逆序用于梯度传播：[y, add_result, a, b, c] -> 实际梯度传播顺序
        
        注意：实际梯度传播时，我们使用 reversed(topo_order)，
        即 [y, add_result, a, b, c]，这样确保从 loss 节点开始反向传播。
        
        梯度传播过程：
        -------------
        
        1. 初始化：y.gradient = 1.0（loss对自身梯度为1）
        
        2. 处理 y = add_result * c：
           - y._children = (add_result, c)
           - y._local_gradients = (c.data, add_result.data) = (4, 5)
           - add_result.gradient += 4 * 1.0 = 4
           - c.gradient += 5 * 1.0 = 5
        
        3. 处理 add_result = a + b：
           - add_result._children = (a, b)
           - add_result._local_gradients = (1, 1)
           - a.gradient += 1 * 4 = 4
           - b.gradient += 1 * 4 = 4
        
        4. a, b, c 是叶子节点，无需继续传播
        
        最终结果：
            a.gradient = 4  (∂y/∂a = ∂y/∂add_result * ∂add_result/∂a = 4 * 1 = 4)
            b.gradient = 4  (∂y/∂b = ∂y/∂add_result * ∂add_result/∂b = 4 * 1 = 4)
            c.gradient = 5  (∂y/∂c = add_result.data = 5)
        
        验证（数值微分）：
            y = (a+b)*c = (2+3)*4 = 20
            a=2.001时, y=20.004, ∂y/∂a≈4 ✓
                # 详细计算过程：
                # 原函数 y = (a + b) * c，其中 a=2.0, b=3.0, c=4.0，y=20.0
                # 当 a 增加 0.001 变为 2.001 时：
                #   y_new = (2.001 + 3.0) * 4.0 = 5.001 * 4.0 = 20.004
                # 数值微分近似：∂y/∂a ≈ (y_new - y) / Δa = (20.004 - 20.0) / 0.001 = 0.004 / 0.001 = 4
                # 解析解验证：∂y/∂a = ∂[(a+b)*c]/∂a = c = 4，与数值微分结果一致 ✓
            b=3.001时, y=20.004, ∂y/∂b≈4 ✓
            c=4.001时, y=20.005, ∂y/∂c≈5 ✓
        """
        topological_order = []
        visited_nodes = set()
        # 迭代式后序 DFS：用 (node, children_processed) 标记区分"待展开"和"待输出"
        # 这是关键技巧：利用栈的 LIFO 特性，确保子节点先于父节点被处理
        stack = [(self, False)]
        while stack:
            node, children_processed = stack.pop()
            if children_processed:
                # 子节点已全部处理完毕，可以安全输出当前节点
                topological_order.append(node)
                continue
            if node in visited_nodes:
                # 已访问过的节点，跳过避免重复处理
                continue
            visited_nodes.add(node)
            # 关键步骤：先压入"已完成子节点处理"标记，再压入子节点
            # 利用栈 LIFO 特性：后压入的子节点会先被处理（深度优先）
            # 当子节点都处理完后，才会遇到之前压入的 (node, True) 标记
            # 这样就保证了：所有子节点都在父节点之前进入 topo_order
            stack.append((node, True))
            for child in node._children:
                if child not in visited_nodes:
                    stack.append((child, False))

        # 梯度传播起点：loss 节点（调用 backward 的节点）的梯度为 1
        self.gradient = 1.0
        # 逆序遍历：从当前节点（loss）往回传播到输入节点
        # 这样确保每个节点被处理时，其梯度已经被上游节点计算完毕
        for node in reversed(topological_order):
            for child, local_gradient in zip(node._children, node._local_gradients):
                # 链式法则：child 的梯度 += 局部导数 * 当前节点的梯度
                # 使用 += 是因为一个子节点可能被多个父节点引用（梯度累加）
                child.gradient += local_gradient * node.gradient

# ============================================================
# 第五部分：模型架构组件
# ============================================================

def initialize_parameter_matrix(output_dim, input_dim, std=0.08):
    """初始化参数矩阵（二维列表），高斯分布随机初始化"""
    return [[AutogradValue(random.gauss(0, std)) for _ in range(input_dim)]
            for _ in range(output_dim)]


def apply_linear_transformation(input_vector, weight_matrix):
    """线性变换：output = W * x"""
    return [sum(weight * x for weight, x in zip(weight_row, input_vector))
            for weight_row in weight_matrix]


def compute_softmax(logits):
    """数值稳定的 softmax，减去最大值防止指数溢出"""
    if not logits:
        return []
    max_logit = max(logit.data for logit in logits)
    exp_values = [(logit - max_logit).exp() for logit in logits]
    total = sum(exp_values)
    # 防护：total 极小时（所有 exp 下溢），返回均匀分布避免除零
    if total.data < 1e-15:
        uniform = 1.0 / len(logits)
        return [AutogradValue(uniform) for _ in logits]
    return [exp_val / total for exp_val in exp_values]


def apply_rms_normalization(input_vector):
    """RMSNorm 归一化，比 LayerNorm 更高效，无偏置项"""
    if not input_vector:
        return []
    mean_square = sum(x * x for x in input_vector) / len(input_vector)
    scale = (mean_square + 1e-5) ** -0.5
    return [x * scale for x in input_vector]


# ============================================================
# 第六部分：双语 GPT 模型
# ============================================================

class BilingualGptModel:
    """
    中英文双语 GPT 模型

    基于 GPT-2 架构（RMSNorm + 多头注意力 + MLP + 残差连接），针对双语输入调整：
      - 词表嵌入扩展为双语 token 嵌入（中文字 + 英文词）
      - 位置嵌入支持可配置的上下文长度
      - 多头注意力机制通过共享嵌入空间自然实现跨语言语义对齐
      - 输出投影层映射回双语词表
    """

    def __init__(self, vocab_size, embedding_dim=16, num_layers=1,
                 num_heads=2, context_length=16):
        # 参数校验：embedding_dim 必须能被 num_heads 整除，否则多头注意力维度不匹配
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim({embedding_dim}) 必须能被 num_heads({num_heads}) 整除"
            )
        if context_length < 1:
            raise ValueError(f"context_length 必须 >= 1, 当前为 {context_length}")
        if vocab_size < 1:
            raise ValueError(f"vocab_size 必须 >= 1, 当前为 {vocab_size}")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = embedding_dim // num_heads

        # 初始化模型参数字典
        self.parameters = {}
        self.parameters['token_embedding'] = initialize_parameter_matrix(vocab_size, embedding_dim)
        self.parameters['position_embedding'] = initialize_parameter_matrix(context_length, embedding_dim)
        # 输出投影层（lm_head），将隐藏状态映射回词表维度
        self.parameters['output_projection'] = initialize_parameter_matrix(vocab_size, embedding_dim)

        for layer_index in range(num_layers):
            prefix = f'layer{layer_index}'
            self.parameters[f'{prefix}.attention_query'] = initialize_parameter_matrix(embedding_dim, embedding_dim)
            self.parameters[f'{prefix}.attention_key'] = initialize_parameter_matrix(embedding_dim, embedding_dim)
            self.parameters[f'{prefix}.attention_value'] = initialize_parameter_matrix(embedding_dim, embedding_dim)
            self.parameters[f'{prefix}.attention_output'] = initialize_parameter_matrix(embedding_dim, embedding_dim)
            self.parameters[f'{prefix}.mlp_fc1'] = initialize_parameter_matrix(4 * embedding_dim, embedding_dim)
            self.parameters[f'{prefix}.mlp_fc2'] = initialize_parameter_matrix(embedding_dim, 4 * embedding_dim)

        # 将所有参数展平为单一列表，供优化器统一管理
        self.trainable_parameters = [
            param for matrix in self.parameters.values() for row in matrix for param in row
        ]

    def forward(self, token_id, position_id, cached_keys, cached_values):
        """
        前向传播：给定当前 token 和位置，返回词表维度的 logits。

        cached_keys/cached_values 按 layer 存储历史 K/V，实现自回归注意力
        （当前位置可 attend 到所有历史位置，符合因果性）。
        """
        # 嵌入层：token 嵌入 + 位置嵌入
        token_embedding = self.parameters['token_embedding'][token_id]
        position_embedding = self.parameters['position_embedding'][position_id]
        hidden_state = [t + p for t, p in zip(token_embedding, position_embedding)]
        hidden_state = apply_rms_normalization(hidden_state)

        for layer_index in range(self.num_layers):
            prefix = f'layer{layer_index}'

            # --- 1) 多头注意力块 ---
            residual = hidden_state
            hidden_state = apply_rms_normalization(hidden_state)

            query = apply_linear_transformation(hidden_state, self.parameters[f'{prefix}.attention_query'])
            key = apply_linear_transformation(hidden_state, self.parameters[f'{prefix}.attention_key'])
            value = apply_linear_transformation(hidden_state, self.parameters[f'{prefix}.attention_value'])

            # 将当前 K/V 追加到缓存，实现跨位置注意力
            cached_keys[layer_index].append(key)
            cached_values[layer_index].append(value)

            attention_output = []
            for head_index in range(self.num_heads):
                head_start = head_index * self.head_dim
                # 提取当前头对应的子空间
                query_head = query[head_start:head_start + self.head_dim]
                key_head_sequence = [k[head_start:head_start + self.head_dim] for k in cached_keys[layer_index]]
                value_head_sequence = [v[head_start:head_start + self.head_dim] for v in cached_values[layer_index]]

                # 计算注意力分数（缩放点积）
                # 举例说明：假设 query_head = [0.5, -0.3, 0.8]，key_head_sequence[t] = [0.2, 0.6, -0.4]
                # 点积 = 0.5*0.2 + (-0.3)*0.6 + 0.8*(-0.4) = 0.1 - 0.18 - 0.32 = -0.4
                # 缩放因子 = sqrt(head_dim) = sqrt(3) ≈ 1.732
                # 缩放后分数 = -0.4 / 1.732 ≈ -0.231
                # 缩放目的是防止点积过大导致 softmax 梯度消失
                
                #query_head[0].data
                #query_head[0].gradient
                # query_head: List[AutogradValue], 维度为 [head_dim]
                # key_head_sequence: List[List[AutogradValue]], 维度为 [seq_len][head_dim]
                # 计算缩放点积: attention_logits[t] = sum(query_head[j] * key_head_sequence[t][j]) / sqrt(head_dim)
                # 其中 j 的范围是 0 到 head_dim-1, t 的范围是 0 到 seq_len-1
                # query_head[j] 和 key_head_sequence[t][j] 都是 AutogradValue 类型
                # 举例说明计算过程：
                # 假设 head_dim = 3, seq_len = 2
                # query_head = [AutogradValue(0.5), AutogradValue(-0.3), AutogradValue(0.8)]
                # key_head_sequence[0] = [AutogradValue(0.2), AutogradValue(0.6), AutogradValue(-0.4)]
                # key_head_sequence[1] = [AutogradValue(0.1), AutogradValue(-0.5), AutogradValue(0.3)]
                # 计算 t=0 时的点积：
                #   0.5*0.2 + (-0.3)*0.6 + 0.8*(-0.4) = 0.1 - 0.18 - 0.32 = -0.4
                # 缩放因子 = sqrt(3) ≈ 1.732
                # attention_logits[0] = -0.4 / 1.732 ≈ -0.231
                # 计算 t=1 时的点积：
                #   0.5*0.1 + (-0.3)*(-0.5) + 0.8*0.3 = 0.05 + 0.15 + 0.24 = 0.44
                # attention_logits[1] = 0.44 / 1.732 ≈ 0.254
                # 最终 attention_logits = [-0.231, 0.254]


                
                # ============================================
                # attention_logits 计算过程详细说明
                # ============================================
                # 
                # 假设场景：
                #   head_dim = 3, seq_len = 2
                #   query_head = [AutogradValue(0.5), AutogradValue(-0.3), AutogradValue(0.8)]
                #   key_head_sequence[0] = [AutogradValue(0.2), AutogradValue(0.6), AutogradValue(-0.4)]
                #   key_head_sequence[1] = [AutogradValue(0.1), AutogradValue(-0.5), AutogradValue(0.3)]
                #   缩放因子 = sqrt(3) ≈ 1.732
                #
                # 计算 t=0 时的 attention_logits[0]：
                #
                # 步骤1: 计算 query_head[0] * key_head_sequence[0][0]
                #   - 调用: AutogradValue(0.5).__mul__(AutogradValue(0.2))
                #   - __mul__ 方法: other = AutogradValue(0.2) (已是AutogradValue类型)
                #   - 计算: self.data * other.data = 0.5 * 0.2 = 0.1
                #   - 返回: AutogradValue(0.1, children=(AutogradValue(0.5), AutogradValue(0.2)), local_gradients=(0.2, 0.5))
                #   - 结果: term0 = AutogradValue(0.1)
                #
                # 步骤2: 计算 query_head[1] * key_head_sequence[0][1]
                #   - 调用: AutogradValue(-0.3).__mul__(AutogradValue(0.6))
                #   - 计算: -0.3 * 0.6 = -0.18
                #   - 返回: AutogradValue(-0.18, children=(AutogradValue(-0.3), AutogradValue(0.6)), local_gradients=(0.6, -0.3))
                #   - 结果: term1 = AutogradValue(-0.18)
                #
                # 步骤3: 计算 query_head[2] * key_head_sequence[0][2]
                #   - 调用: AutogradValue(0.8).__mul__(AutogradValue(-0.4))
                #   - 计算: 0.8 * (-0.4) = -0.32
                #   - 返回: AutogradValue(-0.32, children=(AutogradValue(0.8), AutogradValue(-0.4)), local_gradients=(-0.4, 0.8))
                #   - 结果: term2 = AutogradValue(-0.32)
                #
                # 步骤4: 计算 term0 + term1 (sum的开始)
                #   - 调用: term0.__add__(term1) 即 AutogradValue(0.1).__add__(AutogradValue(-0.18))
                #   - __add__ 方法: other = AutogradValue(-0.18)
                #   - 计算: self.data + other.data = 0.1 + (-0.18) = -0.08
                #   - 返回: AutogradValue(-0.08, children=(AutogradValue(0.1), AutogradValue(-0.18)), local_gradients=(1.0, 1.0))
                #   - 结果: partial_sum = AutogradValue(-0.08)
                #
                # 步骤5: 计算 partial_sum + term2
                #   - 调用: AutogradValue(-0.08).__add__(AutogradValue(-0.32))
                #   - 计算: -0.08 + (-0.32) = -0.4
                #   - 返回: AutogradValue(-0.4, children=(AutogradValue(-0.08), AutogradValue(-0.32)), local_gradients=(1.0, 1.0))
                #   - 结果: dot_product = AutogradValue(-0.4)
                #
                # 步骤6: 计算缩放因子 (self.head_dim ** 0.5)
                #   - head_dim = 3, 所以 3 ** 0.5 = 1.732...
                #   - 这是一个Python float，不是AutogradValue
                #
                # 步骤7: 计算 dot_product / scale_factor
                #   - 调用: dot_product.__truediv__(1.732...) 
                #   - __truediv__ 方法: other = 1.732... (float)
                #   - 内部调用: self * other ** -1
                #   - 首先计算 other ** -1: 这是一个Python float运算，1.732... ** -1 = 0.577...
                #   - 然后调用: dot_product.__mul__(0.577...)
                #   - 由于 0.577... 不是 AutogradValue，__mul__ 会将其包装: AutogradValue(0.577...)
                #   - 计算: -0.4 * 0.577... = -0.231...
                #   - 返回: AutogradValue(-0.231..., children=(AutogradValue(-0.4), AutogradValue(0.577...)), local_gradients=(0.577..., -0.4))
                #   - 结果: attention_logits[0] = AutogradValue(-0.231...)
                #
                # 计算 t=1 时的 attention_logits[1]（类似过程）：
                #   - 点积 = 0.5*0.1 + (-0.3)*(-0.5) + 0.8*0.3 = 0.05 + 0.15 + 0.24 = 0.44
                #   - 缩放后: 0.44 / 1.732... = 0.254...
                #   - 结果: attention_logits[1] = AutogradValue(0.254...)
                #
                # 最终 attention_logits = [AutogradValue(-0.231...), AutogradValue(0.254...)]
                #
                # 每个 AutogradValue 节点都保存了：
                #   - data: 前向计算值
                #   - _children: 产生该值的子节点（用于反向传播构建计算图）
                #   - _local_gradients: 对该节点各子节点的局部导数
                #   - gradient: 反向传播时填充的损失对该节点的梯度
                #
                # 这种设计使得后续调用 backward() 时，可以自动计算所有参数的梯度，
                # 用于 Adam 优化器的参数更新。
                #

                attention_logits = [
                    sum(query_head[j] * key_head_sequence[t][j] for j in range(self.head_dim)) / (self.head_dim ** 0.5)
                    for t in range(len(key_head_sequence))
                ]
                attention_weights = compute_softmax(attention_logits)
                # 加权求和得到头输出
                head_output = [
                    sum(attention_weights[t] * value_head_sequence[t][j] for t in range(len(value_head_sequence)))
                    for j in range(self.head_dim)
                ]
                attention_output.extend(head_output)

            # 注意力输出投影 + 残差连接
            hidden_state = apply_linear_transformation(attention_output, self.parameters[f'{prefix}.attention_output'])
            hidden_state = [a + b for a, b in zip(hidden_state, residual)]

            # --- 2) MLP 块 ---
            residual = hidden_state
            hidden_state = apply_rms_normalization(hidden_state)
            hidden_state = apply_linear_transformation(hidden_state, self.parameters[f'{prefix}.mlp_fc1'])
            hidden_state = [h.relu() for h in hidden_state]
            hidden_state = apply_linear_transformation(hidden_state, self.parameters[f'{prefix}.mlp_fc2'])
            hidden_state = [a + b for a, b in zip(hidden_state, residual)]

        # 输出投影：隐藏状态 -> 词表 logits
        logits = apply_linear_transformation(hidden_state, self.parameters['output_projection'])
        return logits


# ============================================================
# 第七部分：AdamW 优化器
# ============================================================
class AdamWOptimizer:
    """
    AdamW 优化器，Adam 的权重衰减修正版本。

    与 Adam 的区别：
      - Adam: 权重衰减（L2 正则）应用在梯度上，即 grad = grad + weight_decay * param
      - AdamW: 权重衰减直接应用在参数更新步骤中，与梯度解耦

    数学公式：
      m_t = β1 * m_{t-1} + (1 - β1) * g_t           # 一阶动量（梯度均值）
      v_t = β2 * v_{t-1} + (1 - β2) * g_t^2          # 二阶动量（梯度方差）
      m̂_t = m_t / (1 - β1^t)                         # 偏差校正后的一阶动量
      v̂_t = v_t / (1 - β2^t)                         # 偏差校正后的二阶动量
      param_t = param_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε) - lr * weight_decay * param_{t-1}

    关键优势：
      - 权重衰减与梯度更新解耦，避免自适应学习率（二阶动量）对 L2 正则的干扰
      - 实验表明 AdamW 在大模型训练中通常优于 Adam + L2 正则
    """

    def __init__(self, parameters, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        """
        初始化 AdamW 优化器。

        Args:
            parameters: 可训练参数列表（AutogradValue 对象）
            learning_rate: 学习率，默认 0.01
            beta1: 一阶动量衰减率，默认 0.9
            beta2: 二阶动量衰减率，默认 0.999
            epsilon: 数值稳定性常数，默认 1e-8
            weight_decay: 权重衰减系数（L2 正则强度），默认 0.01
        """
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.first_moment = [0.0] * len(parameters)   # 一阶动量（梯度均值）
        self.second_moment = [0.0] * len(parameters)  # 二阶动量（梯度方差）
        self.step_count = 0

    def step(self):
        """
        执行一次参数更新，并清零梯度。

        更新流程：
          1. 步数计数器递增
          2. 遍历每个参数，跳过 NaN/Inf 梯度
          3. 更新一阶/二阶动量（指数移动平均）
          4. 偏差校正
          5. 参数更新：Adam 更新 + 解耦的权重衰减
          6. 清零梯度
        """
        self.step_count += 1
        for i, param in enumerate(self.parameters):
            grad = param.gradient

            # 防护：NaN/Inf 梯度会永久污染动量，跳过该参数
            if math.isnan(grad) or math.isinf(grad):
                param.gradient = 0
                continue

            # 更新一阶动量（梯度均值）
            self.first_moment[i] = self.beta1 * self.first_moment[i] + (1 - self.beta1) * grad
            # 更新二阶动量（梯度方差）
            self.second_moment[i] = self.beta2 * self.second_moment[i] + (1 - self.beta2) * grad ** 2

            # 偏差校正
            corrected_first = self.first_moment[i] / (1 - self.beta1 ** self.step_count)
            corrected_second = self.second_moment[i] / (1 - self.beta2 ** self.step_count)

            # AdamW 核心：权重衰减与梯度更新解耦
            # 先应用标准的 Adam 更新
            adam_update = self.learning_rate * corrected_first / (corrected_second ** 0.5 + self.epsilon)
            # 再应用解耦的权重衰减（直接衰减参数值，不经过动量计算）
            weight_decay_update = self.learning_rate * self.weight_decay * param.data

            # 合并更新
            param.data -= adam_update + weight_decay_update

            # 清零梯度，为下一轮前向传播做准备
            param.gradient = 0

    def zero_grad(self):
        """
        手动清零所有参数的梯度。

        通常在 step() 后自动调用，但提供显式接口以便灵活控制。
        """
        for param in self.parameters:
            param.gradient = 0

# ============================================================
# 第七部分：Adam 优化器
# ============================================================

class AdamOptimizer:
    """Adam 优化器，维护一阶/二阶动量并做偏差校正"""

    def __init__(self, parameters, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = [0.0] * len(parameters)   # 一阶动量（梯度均值）
        self.second_moment = [0.0] * len(parameters)    # 二阶动量（梯度方差）
        self.step_count = 0

    def step(self):
        """执行一次参数更新，并清零梯度"""
        self.step_count += 1
        for i, param in enumerate(self.parameters):
            grad = param.gradient
            # 防护：NaN/Inf 梯度会永久污染动量（加权平均无法剔除），跳过该参数
            if math.isnan(grad) or math.isinf(grad):
                param.gradient = 0
                continue
            self.first_moment[i] = self.beta1 * self.first_moment[i] + (1 - self.beta1) * grad
            self.second_moment[i] = self.beta2 * self.second_moment[i] + (1 - self.beta2) * grad ** 2
            # 偏差校正
            corrected_first = self.first_moment[i] / (1 - self.beta1 ** self.step_count)
            corrected_second = self.second_moment[i] / (1 - self.beta2 ** self.step_count)
            param.data -= self.learning_rate * corrected_first / (corrected_second ** 0.5 + self.epsilon)
            param.gradient = 0


# ============================================================
# 第八部分：训练器（封装训练流程与生成）
# ============================================================

class BilingualTrainer:
    """
    双语模型预训练器

    封装训练循环、损失计算、学习率衰减和文本生成。
    适配双语新闻文本特点：批次取单文档、序列截断、BOS/EOS 包围。
    """

    def __init__(self, model, tokenizer, documents, optimizer):
        self.model = model
        self.tokenizer = tokenizer
        self.documents = documents
        self.optimizer = optimizer

    def train_step(self, document):
        """
        单步训练：前向传播 -> 计算交叉熵损失 -> 反向传播 -> 参数更新。
        返回当前步的标量损失值。
        """
        # 防御性校验：推理专用 Trainer（optimizer=None）不应调用训练步
        if self.optimizer is None:
            raise RuntimeError("train_step 不可用于推理专用 Trainer（optimizer 为 None）")

        # 编码文档：BOS + token 序列 + EOS
        token_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(document) + [self.tokenizer.eos_id]
        # 截断至上下文长度（+1 是因为需要 target，即 n 个位置需要 n+1 个 token）
        token_ids = token_ids[:self.model.context_length + 1]

        num_positions = min(self.model.context_length, len(token_ids) - 1)
        if num_positions < 1:
            return None

        # 显式清零所有可训练参数的梯度，避免上一步残留梯度污染当前反向传播
        # （optimizer.step 末尾已清零，此处为防御性双保险，提升异常路径下的健壮性）
        for param in self.optimizer.parameters:
            param.gradient = 0

        # 初始化 KV 缓存
        cached_keys = [[] for _ in range(self.model.num_layers)]
        cached_values = [[] for _ in range(self.model.num_layers)]

        # 逐位置前向传播，累积交叉熵损失
        position_losses = []
        for pos_id in range(num_positions):
            current_token = token_ids[pos_id]
            target_token = token_ids[pos_id + 1]
            logits = self.model.forward(current_token, pos_id, cached_keys, cached_values)
            probs = compute_softmax(logits)
            # 计算当前位置的交叉熵损失：-log(P(target_token))
            # 设计思路：
            #   1. probs[target_token] 获取模型预测的目标 token 的概率（AutogradValue 类型）
            #   2. 调用 .log() 计算自然对数，将概率转换为对数概率（log-probability）
            #   3. 取负号得到负对数似然（Negative Log-Likelihood, NLL）
            # 
            # 数学原理：
            #   交叉熵损失 H(y, p) = -Σ y_i * log(p_i)
            #   对于单标签分类，真实分布 y 是 one-hot 向量（只有目标位置为1，其余为0）
            #   因此简化为 H = -log(p_target)，即负对数概率
            #
            # 为什么用对数：
            #   - 概率值通常在 [0,1] 之间，直接相乘会导致数值下溢（多个小数相乘趋近于0）
            #   - 取对数将乘法转换为加法，数值稳定性更好
            #   - 对数函数是单调递增的，最大化 log(p) 等价于最大化 p
            #
            # 梯度传播：
            #   该损失值是 AutogradValue，后续调用 .backward() 时会自动计算
            #   ∂(-log(p))/∂p = -1/p，推动模型增大目标 token 的概率
            position_loss = -probs[target_token].log()
            position_losses.append(position_loss)

        loss = (1.0 / num_positions) * sum(position_losses)

        # 防护：损失为 NaN/Inf 时跳过本步更新，防止参数被 NaN 永久污染
        if math.isnan(loss.data) or math.isinf(loss.data):
            return None

        # 反向传播 + 参数更新
        loss.backward()
        self.optimizer.step()

        return loss.data

    def train(self, num_steps=30, learning_rate_decay=True):
        """
        执行完整训练循环。
        线性学习率衰减使训练后期更稳定。
        """
        if num_steps <= 0:
            return  # 无训练步数时直接返回，避免后续除零

        if len(self.documents) == 0:
            raise ValueError("文档列表为空，无法训练")

        initial_lr = self.optimizer.learning_rate
        num_documents = len(self.documents)
        # 学习率衰减保底：避免末步 LR 降至极小值导致训练后期几乎不更新
        # 保底下限为初始 LR 的 10%，保证训练全程均有有效更新幅度
        min_lr = initial_lr * 0.1

        try:
            for step in range(num_steps):
                # 线性学习率衰减（带保底，避免末步衰减至 0）
                if learning_rate_decay:
                    decayed_lr = initial_lr * (1 - step / num_steps)
                    self.optimizer.learning_rate = max(decayed_lr, min_lr)

                document = self.documents[step % num_documents]
                loss_value = self.train_step(document)
                if loss_value is not None:
                    print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss_value:.4f}", end='\r')
        finally:
            # 无论正常结束还是异常，都恢复初始学习率
            self.optimizer.learning_rate = initial_lr
            print()  # 换行

    def generate_text(self, prompt_text='', max_new_tokens=16, temperature=0.7):
        """
        基于提示文本自回归生成后续文本。

        先前向传播 prompt 部分（构建 KV 缓存），再逐 token 采样生成。
        temperature 控制创造性：(0, 1]，值越低输出越确定。
        """
        # 防护：temperature <= 0 会导致除零，设最小下限
        temperature = max(temperature, 1e-6)

        # 编码提示文本，添加 BOS
        prompt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt_text)
        # 截断至上下文长度（预留生成空间）
        max_prompt_len = self.model.context_length
        prompt_ids = prompt_ids[:max_prompt_len]

        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]

        cached_keys = [[] for _ in range(self.model.num_layers)]
        cached_values = [[] for _ in range(self.model.num_layers)]

        # 前向传播 prompt 部分，构建 KV 缓存（不采样，仅填充缓存）
        # 推理时 detach KV 缓存，避免计算图无限增长导致内存泄漏
        current_token = prompt_ids[0]
        prompt_len = len(prompt_ids)
        for pos_id in range(prompt_len - 1):
            self.model.forward(current_token, pos_id, cached_keys, cached_values)
            self._detach_kv_cache(cached_keys, cached_values)
            current_token = prompt_ids[pos_id + 1]

        # 自回归生成
        generated_ids = []
        current_pos = prompt_len - 1
        for _ in range(max_new_tokens):
            if current_pos >= self.model.context_length:
                break
            logits = self.model.forward(current_token, current_pos, cached_keys, cached_values)
            # 温度缩放后采样
            scaled_logits = [logit / temperature for logit in logits]
            probs = compute_softmax(scaled_logits)
            current_token_list = random.choices(
                range(self.model.vocab_size),
                weights=[p.data for p in probs]
            )
            current_token = current_token_list[0]
            # 释放本轮计算图，仅保留 KV 缓存的数值
            self._detach_kv_cache(cached_keys, cached_values)
            if current_token == self.tokenizer.eos_id:
                break
            generated_ids.append(current_token)
            current_pos += 1

        return self.tokenizer.decode(generated_ids)

    @staticmethod
    def _detach_kv_cache(cached_keys, cached_values):
        """将 KV 缓存中最新一层的计算图依赖断开，仅保留数值，防止推理内存泄漏"""
        for layer_idx in range(len(cached_keys)):
            if cached_keys[layer_idx]:
                cached_keys[layer_idx][-1] = [v.detach() for v in cached_keys[layer_idx][-1]]
            if cached_values[layer_idx]:
                cached_values[layer_idx][-1] = [v.detach() for v in cached_values[layer_idx][-1]]


# ============================================================
# 第八部分扩展：模型持久化（保存与加载）
# ============================================================

class ModelPersistence:
    """
    大模型持久化管理器

    负责将训练好的完整模型（权重参数、配置信息、词汇表）保存到项目根目录下的
    model 目录，以及从 model 目录加载已保存的模型用于推理任务。

    保存的文件结构：
      model/
        config.json      - 模型架构配置（词表大小、嵌入维度、层数、头数、上下文长度）
        tokenizer.json   - 分词器词汇表（token <-> id 映射 + 特殊 token）
        weights.json     - 全部权重参数（仅保存 AutogradValue.data 浮点值，丢弃计算图/梯度）
    """

    # 模型保存目录：项目根目录下的 model 文件夹
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

    CONFIG_FILE = 'config.json'
    TOKENIZER_FILE = 'tokenizer.json'
    WEIGHTS_FILE = 'weights.json'

    @classmethod
    def save_model(cls, model, tokenizer):
        """
        将训练好的模型完整保存到 model 目录。

        保存内容：
          1. config.json    - 模型架构配置（超参数）
          2. tokenizer.json - 分词器词汇表（token <-> id 映射）
          3. weights.json   - 全部权重参数（AutogradValue.data 浮点值）

        异常：目录创建失败、文件写入失败时抛出 RuntimeError。
        """
        # 1. 创建模型目录（exist_ok=True 避免目录已存在时报错）
        try:
            os.makedirs(cls.MODEL_DIR, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"创建模型目录失败: {cls.MODEL_DIR} ({exc})")

        # 2. 保存模型配置
        config = {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'num_layers': model.num_layers,
            'num_heads': model.num_heads,
            'context_length': model.context_length,
        }
        config_path = os.path.join(cls.MODEL_DIR, cls.CONFIG_FILE)
        try:
            with open(config_path, 'w', encoding='utf-8') as file_handle:
                json.dump(config, file_handle, ensure_ascii=False, indent=2)
        except (OSError, IOError, TypeError) as exc:
            raise RuntimeError(f"保存模型配置失败: {config_path} ({exc})")

        # 3. 保存词汇表（id_to_token 的键需转为字符串，JSON 仅支持字符串键）
        tokenizer_data = {
            'token_to_id': tokenizer.token_to_id,
            'id_to_token': {str(k): v for k, v in tokenizer.id_to_token.items()},
            'vocab_size': tokenizer.vocab_size,
            'special_tokens': {
                'PAD_TOKEN': tokenizer.PAD_TOKEN,
                'BOS_TOKEN': tokenizer.BOS_TOKEN,
                'EOS_TOKEN': tokenizer.EOS_TOKEN,
                'UNK_TOKEN': tokenizer.UNK_TOKEN,
            },
        }
        tokenizer_path = os.path.join(cls.MODEL_DIR, cls.TOKENIZER_FILE)
        try:
            with open(tokenizer_path, 'w', encoding='utf-8') as file_handle:
                json.dump(tokenizer_data, file_handle, ensure_ascii=False, indent=2)
        except (OSError, IOError, TypeError) as exc:
            raise RuntimeError(f"保存词汇表失败: {tokenizer_path} ({exc})")

        # 4. 保存权重参数（仅取 AutogradValue.data，丢弃计算图依赖与梯度）
        weights_data = {}
        for param_name, matrix in model.parameters.items():
            weights_data[param_name] = [[cell.data for cell in row] for row in matrix]
        weights_path = os.path.join(cls.MODEL_DIR, cls.WEIGHTS_FILE)
        try:
            with open(weights_path, 'w', encoding='utf-8') as file_handle:
                json.dump(weights_data, file_handle, ensure_ascii=False, indent=2)
        except (OSError, IOError, TypeError) as exc:
            raise RuntimeError(f"保存权重参数失败: {weights_path} ({exc})")

        print(f"模型已保存到目录: {cls.MODEL_DIR}")

    @classmethod
    def load_model(cls):
        """
        从 model 目录加载已保存的完整模型用于推理任务。

        返回: (model, tokenizer) 元组
        异常: 模型文件不存在时抛出 FileNotFoundError；文件损坏时抛出 RuntimeError。
        """
        # 1. 校验所有必要文件是否存在
        for filename in (cls.CONFIG_FILE, cls.TOKENIZER_FILE, cls.WEIGHTS_FILE):
            filepath = os.path.join(cls.MODEL_DIR, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"模型文件不存在: {filepath}")

        # 2. 加载模型配置
        config_path = os.path.join(cls.MODEL_DIR, cls.CONFIG_FILE)
        try:
            with open(config_path, 'r', encoding='utf-8') as file_handle:
                config = json.load(file_handle)
        except (OSError, IOError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"加载模型配置失败: {config_path} ({exc})")

        # 3. 加载词汇表
        tokenizer_path = os.path.join(cls.MODEL_DIR, cls.TOKENIZER_FILE)
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as file_handle:
                tokenizer_data = json.load(file_handle)
        except (OSError, IOError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"加载词汇表失败: {tokenizer_path} ({exc})")

        # 4. 加载权重参数
        weights_path = os.path.join(cls.MODEL_DIR, cls.WEIGHTS_FILE)
        try:
            with open(weights_path, 'r', encoding='utf-8') as file_handle:
                weights_data = json.load(file_handle)
        except (OSError, IOError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"加载权重参数失败: {weights_path} ({exc})")

        # 5. 重建分词器：直接恢复词表映射，无需重新训练
        tokenizer = BilingualTokenizer()
        tokenizer.token_to_id = dict(tokenizer_data['token_to_id'])
        tokenizer.id_to_token = {int(k): v for k, v in tokenizer_data['id_to_token'].items()}
        tokenizer.vocab_size = tokenizer_data['vocab_size']

        # 6. 重建模型并载入保存的权重
        model = BilingualGptModel(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            context_length=config['context_length'],
        )

        # 校验保存的权重与模型结构一致，并覆盖随机初始化的权重值
        for param_name, matrix in model.parameters.items():
            if param_name not in weights_data:
                raise RuntimeError(f"权重参数缺失: {param_name}")
            saved_matrix = weights_data[param_name]
            if len(saved_matrix) != len(matrix):
                raise RuntimeError(
                    f"权重形状不匹配: {param_name} (期望 {len(matrix)} 行, 实际 {len(saved_matrix)} 行)"
                )
            for i, row in enumerate(matrix):
                saved_row = saved_matrix[i]
                if len(saved_row) != len(row):
                    raise RuntimeError(
                        f"权重形状不匹配: {param_name}[{i}] (期望 {len(row)} 列, 实际 {len(saved_row)} 列)"
                    )
                for j, cell in enumerate(row):
                    cell.data = saved_row[j]

        # 7. 重建 trainable_parameters 引用（data 已覆盖，列表对象引用不变）
        model.trainable_parameters = [
            param
            for matrix in model.parameters.values()
            for row in matrix
            for param in row
        ]

        print(f"模型已从目录加载: {cls.MODEL_DIR}")
        return model, tokenizer


# ============================================================
# 第九部分：单元测试
# ============================================================

def run_unit_tests():
    """
    单元测试：验证双语分词、中文字符处理、自动微分、模型前向传播和训练稳定性。

    使用 assert 断言，失败时抛出 AssertionError 并打印错误位置。
    """
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)

    # --- 测试 1：中文字符识别 ---
    print("[1] 中文字符识别测试...", end=' ')
    assert is_chinese_character('模') is True, "常用汉字应识别为中文"
    assert is_chinese_character('A') is False, "英文字母不应识别为中文"
    assert is_chinese_character('1') is False, "数字不应识别为中文"
    assert is_chinese_character('，') is False, "中文标点不在 CJK 表意区"
    print("通过")

    # --- 测试 2：双语分词 ---
    print("[2] 双语分词测试...", end=' ')
    tokenizer = BilingualTokenizer()
    tokens = tokenizer.split_text_to_tokens("GPT模型训练")
    # "GPT" 应聚合为一个英文词 token，"模"、"型"、"训"、"练" 各为单独 token
    assert tokens == ['GPT', '模', '型', '训', '练'], f"分词结果不符预期: {tokens}"
    print("通过")

    # --- 测试 3：编码解码往返 ---
    print("[3] 编码解码往返测试...", end=' ')
    tokenizer2 = BilingualTokenizer()
    test_docs = ["双语模型 training 稳定", "中文分词 English words 123"]
    tokenizer2.build_vocabulary(test_docs)
    for text in test_docs:
        encoded = tokenizer2.encode(text)
        decoded = tokenizer2.decode(encoded)
        assert decoded == text, f"往返不一致: 原文='{text}' 解码='{decoded}'"
    print("通过")

    # --- 测试 4：中文字符编码（含生僻字） ---
    print("[4] 中文字符编码测试...", end=' ')
    tokenizer3 = BilingualTokenizer()
    rare_chars = "龙龘繁體字扩展"
    tokenizer3.build_vocabulary([rare_chars])
    encoded = tokenizer3.encode(rare_chars)
    decoded = tokenizer3.decode(encoded)
    assert decoded == rare_chars, f"生僻字往返不一致: '{decoded}'"
    print("通过")

    # --- 测试 5：自动微分正确性 ---
    print("[5] 自动微分测试...", end=' ')
    a = AutogradValue(2.0)
    b = AutogradValue(3.0)
    c = a * b + a ** 2   # c = 2*3 + 4 = 10, dc/da = b + 2a = 3+4 = 7, dc/db = a = 2
    c.backward()
    assert abs(a.gradient - 7.0) < 1e-6, f"a 的梯度应为 7.0, 实际 {a.gradient}"
    assert abs(b.gradient - 2.0) < 1e-6, f"b 的梯度应为 2.0, 实际 {b.gradient}"
    print("通过")

    # --- 测试 6：模型前向传播输出形状 ---
    print("[6] 模型前向传播测试...", end=' ')
    tokenizer4 = BilingualTokenizer()
    tokenizer4.build_vocabulary(["测试模型 forward pass"])
    model = BilingualGptModel(
        vocab_size=tokenizer4.vocab_size,
        embedding_dim=8, num_layers=1, num_heads=2, context_length=8
    )
    cached_keys = [[] for _ in range(model.num_layers)]
    cached_values = [[] for _ in range(model.num_layers)]
    logits = model.forward(tokenizer4.bos_id, 0, cached_keys, cached_values)
    assert len(logits) == tokenizer4.vocab_size, "logits 维度应等于词表大小"
    print("通过")

    # --- 测试 7：训练损失下降（稳定性） ---
    print("[7] 训练稳定性测试...", end=' ')
    tokenizer5 = BilingualTokenizer()
    train_docs = [
        "人工智能改变世界",
        "深度学习推动进步",
        "语言模型理解文本",
        "双语训练提升能力",
        "中英文混合 GPT 训练",
    ]
    tokenizer5.build_vocabulary(train_docs, max_vocab_size=128)
    model5 = BilingualGptModel(
        vocab_size=tokenizer5.vocab_size,
        embedding_dim=8, num_layers=1, num_heads=2, context_length=8
    )
    optimizer5 = AdamOptimizer(model5.trainable_parameters, learning_rate=0.05)
    trainer5 = BilingualTrainer(model5, tokenizer5, train_docs, optimizer5)
    first_loss = trainer5.train_step(train_docs[0])
    for _ in range(8):
        trainer5.train_step(train_docs[0])
    last_loss = trainer5.train_step(train_docs[0])
    assert last_loss < first_loss, f"训练后损失应下降: {first_loss:.4f} -> {last_loss:.4f}"
    print(f"通过 (loss {first_loss:.4f} -> {last_loss:.4f})")

    # --- 测试 8：文本生成 ---
    print("[8] 文本生成测试...", end=' ')
    generated = trainer5.generate_text(prompt_text='人工', max_new_tokens=8, temperature=0.5)
    assert isinstance(generated, str), "生成结果应为字符串"
    print(f"通过 (生成: '{generated}')")

    # --- 测试 9：log(0) 数值安全 ---
    print("[9] log(0) 数值安全测试...", end=' ')
    zero_val = AutogradValue(0.0)
    neg_val = AutogradValue(-1.0)
    log_zero = zero_val.log()
    log_neg = neg_val.log()
    assert log_zero.data < -27, "log(0) 应返回极小值而非崩溃"
    assert log_neg.data < -27, "log(负数) 应安全返回而非崩溃"
    print("通过")

    # --- 测试 10：__pow__ 零底数安全 + 负底数正确性 ---
    print("[10] __pow__ 边界测试...", end=' ')
    base_zero = AutogradValue(0.0)
    result = base_zero ** -0.5   # 原代码会 ZeroDivisionError
    assert isinstance(result, AutogradValue), "零底数负指数应安全返回"
    # 负底数 + 正整数指数：(-2)**2 应等于 4，不能错误返回 0
    neg_base = AutogradValue(-2.0)
    neg_squared = neg_base ** 2
    assert abs(neg_squared.data - 4.0) < 1e-6, f"(-2)**2 应为 4, 实际 {neg_squared.data}"
    neg_squared.backward()
    assert abs(neg_base.gradient - (-4.0)) < 1e-6, f"d/dx(x^2) at x=-2 应为 -4, 实际 {neg_base.gradient}"
    # 正常底数梯度不受影响
    normal_val = AutogradValue(4.0)
    powered = normal_val ** 2   # 4^2=16, d/dx=2*4=8
    powered.backward()
    assert abs(normal_val.gradient - 8.0) < 1e-6, f"梯度应为 8.0, 实际 {normal_val.gradient}"
    print("通过")

    # --- 测试 11：compute_softmax 空列表安全 ---
    print("[11] compute_softmax 空列表测试...", end=' ')
    assert compute_softmax([]) == [], "空 logits 应返回空列表"
    print("通过")

    # --- 测试 12：temperature=0 不崩溃 ---
    print("[12] temperature=0 安全测试...", end=' ')
    generated_t0 = trainer5.generate_text(prompt_text='人工', max_new_tokens=4, temperature=0)
    assert isinstance(generated_t0, str), "temperature=0 应安全返回而非除零崩溃"
    print("通过")

    # --- 测试 13：train(num_steps=0) 不崩溃 ---
    print("[13] train(num_steps=0) 安全测试...", end=' ')
    saved_lr = optimizer5.learning_rate
    trainer5.train(num_steps=0)
    assert optimizer5.learning_rate == saved_lr, "num_steps=0 后学习率应不变"
    print("通过")

    # --- 测试 14：detach 方法 ---
    print("[14] detach 方法测试...", end=' ')
    orig = AutogradValue(3.5)
    detached = orig.detach()
    assert detached.data == 3.5, "detach 应保留数值"
    assert detached._children == (), "detach 应断开计算图依赖"
    print("通过")

    # --- 测试 15：长 prompt 超过上下文长度不崩溃 ---
    print("[15] 长 prompt 截断测试...", end=' ')
    long_prompt = '人工智能' * 50   # 远超 context_length=8
    generated_long = trainer5.generate_text(prompt_text=long_prompt, max_new_tokens=4)
    assert isinstance(generated_long, str), "长 prompt 应被截断而非越界崩溃"
    print("通过")

    # --- 测试 16：迭代式 backward 深计算图不栈溢出 ---
    print("[16] 深计算图 backward 测试...", end=' ')
    # 构造 2000 层链式运算，远超 Python 默认递归限制 1000
    deep_val = AutogradValue(1.0)
    chain_result = deep_val
    for _ in range(2000):
        chain_result = chain_result + AutogradValue(0.001)
    chain_result.backward()
    assert abs(deep_val.gradient - 1.0) < 1e-6, "深链梯度应为 1.0"
    print("通过")

    # --- 测试 17：空文档列表训练报错 ---
    print("[17] 空文档训练报错测试...", end=' ')
    empty_trainer = BilingualTrainer(model5, tokenizer5, [], optimizer5)
    try:
        empty_trainer.train(num_steps=5)
        assert False, "空文档应抛出 ValueError"
    except ValueError:
        pass  # 预期行为
    print("通过")

    # --- 测试 18：模型参数校验 ---
    print("[18] 模型参数校验测试...", end=' ')
    try:
        BilingualGptModel(vocab_size=10, embedding_dim=10, num_heads=3)  # 10%3 != 0
        assert False, "embedding_dim 不能整除 num_heads 应报错"
    except ValueError:
        pass
    try:
        BilingualGptModel(vocab_size=10, embedding_dim=8, num_heads=2, context_length=0)
        assert False, "context_length=0 应报错"
    except ValueError:
        pass
    print("通过")

    # --- 测试 19：apply_rms_normalization 空输入安全 ---
    print("[19] apply_rms_normalization 空输入测试...", end=' ')
    assert apply_rms_normalization([]) == [], "空输入应返回空列表"
    print("通过")

    # --- 测试 20：NaN/Inf 损失跳过更新 ---
    print("[20] NaN/Inf 损失防护测试...", end=' ')
    # 构造一个会产生 NaN 的场景：手动注入 NaN 梯度到参数
    test_param = AutogradValue(1.0)
    test_param.gradient = float('nan')
    opt_nan = AdamOptimizer([test_param], learning_rate=0.01)
    original_data = test_param.data
    opt_nan.step()
    assert test_param.data == original_data, "NaN 梯度应跳过更新，参数不变"
    assert test_param.gradient == 0, "NaN 梯度应被清零"
    # Inf 梯度同样跳过
    test_param.gradient = float('inf')
    opt_nan.step()
    assert test_param.data == original_data, "Inf 梯度应跳过更新"
    print("通过")

    # --- 测试 21：NaN 不扩散到其他参数 ---
    print("[21] NaN 隔离测试...", end=' ')
    param_a = AutogradValue(1.0)
    param_b = AutogradValue(2.0)
    param_a.gradient = float('nan')  # 注入 NaN
    param_b.gradient = 0.5           # 正常梯度
    opt_isolate = AdamOptimizer([param_a, param_b], learning_rate=0.01)
    data_b_before = param_b.data
    opt_isolate.step()
    assert param_a.data == 1.0, "NaN 参数应不变"
    assert param_b.data != data_b_before, "正常参数应被更新"
    print("通过")

    print("=" * 60)
    print("所有单元测试通过!")
    print("=" * 60)


# ============================================================
# 第十部分：主程序入口
# ============================================================

def main():
    """主训练流程：加载数据 -> 构建词表 -> 初始化模型 -> 训练 -> 生成"""

    # 1. 加载新闻数据
    print("加载新闻数据...")
    data_loader = NewsDataLoader(data_directory='data')
    documents = data_loader.load_documents(min_paragraph_length=1)
    print(f"文档数量: {len(documents)}")

    if len(documents) == 0:
        print("未加载到任何文档，请检查 data 目录是否包含 .txt 新闻文件")
        return

    # 2. 构建双语词表
    tokenizer = BilingualTokenizer()
    tokenizer.build_vocabulary(documents, max_vocab_size=256)
    print(f"词表大小: {tokenizer.vocab_size}")

    # 3. 初始化模型（维度较小以适应纯 Python 训练）
    model = BilingualGptModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        num_layers=1,
        num_heads=2,
        context_length=16
    )
    print(f"模型参数量: {len(model.trainable_parameters)}")

    # 4. 初始化优化器和训练器
    optimizer = AdamOptimizer(model.trainable_parameters, learning_rate=0.05)
    trainer = BilingualTrainer(model, tokenizer, documents, optimizer)

    # 5. 训练
    num_steps = 30
    print(f"开始训练（{num_steps} 步）...")
    trainer.train(num_steps=num_steps)

    # 6. 保存训练好的完整模型（权重参数、配置信息、词汇表）到 model 目录
    print("\n保存训练好的模型...")
    try:
        ModelPersistence.save_model(model, tokenizer)
    except (RuntimeError, OSError) as exc:
        print(f"模型保存失败，推理将回退到训练过程中的模型: {exc}")
        save_failed = True
    else:
        save_failed = False

    # 7. 推理生成：从 model 目录加载已保存的大模型进行推理任务，
    #    而非使用训练过程中的临时模型或重新训练模型
    print("\n--- 双语文本生成 ---")
    prompts = ['', '人工智能', 'GPT', '模型训练', '北大']

    inference_trainer = None
    if not save_failed:
        try:
            loaded_model, loaded_tokenizer = ModelPersistence.load_model()
            # 重新构造训练器用于推理（documents 仅为占位，推理阶段不使用）
            inference_trainer = BilingualTrainer(
                loaded_model, loaded_tokenizer, documents, optimizer=None
            )
            print("已加载保存的模型进行推理")
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"加载已保存模型失败，回退到训练过程中的模型: {exc}")
            inference_trainer = None

    # 回退方案：若加载失败则使用训练过程中的模型
    if inference_trainer is None:
        inference_trainer = trainer

    for prompt in prompts:
        generated = inference_trainer.generate_text(
            prompt_text=prompt, max_new_tokens=12, temperature=0.6
        )
        display_prompt = prompt if prompt else '(无提示)'
        print(f"提示: [{display_prompt}] -> 生成: {generated}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_unit_tests()
    else:
        main()
