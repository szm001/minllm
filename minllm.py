"""
microgpt_chinese.py
基于原生 Python 实现的中英文双语 GPT 大模型预训练系统

本文件在 microgpt0.py 的纯 Python GPT 算法基础上，扩展为支持中英文双语的预训练系统。
核心改进：
  1. 双语分词器：中文字符级 + 英文词级的混合分词策略
  2. 中文字符编码：覆盖 CJK 主区及扩展区，UTF-8 全链路处理
  3. 架构调整：扩展词表嵌入、特殊 token（PAD/BOS/EOS/UNK）、词表大小上限
  4. 数据加载：读取 data 目录下新闻文本文件，按段落切分文档
  5. 训练适配：支持提示文本引导的生成、序列截断、线性学习率衰减

所有变量/函数使用 snake_case，类使用 PascalCase，命名完全重写。
"""

import os
import math
import random
import sys

random.seed(42)


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

    相比 microgpt0.py 的纯字符级分词，本方案能更准确地处理双语的词汇边界。
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

        # 按频率降序排序
        sorted_tokens = sorted(token_counts.items(), key=lambda item: -item[1])

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
    替代 microgpt0.py 中使用名字列表的数据方式。
    全程使用 UTF-8 编码，确保中文字符正确解析。
    """

    def __init__(self, data_directory='data'):
        self.data_directory = data_directory

    def load_documents(self, min_paragraph_length=20):
        """
        加载所有新闻文件并按行切分为文档列表。

        参数 min_paragraph_length：过滤过短的行（如"原文链接"等噪声）。
        """
        documents = []
        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"数据目录不存在: {self.data_directory}")

        for filename in sorted(os.listdir(self.data_directory)):
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
    对应 microgpt0.py 中的 Value 类，重命名为更具语义的 AutogradValue。
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
        # 防护：零底数 + 非正指数导致 ZeroDivisionError
        if abs(base) < 1e-12 and exponent <= 0:
            base = 1e-12
        # 防护：负底数 + 分数指数返回复数，取绝对值兜底（模型中不应出现此情况）
        if base < 0 and not float(exponent).is_integer():
            base = abs(base)
        value = base ** exponent
        grad = exponent * base ** (exponent - 1)
        return AutogradValue(value, (self,), (grad,))

    def log(self):
        # 防护：data <= 0 时 math.log 会抛 ValueError，用极小正值兜底（概率下溢场景）
        safe_data = max(self.data, 1e-12)
        return AutogradValue(math.log(safe_data), (self,), (1.0 / safe_data,))

    def exp(self):
        # 防护：过大值会导致 OverflowError，截断到合理范围
        safe_data = max(-700, min(self.data, 700))
        return AutogradValue(math.exp(safe_data), (self,), (math.exp(safe_data),))

    def relu(self):
        return AutogradValue(max(0, self.data), (self,), (float(self.data > 0),))

    def detach(self):
        """返回同值但无计算图依赖的新节点，用于推理时释放内存"""
        return AutogradValue(self.data)

    def __repr__(self):
        return f"AutogradValue(data={self.data:.6f}, grad={self.gradient:.6f})"

    def __neg__(self):
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
        """
        topological_order = []
        visited_nodes = set()
        # 迭代式后序 DFS：用 (node, children_processed) 标记区分"待展开"和"待输出"
        stack = [(self, False)]
        while stack:
            node, children_processed = stack.pop()
            if children_processed:
                topological_order.append(node)
                continue
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            # 先压入"已完成子节点处理"标记，再压入子节点
            # 利用栈 LIFO 特性保证子节点先于父节点被处理
            stack.append((node, True))
            for child in node._children:
                if child not in visited_nodes:
                    stack.append((child, False))

        self.gradient = 1.0
        for node in reversed(topological_order):
            for child, local_gradient in zip(node._children, node._local_gradients):
                child.gradient += local_gradient * node.gradient


# ============================================================
# 第五部分：模型架构组件
# ============================================================

def initialize_parameter_matrix(output_dim, input_dim, std=0.08):
    """初始化参数矩阵（二维列表），高斯分布随机初始化"""
    return [[AutogradValue(random.gauss(0, std)) for _ in range(input_dim)]
            for _ in range(output_dim)]


def apply_linear_transformation(input_vector, weight_matrix):
    """线性变换：output = W * x，对应 microgpt0.py 的 linear 函数"""
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
        # 编码文档：BOS + token 序列 + EOS
        token_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(document) + [self.tokenizer.eos_id]
        # 截断至上下文长度（+1 是因为需要 target，即 n 个位置需要 n+1 个 token）
        token_ids = token_ids[:self.model.context_length + 1]

        num_positions = min(self.model.context_length, len(token_ids) - 1)
        if num_positions < 1:
            return None

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

        try:
            for step in range(num_steps):
                # 线性学习率衰减
                if learning_rate_decay:
                    self.optimizer.learning_rate = initial_lr * (1 - step / num_steps)

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
            current_token = random.choices(
                range(self.model.vocab_size),
                weights=[p.data for p in probs]
            )[0]
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
    documents = data_loader.load_documents(min_paragraph_length=20)
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

    # 6. 推理生成
    print("\n--- 双语文本生成 ---")
    prompts = ['', '人工智能', 'GPT', '模型训练', '北大']
    for prompt in prompts:
        generated = trainer.generate_text(prompt_text=prompt, max_new_tokens=12, temperature=0.6)
        display_prompt = prompt if prompt else '(无提示)'
        print(f"提示: [{display_prompt}] -> 生成: {generated}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        run_unit_tests()
    else:
        main()
