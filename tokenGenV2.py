from __future__ import absolute_import
from __future__ import print_function

from pyverilog.vparser.ast import Identifier, IntConst
from pyverilog.vparser.parser import parse
import re
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sklearn.metrics import confusion_matrix


import json
import numpy as np

# ================== 核心修改：替换为预训练Transformer模型 ==================
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

# 全局配置
MAX_SEQ_LENGTH = 256
MODEL_NAME = 'bert-base-uncased'


def verilog_ast_tree(filename):
    """解析Verilog生成语法树（保持不变）"""
    filelist = [filename]
    ast, directives = parse(filelist)
    for lineno, directive in directives:
        print('Line %d : %s' % (lineno, directive))
    return ast


def load_trojan_gates(verilog_path):
    """加载木马门列表（保持不变）"""
    # ... 原有代码不变 ...
    base_dir = os.path.dirname(verilog_path)
    trojan_path = os.path.join(base_dir, "trojan.txt")

    if os.path.exists(trojan_path):
        with open(trojan_path, 'r') as f:
            trojan_set = set(line.strip() for line in f)
            print(f"[DEBUG] 加载到的木马门列表: {trojan_set}")  # 调试输出
            return trojan_set
    print("[WARNING] 未找到trojan.txt文件")
    return set()


def extract_gate_name_from_trojan(trojan_string):
    """... 原有代码不变 ..."""
    # 匹配门实例名（紧跟模块名后的标识符）
    match = re.search(r"[A-Za-z_]\w*(?=\()", trojan_string)
    if match:
        return match.group()
    return None

def is_trojan_gate(gate_name, trojan_gates):
    """... 原有代码不变 ..."""
    for trojan_string in trojan_gates:
        trojan_gate_name = extract_gate_name_from_trojan(trojan_string)
        if trojan_gate_name == gate_name:
            return True
    return False

# ================== 修改点1：增强正则表达式处理Trojan相关字符 ==================
TROJAN_PATTERNS = [
    r'Trojan_?', r'Tj_?', r'Payload_?', r'Trigger_?',
    r'Tg_?', r'_T\d+', r'_troj\b', r'WX\d+_?Tj_?'
]


def sanitize_trojan_terms(signal):
    """动态替换Trojan特征字符（论文改进）"""
    for pattern in TROJAN_PATTERNS:
        signal = re.sub(pattern, lambda x: 'NODE' + str(np.random.randint(1000)), signal)
    return signal


def extract_gates_with_labels(ast, trojan_gates):
    """带木马标注的门信息提取（改进版）"""
    gates = []

    def _walk(node):
        if isinstance(node, tuple):
            for n in node:
                _walk(n)
            return
        if not hasattr(node, 'children'):
            return
        if node.__class__.__name__ == 'Instance':
            gate_type = node.module
            gate_name = node.name
            inputs = []
            outputs = []

            # 提取并清洗端口信息
            for port in node.portlist:
                portname = port.portname
                signal = ''
                if port.children():
                    child = port.children()[0]
                    if isinstance(child, Identifier):
                        # = 修改点2：应用Trojan特征清洗 =
                        signal = sanitize_trojan_terms(child.name)
                    elif isinstance(child, IntConst):
                        signal = str(child.value)

                cleaned_port = f"{portname}:{signal}"
                if portname.upper() in ('Y', 'Q', 'QN', 'OUT'):
                    outputs.append(cleaned_port)
                else:
                    inputs.append(cleaned_port)

            # = 修改点3：遵循论文NtN格式要求 =
            input_key = 'Input_ports' if len(inputs) > 1 else 'Input_port'
            formatted_str = (
                f"[CLS] Gate_type:{gate_type} "  # 添加Transformer特殊标记
                f"{input_key}:[{';'.join(inputs)}] "
                f"Output_port:[{';'.join(outputs)}] [SEP]"  # 明确结构化边界
            )

            is_trojan = is_trojan_gate(gate_name, trojan_gates)
            gates.append({
                "raw_text": formatted_str,
                "label": 1 if is_trojan else 0
            })
        for child in node.children():
            _walk(child)

    _walk(ast)
    return gates


# ================== 修改点4：使用BERT特征提取代替Word2Vec ==================
class NetlistDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ================== 修改点5：训练流水线重构 ==================
def train_model(gates_data):
    # 初始化预训练模型
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    # 准备数据集
    texts = [gate['raw_text'] for gate in gates_data]
    labels = [gate['label'] for gate in gates_data]
    dataset = NetlistDataset(texts, labels, tokenizer)

    # 动态计算class weights
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    weights = torch.tensor([1.0 / neg_count ** 0.5, 1.0 / pos_count ** 0.5])
    sampler = torch.utils.data.WeightedRandomSampler(
        weights[labels], len(labels), replacement=True
    )

    # 训练配置
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, sampler=sampler
    )
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 迁移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 训练循环
    model.train()
    for epoch in range(10):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    return model, tokenizer


def evaluate_model(model, tokenizer, gates_data):
    # 准备数据集
    texts = [gate['raw_text'] for gate in gates_data]
    labels = [gate['label'] for gate in gates_data]
    dataset = NetlistDataset(texts, labels, tokenizer)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['label'].to(device)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()
    # 计算TPR和TNR
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    return TPR, TNR

# ================== 主流程调整 ==================
if __name__ == "__main__":
    # 配置文件路径
    verilog_path = "Trojan In/RS232-T1000/uart.v"
    trojan_path = "Trojan In/RS232-T1000/trojan.txt"


    # 1. 加载木马数据
    print("\n==== 加载木马门列表 ====")
    trojan_gates = load_trojan_gates(trojan_path)

    # 2. 解析Verilog并标注
    print("\n==== 解析Verilog文件 ====")
    ast = verilog_ast_tree(verilog_path)
    gates = extract_gates_with_labels(ast, trojan_gates)

    # 3. 训练模型
    print("\n==== 训练Transformer模型 ====")
    model, tokenizer = train_model(gates)

    # 4. 保存模型
    output_dir = os.path.join(os.path.dirname(verilog_path), "ntn_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n==== 模型已保存至 {output_dir} ====")

    # 5. 评估模型并输出TPR和TNR
    print("\n==== 评估模型性能 ====")
    TPR, TNR = evaluate_model(model, tokenizer, gates)
    print(f"True Positive Rate (TPR): {TPR:.4f}")
    print(f"True Negative Rate (TNR): {TNR:.4f}")
