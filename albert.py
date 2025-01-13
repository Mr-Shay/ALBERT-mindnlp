import mindspore.dataset as ds
from mindnlp.transformers import AlbertTokenizer
import numpy as np
from mindnlp.transformers import AlbertForSequenceClassification
from mindspore import Tensor, context
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd  # 用于读取Parquet文件
import mindspore

def load_sst2_parquet(file_path):
    """
    读取SST-2数据集的Parquet文件，提取句子和标签。

    Args:
        file_path (str): 数据文件路径。

    Returns:
        sentences (list): 句子列表。
        labels (list): 标签列表。
    """
    df = pd.read_parquet(file_path)
    # 确认Parquet文件包含'sentence'和'label'列
    if 'sentence' not in df.columns or 'label' not in df.columns:
        raise ValueError("Parquet文件必须包含'sentence'和'label'列")
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    return sentences, labels

# 加载训练和验证数据

dev_sentences, dev_labels = load_sst2_parquet('sst-2/validation.parquet')
# 如果测试集也有Parquet文件，可以类似加载
# test_sentences, test_labels = load_sst2_parquet('sst-2/test.parquet')

# 创建标签映射
label_list = sorted(list(set(dev_labels)))
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}
num_labels = len(label_list)

def create_dataset(sentences, labels, tokenizer, max_length=128, batch_size=32, repeat_num=1, shuffle=True):
    """
    创建MindSpore数据集。

    Args:
        sentences (list): 句子列表。
        labels (list): 标签列表。
        tokenizer (AlbertTokenizer): ALBERT分词器。
        max_length (int): 最大序列长度。
        batch_size (int): 批次大小。
        repeat_num (int): 重复次数。
        shuffle (bool): 是否打乱数据。

    Returns:
        dataset (mindspore.dataset): 创建的数据集。
    """
    input_ids = []
    attention_masks = []
    label_ids = []
    for sentence, label in zip(sentences, labels):
        encoded = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        label_ids.append(label)

    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)
    label_ids = np.array(label_ids)

    dataset = ds.NumpySlicesDataset(
        {'input_ids': input_ids, 'attention_mask': attention_masks, 'labels': label_ids},
        column_names=['input_ids', 'attention_mask', 'labels']
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_num)
    return dataset

# 初始化分词器
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# 创建训练和验证数据集

dev_dataset = create_dataset(dev_sentences, dev_labels, tokenizer, batch_size=32, repeat_num=1, shuffle=False)
# 如果有测试集，也可以创建测试数据集
# test_dataset = create_dataset(test_sentences, test_labels, tokenizer, batch_size=32, repeat_num=1, shuffle=False)

# 加载预训练模型，忽略尺寸不匹配的权重
model = AlbertForSequenceClassification.from_pretrained(
    'Alireza1044/albert-base-v2-sst2',
    num_labels=num_labels,
    ignore_mismatched_sizes=True  # 允许模型加载时忽略尺寸不匹配
)

# 打印可训练参数的数量和形状以进行调试
params = model.trainable_params()
print(f"Number of trainable parameters: {len(params)}")
for param in params:
    print(param.shape)

if not params:
    raise ValueError("模型没有可训练的参数，请检查模型加载是否正确。")

# 设置MindSpore上下文
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')  # 根据实际情况选择 'GPU' 或 'Ascend'

def infer(model, dataset):
    """
    推理函数，在验证集或测试集上计算准确率和F1分数。

    Args:
        model (AlbertForSequenceClassification): ALBERT模型。
        dataset (mindspore.dataset): 验证或测试数据集。

    Returns:
        metrics (dict): 评估指标，包括准确率和F1分数。
    """
    model.set_train(False)
    all_preds = []
    all_labels = []

    for batch in tqdm(dataset.create_dict_iterator(), total=dataset.get_dataset_size()):
        input_ids = Tensor(batch['input_ids'], mindspore.int32)
        attention_mask = Tensor(batch['attention_mask'], mindspore.int32)
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits.asnumpy()
        preds = np.argmax(logits, axis=-1)

        all_preds.extend(preds)
        all_labels.extend(labels.asnumpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return metrics



# 在验证集上进行推理和评估
eval_metrics = infer(model, dev_dataset)
print(f"Validation Accuracy: {eval_metrics['accuracy']:.4f}, F1: {eval_metrics['f1']:.4f}, Precision: {eval_metrics['precision']:.4f}, Recall: {eval_metrics['recall']:.4f}")

# 如果需要生成详细的分类报告
# 注意：这里假设 label_list 包含所有标签的名称，并且 all_labels 和 all_preds 已经包含了所有有效标签
print(classification_report(dev_labels, eval_metrics, target_names=label_list))
