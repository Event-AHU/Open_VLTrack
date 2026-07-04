import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class Vocab:
    def __init__(self, data):
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.upos_to_id = {'<PAD>': 0}
        self.id_to_upos = {0: '<PAD>'}
        # 显式添加root标签（与数据一致）
        self.deprel_to_id = {'<PAD>': 0, 'root': 1}  # 明确root的映射
        self.id_to_deprel = {0: '<PAD>', 1: 'root'}
        # -------------------------- 核心新增：二次数字→原始字符串映射 --------------------------
        # 从数据集获取原生 upos 数字（0-17）→ 字符串标签的映射（必须放在_build_vocab前）
        self.native_upos_id_to_str = self._get_native_upos_mapping(data)
        # 新增：Vocab二次映射后的新id → 原始字符串标签（最终需要的映射）
        self.vocab_upos_id_to_str = {}  # 如：0→'<PAD>', 1→'NOUN', 2→'PUNCT', ..., 19→'AUX'
        self._build_vocab(data)
        # -------------------------- 完善新增映射 + 调试信息 --------------------------
        self._fill_vocab_upos_id_to_str()  # 填充二次数字→字符串映射
        # 调试信息
        print(f"词汇表大小 - Words: {len(self.word_to_id)}, "
              f"Deprels: {len(self.deprel_to_id)}")
        print(f"Deprel标签: {list(self.deprel_to_id.keys())[:10]}...")  # 打印前10个
        # 新增调试：验证二次数字→字符串映射（前5个）
        print(f"Vocab UPOS id→字符串映射（前5个）: {[(k, self.vocab_upos_id_to_str[k]) for k in sorted(self.vocab_upos_id_to_str.keys())[:5]]}...")
    
    def _get_native_upos_mapping(self, data):
        """从数据集中提取原生 upos 数字（0-17）→ 字符串标签的映射（关键！）"""
        # 从train集的features获取ClassLabel的names列表
        upos_classlabel = data["train"].features["upos"].feature  # List(ClassLabel(...))→ClassLabel
        native_mapping = {}
        # 原生数字0-17对应names列表中的字符串（如0→'NOUN'，1→'PUNCT'...17→'AUX'）
        for native_id, label_str in enumerate(upos_classlabel.names):
            native_mapping[native_id] = label_str
        return native_mapping

    def _fill_vocab_upos_id_to_str(self):
        """填充：Vocab二次映射后的新id → 原始字符串标签"""
        # 遍历二次映射的所有新id（如0,1,2,...,18）
        for vocab_id, native_upos_val in self.id_to_upos.items():
            if native_upos_val == '<PAD>':
                # PAD对应的字符串还是<PAD>
                self.vocab_upos_id_to_str[vocab_id] = '<PAD>'
            else:
                # 原生upos数字（0-17）→ 对应字符串标签
                self.vocab_upos_id_to_str[vocab_id] = self.native_upos_id_to_str[native_upos_val]


    def _build_vocab(self, data):
        all_words = set()
        all_upos = set()
        all_deprel = set()
        
        for split in data.values():
            # breakpoint()
            for sentence in split:
                all_words.update(sentence['tokens'])
                all_upos.update(sentence['upos'])
                all_deprel.update(sentence['deprel'])
                
        # 填充词汇表
        for word in sorted(list(all_words)):
            self.add_word(word)
        for upos in sorted(list(all_upos)):
            self.add_upos(upos)
        for deprel in sorted(list(all_deprel)):
            self.add_deprel(deprel)

    def add_word(self, item):
        if item not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[item] = idx
            self.id_to_word[idx] = item
            
    def add_upos(self, item):
        if item not in self.upos_to_id:
            idx = len(self.upos_to_id)
            self.upos_to_id[item] = idx
            self.id_to_upos[idx] = item
            
    def add_deprel(self, item):
        if item not in self.deprel_to_id:
            idx = len(self.deprel_to_id)
            self.deprel_to_id[item] = idx
            self.id_to_deprel[idx] = item


class UDEWTSentenceDataset(Dataset):
    def __init__(self, dataset_split, vocab):
        self.data = dataset_split
        self.vocab = vocab

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sentence = self.data[idx]
        
        words = ['<ROOT>'] + sentence['tokens']
        
        def safe_int_convert(head_str):
            try:
                return int(head_str)
            except (ValueError, TypeError):
                return 0
        
        original_heads = [safe_int_convert(h) for h in sentence['head']]
        
        # *** 关键修复：验证 head 索引 ***
        original_len = len(sentence['tokens'])
        
        heads = [0]  # ROOT指向自己
        for i, head in enumerate(original_heads):
            if head == 0:
                heads.append(0)
            elif head > original_len:
                # 数据错误：head 超出原始句子长度
                print(f"Warning: 句子 {idx} 的第 {i} 个词 head={head} 超出长度 {original_len}，设为 0")
                heads.append(0)
            else:
                heads.append(head)  # 正常情况：head 加 1

        deprels = ['PAD'] + sentence['deprel']
        upos = ['PAD'] + sentence['upos']

        # 2. 转换为ID
        word_ids = [self.vocab.word_to_id.get(w, self.vocab.word_to_id['<UNK>']) for w in words]
        
        deprel_ids = [self.vocab.deprel_to_id.get(d, 0) for d in deprels]
        upos_ids = [self.vocab.upos_to_id.get(u, 0) for u in upos]
        
        # *** 添加最终验证 ***
        final_len = len(word_ids)
        for i, h in enumerate(heads):
            if h >= final_len:
                print(f"Error: 句子 {idx} 的 head[{i}]={h} >= 长度 {final_len}")
                heads[i] = 0  # 修正为 ROOT

        return {
            'word_ids': torch.tensor(word_ids, dtype=torch.long),
            'upos_ids': torch.tensor(upos_ids, dtype=torch.long),
            'heads': torch.tensor(heads, dtype=torch.long),
            'deprel_ids': torch.tensor(deprel_ids, dtype=torch.long),
            'length': len(word_ids),
            'original_sentence': words,
            'original_heads': original_heads
        }
def collate_fn(batch):
    max_len = max(item['length'] for item in batch)
    
    # 预分配张量
    batch_size = len(batch)
    word_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    upos_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    heads = torch.zeros(batch_size, max_len, dtype=torch.long)
    deprel_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = []
    original_sentences = []
    
    for i, item in enumerate(batch):
        length = item['length']
        
        # 填充数据
        word_ids[i, :length] = item['word_ids']
        upos_ids[i, :length] = item['upos_ids']
        heads[i, :length] = item['heads']
        deprel_ids[i, :length] = item['deprel_ids']
        mask[i, :length] = True
        
        lengths.append(length)
        original_sentences.append(item.get('original_sentence', []))
    
    return {
        'word_ids': word_ids,
        'upos_ids': upos_ids,
        'heads': heads,
        'deprel_ids': deprel_ids,
        'mask': mask,
        'length': torch.tensor(lengths, dtype=torch.long),
        'original_sentences': original_sentences
    }

def test_dataset():
    def check_raw_deprel(data, top_n=5):
        """检查数据集中所有deprel标签，重点看根节点标签"""
        deprel_counts = {}
        # 遍历所有数据split（如train/valid/test）
        for split_name, split_data in data.items():
            for sent in split_data:
                for deprel in sent['deprel']:
                    deprel_counts[deprel] = deprel_counts.get(deprel, 0) + 1
        
        # 按出现次数排序
        sorted_deprels = sorted(deprel_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"所有deprel标签（共{len(sorted_deprels)}种）：")
        for deprel, count in sorted_deprels[:top_n]:  # 打印出现次数最多的前5种
            print(f"  {deprel}: {count}次")
        
        # 检查是否存在根节点相关标签
        root_candidates = [d for d in deprel_counts if 'root' in d.lower()]
        print(f"\n可能的根节点标签：{root_candidates}")

    # 使用你的数据调用（假设你的数据变量名为`all_data`）
    # check_raw_deprel(all_data)
    """测试数据集类是否正常工作"""
    
    # 加载一小部分数据进行测试
    from datasets import load_dataset
    ds = load_dataset("universal_dependencies", "en_ewt", cache_dir='/rydata/jinliye/treeTrack/Dependency Parser/dataset')

    # 创建词汇表
    vocab = Vocab(ds)
    
    # 创建数据集
    train_dataset = UDEWTSentenceDataset(ds['train'], vocab)
    check_raw_deprel(ds)
    # 测试单个样本
    sample = train_dataset[0]
    print("=== 测试单个样本 ===")
    print(f"句子长度: {sample['length']}")
    print(f"原始句子: {sample['original_sentence']}")
    print(f"Word IDs: {sample['word_ids']}")
    # print(f"UPOS IDs: {sample['upos_ids']}")
    print(f"Heads: {sample['heads']}")
    print(f"Deprel IDs: {sample['deprel_ids']}")
    # breakpoint()
    
    # 测试DataLoader
    print("\n=== 测试DataLoader ===")
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    
    for i, batch in enumerate(dataloader):
        if i >= 1:  # 只看第一个批次
            break
            
        print(f"批次 {i}:")
        print(f"  Word IDs shape: {batch['word_ids'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  句子长度: {batch['length'].tolist()}")
        print(f"  原始句子: {batch['original_sentences']}")
        
        # 检查填充是否正确
        for j in range(len(batch['length'])):
            length = batch['length'][j]
            mask = batch['mask'][j]
            print(f"  句子{j} - 实际长度: {length}, 掩码True数量: {mask.sum().item()}")

# 运行测试
# test_dataset()