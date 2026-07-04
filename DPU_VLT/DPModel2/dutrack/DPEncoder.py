import torch
import torch.nn as nn
from .model.dataset import Vocab
from .model.backbone import BiaffineDependencyParser
import re
import numpy as np
import spacy
from typing import List, Tuple, Optional

class DependencySemanticEncoder(nn.Module):
    def __init__(self, 
                 pretrained_parser_path, 
                 pre_trained_embeddings_path,  # 训练时用的预训练嵌入（GloVe）
                 embed_dim=300,  # 训练时的 GloVe 维度（如50/100/300，按实际调整）
                 hidden_dim=256,  # 训练时显式设置的 256
                 lstm_layers=3,   # 训练时显式设置的 3（对应 Transformer 层数）
                 ffnn_dim=256,    # 训练时显式设置的 256
                 num_heads=6,     # Parser 类默认值（训练时未改则保持）
                 dropout=0.3,     # Parser 类默认值（训练时未改则保持）
                 track_embed_dim=512,
                 spacy_model_name="en_core_web_sm",
                 device: Optional[torch.device] = None):
        super().__init__()
        self.nlp = spacy.load(spacy_model_name, disable=["parser", "ner"]) 
        # 设备初始化：优先使用外部传入的device，否则自动适配CPU/GPU
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 1. 保存词表和预训练嵌入（与训练时完全一致）
        self.vocab = self._load_vocab()
        self.pre_trained_embeddings = self._load_glove_embeddings(pre_trained_embeddings_path,self.vocab,embed_dim)
        
        # 2. 初始化 Parser（参数与训练过程 1:1 对齐）
        self.parser = BiaffineDependencyParser(
            vocab_size=len(self.vocab.word_to_id),    # 训练时的词表大小
            upos_size=len(self.vocab.upos_to_id),     # 训练时的 upos 标签数（关键！之前错用了 deprel_to_id）
            deprel_size=len(self.vocab.deprel_to_id), # 训练时的依存标签数
            embed_dim=embed_dim,                 # 训练时的 GloVe 维度
            hidden_dim=hidden_dim,               # 训练时的 256（Transformer feedforward 维度）
            lstm_layers=lstm_layers,             # 训练时的 3（Transformer 层数）
            ffnn_dim=ffnn_dim,                   # 训练时的 256（Biaffine 前馈层维度）
            num_heads=num_heads,                 # 训练时未改，用 Parser 默认值 6
            dropout=dropout,                     # 训练时未改，用 Parser 默认值 0.3
            pre_trained_embeddings=self.pre_trained_embeddings  # 训练时的预训练嵌入
        )
        

        self._load_pretrained_weights(pretrained_parser_path)
        # 1. 语义特征投影层：适配 Transformer 输出（d_model），替换原 embed_dim*3
        # 逻辑：主体+动作+客体 各是 [B, S, d_model]，拼接后维度为 d_model*3
        self.semantic_proj = nn.Linear(embed_dim * 3, track_embed_dim)  
        
        # 方案：用 Transformer 输出的平均池化作为全局特征（维度 d_model）
        self.global_text_proj = nn.Linear(embed_dim, track_embed_dim)  
        
        # 3. 最终投影层：保持不变（语义特征+全局特征 拼接后是 512+512=1024）
        self.final_proj = nn.Linear(1024, 512)
        # ------------------------------------------------------------------
        
        # 冻结 parser 权重（逻辑不变）
        self.freeze_parser = True
        if self.freeze_parser:
            for param in self.parser.parameters():
                param.requires_grad = False
    
    def _load_glove_embeddings(self,glove_path, vocab, embed_dim):
        embeddings_dict = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        
        # 初始化嵌入矩阵（使用随机初始化作为回退）
        embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab.word_to_id), embed_dim))
        
        # 填充已知词的 GloVe 向量
        for word, idx in vocab.word_to_id.items():
            if word.lower() in embeddings_dict and len(embeddings_dict[word.lower()]) == embed_dim:
                embedding_matrix[idx] = embeddings_dict[word.lower()]
            elif word in embeddings_dict and len(embeddings_dict[word]) == embed_dim:
                embedding_matrix[idx] = embeddings_dict[word]
        
        return torch.from_numpy(embedding_matrix).float()
    def _split_with_punctuation(self,sent):
        # 匹配标点符号（这里列举常见的，可根据需求补充）
        pattern = r'([.,!?;:"\'()\[\]{}])'
        # 用空格分隔标点，再按空格split（会自动忽略空字符串）
        return re.sub(pattern, r' \1 ', sent).split()

    def _load_vocab(self):
        from datasets import load_dataset
        ds = load_dataset("universal_dependencies", "en_ewt", cache_dir='/rydata/jinliye/treeTrack/Dependency Parser/dataset')
        return Vocab(ds)

    def _load_pretrained_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.parser.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("预训练依存 parser 权重加载成功")

    # TODO: 这部分关系提取不够准
    def _extract_structured_feat(self, dependencies, word_emb):
        """从依存分析结果中提取结构化语义特征（主体+动作+客体）"""
        
        batch_size, seq_len, embed_dim = word_emb.shape
        structured_feat = torch.zeros(batch_size, 3, embed_dim, device=word_emb.device)  # [B, 3, D]
        
        for b in range(batch_size):
            subj_idx = -1  # 主体索引（如名词主语）
            action_idx = -1  # 动作索引（如动词/核心词）
            obj_idx = -1    # 客体索引（如宾语）
            
            # 关键修正：遍历当前句子的所有有效依存关系（dependencies[b]的长度=有效词数）
            # 用enumerate获取依存关系在列表中的索引，对应word_ids中“<ROOT>后的词索引”（i从1开始）
            for dep_idx, (current_word, head_idx, head_word, deprel) in enumerate(dependencies[b]):
                word_idx = dep_idx + 1  # 对应word_ids中的索引（1-based，跳过<ROOT>的0）
                
                # 示例规则：根据依存标签判断主体、动作、客体（可根据你的任务调整规则）
                if deprel == 'nsubj':  # 名词主语→主体
                    subj_idx = word_idx
                elif deprel in ['root', 'verb']:  # 核心词/动词→动作
                    action_idx = word_idx
                elif deprel in ['obj', 'pobj']:  # 宾语/介词宾语→客体
                    obj_idx = word_idx
            
            # 填充结构化特征（若未找到，保持全0；也可改用<UNK>的embedding）
            if subj_idx != -1:
                structured_feat[b, 0, :] = word_emb[b, subj_idx, :]
            if action_idx != -1:
                structured_feat[b, 1, :] = word_emb[b, action_idx, :]
            if obj_idx != -1:
                structured_feat[b, 2, :] = word_emb[b, obj_idx, :]       
        # 拼接并投影到目标维度
        structured_feat = structured_feat.reshape(batch_size, -1)  # [B, 3*D]
        return self.semantic_proj(structured_feat)  # [B, D]

    def _batch_sentence_to_tensor(self, processed_sentences: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量将分词后的句子转换为 word_ids、upos_ids、mask（整合_single_sentence_to_tensor逻辑）"""
        batch_size = len(processed_sentences)
        # 先获取每个句子的words、upos_tags（批量处理POS）
        all_words = []
        all_upos_tags = []
        for tokens in processed_sentences:
            # spaCy POS标注（输入是分词后的tokens拼接成的句子）
            doc = self.nlp(" ".join(tokens))
            upos_tags = [t.pos_ for t in doc]
            # 拼接<ROOT>和<PAD>（与word_ids对齐）
            words = ["<ROOT>"] + tokens
            upos_tags = ["<PAD>"] + upos_tags
            all_words.append(words)
            all_upos_tags.append(upos_tags)
        
        # 计算最大序列长度
        max_seq_len = max(len(words) for words in all_words)
        
        # 初始化张量
        word_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
        upos_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device=self.device)
        masks = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=self.device)
        
        # POS标签映射（从str到id）
        upos_str_to_id = {v: k for k, v in self.vocab.vocab_upos_id_to_str.items()}
        unk_upos_id = upos_str_to_id.get("X", 0)  # 未知POS的默认ID
        
        # 填充张量
        for b in range(batch_size):
            words = all_words[b]
            tags = all_upos_tags[b]
            seq_len = len(words)
            
            # 填充word_ids
            for i, word in enumerate(words):
                word_ids[b, i] = self.vocab.word_to_id.get(word, self.vocab.word_to_id["<UNK>"])
            
            # 填充upos_ids
            for i, tag in enumerate(tags):
                upos_ids[b, i] = upos_str_to_id.get(tag, unk_upos_id)
            
            # 填充mask
            masks[b, :seq_len] = True
        
        return word_ids, upos_ids, masks
    def forward(self, sentences):
        """
        输入：文本描述列表（batch 级）
        输出：融合结构化语义的文本 embedding [B, D]
        """
        # 第一步：预处理句子（统一格式+分词）
        processed_sentences = []
        for item in sentences:
            if isinstance(item, tuple):
                for sent in item:
                    sent = str(sent).strip() if not isinstance(sent, str) else sent.strip()
                    # 用spaCy分词（替代原有_split_with_punctuation，更精准）
                    tokens = [t.text for t in self.nlp(sent)]
                    processed_sentences.append(tokens)
            else:
                sent = str(item).strip() if not isinstance(item, str) else item.strip()
                tokens = [t.text for t in self.nlp(sent)]
                processed_sentences.append(tokens)
        
        # 第二步：批量转换为模型输入（word_ids、upos_ids、mask）
        word_ids, upos_ids, masks = self._batch_sentence_to_tensor(processed_sentences)
        
        # 第三步：依存解析（补充upos_ids参数，与Parser.forward匹配）
        with torch.no_grad():
            # 关键修正：传入3个参数（word_ids、upos_ids、mask）
            arc_scores, label_scores, encoder_out = self.parser(word_ids, upos_ids, masks)
            pred_heads = torch.argmax(arc_scores, dim=2)  # [B, S]
        
        # 第四步：构建批量依存关系（复用parse方法的标签解析逻辑）
        dependencies_batch = []
        deprel_size = len(self.vocab.deprel_to_id)
        batch_size, max_seq_len = word_ids.shape
        
        # 批量解析依存标签
        for b in range(batch_size):
            seq_len = masks[b].sum().item()  # 当前句子的有效长度
            current_tokens = processed_sentences[b]
            current_heads = pred_heads[b, :seq_len].tolist()  # 有效长度内的核心词索引
            
            # 提取当前句子的label_scores（[S, S, D]）
            current_label_scores = label_scores[b, :seq_len, :seq_len, :]
            # 构建依存标签
            dependencies = []
            for i in range(1, seq_len):  # 跳过<ROOT>（i=0）
                dep_word = current_tokens[i-1]
                head_idx = current_heads[i]
                head_word = "<ROOT>" if head_idx == 0 else current_tokens[head_idx-1]
                
                # 提取当前词→核心词的标签得分
                label_score_i = current_label_scores[i, head_idx]
                pred_label_idx = torch.argmax(label_score_i, dim=0).item()
                pred_label = self.vocab.id_to_deprel.get(pred_label_idx, "<UNK_LABEL>")
                
                dependencies.append((dep_word, head_idx, head_word, pred_label))
            dependencies_batch.append(dependencies)
        
        # 第五步：提取结构化语义特征（沿用原有逻辑）
        structured_feat = self._extract_structured_feat(dependencies_batch, encoder_out)  # [B, 300]
        
        # 第六步：全局文本特征（平均池化）
        valid_mask = masks.unsqueeze(-1).float()
        global_text_feat = (encoder_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)  # [B, d_model]
        global_text_feat = self.global_text_proj(global_text_feat)  # [B, 512]
        
        # 第七步：融合特征并输出
        final_feat = torch.cat([structured_feat, global_text_feat], dim=-1)  # [B, 812]
        final_feat = self.final_proj(final_feat)  # [B, 512]
        return final_feat