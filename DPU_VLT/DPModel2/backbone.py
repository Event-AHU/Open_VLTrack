import torch.nn as nn
import torch.nn.functional as F
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class BiaffineDependencyParser(nn.Module):
    def __init__(self, vocab_size, upos_size, deprel_size, embed_dim=512, hidden_dim=2048, lstm_layers=6, ffnn_dim=512, dropout=0.3, num_heads=6, pre_trained_embeddings=None):
        super().__init__()
        
        # 1. 嵌入层
        if pre_trained_embeddings is not None:
            self.word_embed = nn.Embedding.from_pretrained(pre_trained_embeddings, padding_idx=0, freeze=False)
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.upos_embed = nn.Embedding(upos_size, embed_dim, padding_idx=0)
        
        # 2. 上下文编码器 (Transformer 替换 LSTM)
        input_dim = embed_dim
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=lstm_layers)
        transformer_output_dim = input_dim  # Transformer 输出维度与输入相同

        # 3. 前馈网络 (用于 Arc 和 Label)，调整输入维度如果需要
        self.arc_h = nn.Sequential(nn.Linear(transformer_output_dim, ffnn_dim), nn.GELU(), nn.Dropout(dropout))
        self.arc_m = nn.Sequential(nn.Linear(transformer_output_dim, ffnn_dim), nn.GELU(), nn.Dropout(dropout))
        
        self.label_h = nn.Sequential(nn.Linear(transformer_output_dim, ffnn_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_m = nn.Sequential(nn.Linear(transformer_output_dim, ffnn_dim), nn.GELU(), nn.Dropout(dropout))

        # 4. Biaffine 层
        # Arc Biaffine: 输入 [Batch, Seq, FFNN_dim], 输出 [Batch, Seq, Seq]
        # self.arc_attn = nn.Bilinear(ffnn_dim, ffnn_dim, 1, bias=False)
        self.arc_weight = nn.Parameter(torch.randn(ffnn_dim, ffnn_dim))
        # Label Biaffine: 输入 [Batch, Seq, FFNN_dim], 输出 [Batch, Seq, Seq, Deprel_size]
        self.label_attn = nn.Bilinear(ffnn_dim, ffnn_dim, deprel_size, bias=False)
        
        self.deprel_size = deprel_size
        self.dropout = nn.Dropout(dropout)
        # 新增相对位置偏置参数（放在初始化阶段）
        self.max_distance = 20  # 可根据句长及显存情况调整
        self.rel_pos_embed = nn.Embedding(2 * self.max_distance + 1, 1)


    def forward(self, word_ids, upos_ids, mask):      
        # 1-4. 嵌入、编码、投影、Biaffine（保持不变）
        word_emb = self.word_embed(word_ids)
        upos_emb = self.upos_embed(upos_ids)
        emb = self.dropout(word_emb + upos_emb)
        
        # Transformer 输入需要 (seq, batch, dim)
        emb = emb.transpose(0, 1)  # (seq, batch, dim)
        emb = self.pos_encoder(emb)
        
        # 创建 padding mask: True 表示 padding
        src_key_padding_mask = ~mask  # mask is True for valid, so ~mask for padding
        
        transformer_out = self.encoder(emb, src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.transpose(0, 1)  # 回 (batch, seq, dim)
        lstm_out = self.dropout(transformer_out)  # 保持变量名为 lstm_out 以兼容后续代码

        h_arc = self.arc_h(lstm_out)
        h_label = self.label_h(lstm_out)
        m_arc = self.arc_m(lstm_out)
        m_label = self.label_m(lstm_out)
        
        # Arc scores
        # 修正后
        # 核心计算：[B, S, ffnn_dim] × [ffnn_dim, ffnn_dim] → [B, S, ffnn_dim]
        m_arc_transformed = torch.matmul(m_arc, self.arc_weight)  # 修饰词特征变换
        
        # 再与头部特征点积：[B, S, ffnn_dim] × [B, ffnn_dim, S] → [B, S, S]
        arc_scores = torch.matmul(m_arc_transformed, h_arc.transpose(1, 2))
        # ---------- 加入相对位置偏置 ----------
        seq_len = word_ids.size(1)
        # 生成 [S] 的位置索引
        pos_indices = torch.arange(seq_len, device=word_ids.device)
        # 计算 head-pos 与 modifier-pos 的差值 (head - modifier)
        rel_dist = pos_indices.view(1, -1) - pos_indices.view(-1, 1)  # [S, S]
        # clip 到 [-max_distance, max_distance]
        rel_dist_clipped = rel_dist.clamp(-self.max_distance, self.max_distance) + self.max_distance  # 平移到 [0, 2*max_d]
        # 查询偏置并添加到 arc_scores
        rel_bias = self.rel_pos_embed(rel_dist_clipped)  # [S, S, 1]

        rel_bias = rel_bias.squeeze(-1)  # [S, S]
        arc_scores = arc_scores + rel_bias.unsqueeze(0)  # broadcast 到 batch 维
        # 禁止自环（词不能依存于自己）
        batch_size, seq_len = word_ids.shape

        # Label scores
        W_label = self.label_attn.weight
        label_mid = torch.einsum('bik,rkj->birj', m_label, W_label)
        h_label_T = h_label.transpose(1, 2)
        label_scores = torch.einsum('birj,bjk->birk', label_mid, h_label_T)
        label_scores = label_scores.permute(0, 1, 3, 2).contiguous()
        
        # 5. 应用掩码 - *** 关键修复 ***
        
        # *** 使用有限的大负数而不是 -inf ***
        MASK_VALUE = -1e9
        
        # 掩码逻辑：
        # - head_mask: head 位置必须是有效词（非 PAD）
        # - modifier_mask: modifier 位置必须是有效词（非 PAD）
        head_mask = mask.unsqueeze(1).expand_as(arc_scores)  # [B, S, S]
        modifier_mask = mask.unsqueeze(2).expand_as(arc_scores)  # [B, S, S]
        
        # *** 修复：确保 ROOT (索引0) 始终可以作为 head ***
        # ROOT 应该对所有词都可见
        final_arc_mask = head_mask & modifier_mask  # [B, S, S]
        
        # 禁止自环（词不能依存于自己）
        batch_size, seq_len = word_ids.shape
        # diag_mask = torch.eye(seq_len, dtype=torch.bool, device=arc_scores.device).unsqueeze(0)
        # final_arc_mask = final_arc_mask & ~diag_mask
        final_arc_mask[:, 0, :] = False
        # 应用掩码 - 使用大负数而不是 -inf
        arc_scores = arc_scores.masked_fill(~final_arc_mask, MASK_VALUE)
        
        # *** 修复：ROOT 不能作为 modifier，但不影响其他词 ***
        # 将 ROOT 作为 modifier 的行设为大负数（但在训练时我们会忽略这一行）
        # arc_scores[:, 0, :] = MASK_VALUE

        return arc_scores, label_scores