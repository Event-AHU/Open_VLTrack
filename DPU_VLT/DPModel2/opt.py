import torch.nn as nn
import torch.nn.functional as F
import torch
# --- 损失函数 ---
# 弧预测：多分类问题，使用交叉熵
arc_criterion = nn.CrossEntropyLoss() # 忽略 PAD
# 标签预测：多分类问题，使用交叉熵
label_criterion = nn.CrossEntropyLoss(ignore_index=0)

# --- 训练循环骨架 ---
def train_loop(model, dataloader, optimizer, device):
    model.train()
    epoch_total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        word_ids = batch['word_ids'].to(device)
        heads = batch['heads'].to(device)
        deprel_ids = batch['deprel_ids'].to(device)
        mask = batch['mask'].to(device)
        upos_ids = batch['upos_ids'].to(device)
        
        optimizer.zero_grad()
        
        arc_scores, label_scores = model(word_ids, upos_ids,mask)

        # 3. 计算 Arc 损失 - 只计算非填充位置
        # arc_scores: [B, S, S], heads: [B, S]
        # 展平时需要确保 heads 的值在 [0, S) 范围内
        
        # *** 方法1：使用 mask 过滤（推荐）***
        valid_mask = mask[:, 1:].reshape(-1)  # [B*(S-1)]
        
        arc_scores_flat = arc_scores[:, 1:, :].reshape(-1, arc_scores.size(2))  # [B*(S-1), S]
        heads_flat = heads[:, 1:].reshape(-1)  # [B*(S-1)]
        
        # 只计算有效位置的损失
        arc_loss = arc_criterion(
            arc_scores_flat[valid_mask],
            heads_flat[valid_mask]
        )
        
        # 4. 计算 Label 损失
        gold_heads = heads[:, 1:]  # [B, S-1]
        batch_size, seq_len, _, deprel_size = label_scores.shape
        
        label_scores_mod = label_scores[:, 1:, :, :].reshape(-1, seq_len, deprel_size)
        gold_heads_flat = gold_heads.reshape(-1, 1)

        gold_label_scores = torch.gather(
            label_scores_mod, 1, 
            gold_heads_flat.unsqueeze(-1).expand(-1, 1, deprel_size)
        ).squeeze(1)

        gold_deprel_flat = deprel_ids[:, 1:].reshape(-1)
        
        # 只计算有效位置的 label loss
        label_loss = label_criterion(
            gold_label_scores[valid_mask],
            gold_deprel_flat[valid_mask]
        )
        
        # 5. 总损失
        batch_loss = arc_loss + label_loss
        
        # 6. 反向传播
        batch_loss.backward()
        optimizer.step()
        
        epoch_total_loss += batch_loss.item()
        
    return epoch_total_loss / len(dataloader)
# --- 评估循环骨架 (只关注预测部分) ---
# 评估时需要考虑 **Eisner** 或 **MST** 算法来找到最优的依存树，
# 但对于初学者，**贪婪解码**（直接取 Arc 分数最高的 Head）通常作为基线。


def evaluate(model, dataloader, device):
    model.eval()
    UAS, LAS = 0, 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            word_ids = batch['word_ids'].to(device)
            heads = batch['heads'].to(device)
            deprel_ids = batch['deprel_ids'].to(device)
            mask = batch['mask'].to(device)
            upos_ids = batch['upos_ids'].to(device)
            
            arc_scores, label_scores = model(word_ids, upos_ids, mask)
            
            pred_heads = torch.argmax(arc_scores, dim=2)
            
            batch_size, seq_len, _, deprel_size = label_scores.shape
            
            label_scores_mod = label_scores[:, 1:, :, :].reshape(-1, seq_len, deprel_size)
            pred_heads_flat = pred_heads[:, 1:].reshape(-1, 1)
            
            pred_label_scores = torch.gather(
                label_scores_mod, 1, 
                pred_heads_flat.unsqueeze(-1).expand(-1, 1, deprel_size)
            ).squeeze(1)
            
            pred_deprel_ids = torch.argmax(pred_label_scores, dim=1).reshape(batch_size, -1)
            
            # *** 使用 mask 过滤 ***
            non_pad_mask = mask[:, 1:]
            
            correct_arcs = (pred_heads[:, 1:] == heads[:, 1:]) & non_pad_mask
            UAS += correct_arcs.sum().item()
            
            correct_labels = (pred_deprel_ids == deprel_ids[:, 1:]) & correct_arcs
            LAS += correct_labels.sum().item()
            
            total_tokens += non_pad_mask.sum().item()
            
    return UAS / total_tokens, LAS / total_tokens