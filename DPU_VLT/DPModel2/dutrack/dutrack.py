import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.models.dutrack.itpn import fast_itpn_base_3324_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.dutrack.DPEncoder import DependencySemanticEncoder
local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 从环境变量获取当前进程的设备编号
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

class DUTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, semantic_encoder, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.semantic_encoder = semantic_encoder  # 新增：依存语义编码器
        self.aux_loss = aux_loss
        self.head_type = head_type
        # 新增：语义特征与视觉特征的融合层
        self.feat_fusion = nn.Linear(
            self.backbone.embed_dim + self.semantic_encoder.semantic_proj.out_features + self.backbone.embed_dim,
            self.backbone.embed_dim
        )
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                descript,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):
            # 新增：用依存语义编码器处理文本描述
            semantic_feat = self.semantic_encoder([descript[i]]) 
            x, aux_dict = self.backbone(
                                        z=template.copy(), 
                                        x=search[i], 
                                        l=list(descript[i]), 
                                        temporal_query=self.track_query, 
                                        top_K=self.token_len,
                                        semantic_feat=semantic_feat
                                        )
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach()  # stop grad  (B, N, C)

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError

import numpy as np

def build_dutrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')

    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'itpn_base':
        backbone = fast_itpn_base_3324_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,bert_dir=cfg.MODEL.BACKBONE.BERT_DIR)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    semantic_encoder = DependencySemanticEncoder(
        pretrained_parser_path=cfg.MODEL.PRETRAIN_DP,  # 预训练 Parser 路径（配置文件中定义）
        pre_trained_embeddings_path=cfg.MODEL.PRETRAIN_EMD,  # 传入加载的 GloVe 嵌入
        embed_dim=cfg.MODEL.BACKBONE.EMBED_DIM,  # 与 GloVe 维度一致
        # d_model=128,  # 保持配置中的设置（对应 Parser 的 hidden_dim）
        # nhead=4,  # 保持配置中的设置（需满足 d_model % nhead == 0，128%4=32 合法）
        # num_layers=3,  # 保持配置中的设置（对应 Parser 的 lstm_layers）
        ffnn_dim=256,  # 保持配置中的设置
        track_embed_dim=512,  # 保持配置中的设置
        dropout=0.3  # 保持配置中的设置
    ).to(device)
    print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_DP)
    print('Load embeding model from: ' + cfg.MODEL.PRETRAIN_EMD)
    model = DUTrack(
        backbone,
        box_head,
        semantic_encoder=semantic_encoder,  # 传入语义编码器
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOP_K,
    )
    if 'DUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
        file_name = cfg.MODEL.PRETRAIN_FILE
        pth = os.path.join(pretrained_path,file_name)
        checkpoint = torch.load(pth, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
    return model
