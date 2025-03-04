'''
Author: jinliye jinliye@stu.ahu.edu.cn
Description: no descriptions
Date: 2024-11-30 10:07:32
LastEditors: jinliye
LastEditTime: 2024-12-27 19:49:20
'''

import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader,opencv_loader
from lib.train.admin import env_settings

class TNLLT(BaseVideoDataset):
    # TODO：image_loader可能要换一个，tnllt数据集中的图片是*.png
    def __init__(self, root=None, image_loader=opencv_loader, split="train", data_fraction=None):
        """
        args:
            root - path to the lasot dataset.
            image_loader  -  
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """        
        root = env_settings().tnllt_dir if root is None else root
        super().__init__('TNLLT', root, image_loader)

        # Keep a list of all classes
        # self.class_list = [f for f in os.listdir(self.root)]
        # self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        # self.seq_per_class = self._build_class_list()   
        # 
    def is_tracking_sequence(self):
        return True     

    def is_grounding_sequence(self):
        return True
    def is_vl_sequence(self):
        return True
    
    def _build_sequence_list(self, split=None):
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'tnl_lt_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'tnl_lt_val_split.txt')
            elif split == 'test':
                file_path = os.path.join(ltr_path, 'data_specs', 'tnl_lt_test_split.txt')
            else :
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        else:
            raise ValueError('split is none')

        return sequence_list
    
    def _read_target_visible(self, seq_path):
        target_visible = self.read_absent_label(seq_path)
        return target_visible
    
    def read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)
    
    def read_absent_label(self,seq_path):
        abs_label_file = os.path.join(seq_path, "absent_label.txt")
        absent_label = pandas.read_csv(abs_label_file, delimiter=',', header=None, dtype=np.int32, na_filter=False, low_memory=False).values
        # 移除所有维度为1的维度
        return torch.tensor(absent_label).squeeze()
    
    def read_attributes(self,seq_path):
        attributes_file = os.path.join(seq_path, "attributes.txt")
        attributes_label = pandas.read_csv(attributes_file, delimiter=',', header=None, dtype=np.int32, na_filter=False, low_memory=False).values
        # 移除所有维度为1的维度
        return torch.tensor(attributes_label).squeeze()

    def get_nlp(self,seq_id):
        seq_path = self._get_sequence_path(seq_id)
        nlp_file_path = os.path.join(seq_path, 'language.txt')
        
        if os.path.exists(nlp_file_path):
            with open(nlp_file_path, 'r', encoding='utf-8') as file:
                npl_text = file.read()
            return npl_text
        else:
            raise FileNotFoundError(f"No nlp.txt file found at {nlp_file_path}")

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)
    
    def get_sequence_info(self, seq_id):
        """ Returns information about a particular sequences,

        args:
            seq_id - index of the sequence

        returns:
            Dict
            """
        seq_path = self._get_sequence_path(seq_id)
        bbox = self.read_bb_anno(seq_path)
        absent_label = self.read_absent_label(seq_path)
        attributes_label = self.read_attributes(seq_path)
        lan = self.get_nlp(seq_id)

        # 检查宽高是否都大于0
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible = self._read_target_visible(seq_path) & valid.byte()
        visible = self._read_target_visible(seq_path)

        return {'bbox':bbox,'absent_label':absent_label,'attributes_label':attributes_label,'language':lan,'visible':visible ,'valid':valid}

    def get_name(self):
        return 'tnllt'

    # TODO: 下面几个短的函数考证一下，默认has_class_info为false
    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'imgs', '{:05}.png'.format(frame_id+1))  
    
    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))
    

    
    # TODO 确定一下数据集的解压格式  
    def get_frames(self, seq_id, frame_ids, anno=None):
        """ Get a set of frames from a particular sequence

        args:
            seq_id      - index of sequence
            frame_ids   - a list of frame numbers
            anno(None)  - The annotation for the sequence (see get_sequence_info). If None, they will be loaded.

        returns:
            list - List of frames corresponding to frame_ids
            list - List of dicts for each frame
            dict - A dict containing meta information about the sequence, e.g. class of the target object.

        """
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        # breakpoint()
        anno_frames = {}
        for key, value in anno.items():
            try:
                if key == 'language':
                    anno_frames[key] = [value] * len(frame_ids)
                elif key == 'attributes_label':
                    anno_frames[key] = [value] * len(frame_ids)
                else:
                    anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            except TypeError as e:
                print(f"Error processing key {key}: {e}")
                anno_frames[key] = [None] * len(frame_ids)

        object_meta = OrderedDict({'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'language':anno_frames['language']})                
        return frame_list, anno_frames, object_meta

