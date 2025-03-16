import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class TNLLTDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.tnllt_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.sequence_list


    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        absent_label_label_path = '{}/{}/absent_label.txt'.format(self.base_path, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        # full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')
        with open(str(absent_label_label_path), 'r') as file:
            lines = file.read().splitlines()
            absent_label = np.array([list(map(float, line.split())) for line in lines], dtype=np.float64)

        # out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        # out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = absent_label

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)

        frames_list = ['{}/{:05d}.png'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = sequence_name

        language_file = os.path.join(self.base_path, sequence_name,"language.txt")
        
        with open(language_file, 'r') as f:
            language = f.readlines()[0].rstrip()

        return Sequence(sequence_name, frames_list, 'tnllt', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible,language=language)

    def __len__(self):
        return len(self.sequence_list)
    
    def _get_sequence_list(self):
        with open('/wangx/DATA/Code/jinliye/ODTrack/lib/train/data_specs/tnl_lt_test_split.txt', 'r') as file:
            sequence_list = [line.strip() for line in file.readlines()]

        return sequence_list
    


