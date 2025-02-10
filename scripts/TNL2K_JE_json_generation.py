############################################################################################
####                                Fot LaSOT dataset 
############################################################################################
import os
import numpy as np 
import json
import pdb 

LaSOT_path = "/home/wangxiao/Downloads/projects/TNL2K_TPAMI/TNL2K_JE_dataset/"
video_files = os.listdir(LaSOT_path)
video_files = np.sort(video_files)

## use this class to avoid some array or other format issues in json. 
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


dict = {}

for idx in range(len(video_files)): 
# for idx in range(5): ## for test this code work or not. 
    video_name = video_files[idx]
    img_path = LaSOT_path + video_name + '/imgs/'
    gt_path  = LaSOT_path + video_name + '/groundtruth.txt'

    print("==>> video Name: ", video_name, " current-index/total: ", idx, "/", len(video_files), ", please wait ... ")
    
    img_files = sorted([p for p in os.listdir(img_path) if os.path.splitext(p)[1] == '.jpg'])

    if len(img_files) == 0: 
        img_files = sorted([p for p in os.listdir(img_path) if os.path.splitext(p)[1] == '.png'])

    # pdb.set_trace()

    gt_files  = np.loadtxt(gt_path, delimiter=',') 
    init_rect = gt_files[0]
    init_rect_first = init_rect.tolist()
    
    # pdb.set_trace()
    img_names_list = []
    gt_files_list = []

    #### for each image and ground truth 
    for img_idx in range(len(img_files)): 
        img_names = video_name + '/imgs/' + img_files[img_idx]
        img_names_list.append(img_names)
        gt_files_list.append(gt_files[img_idx])
    
    # pdb.set_trace()
    
    #### collect and save into one dict. 
    dict_collect = {'video_dir': video_name, 'init_rect': init_rect_first, 'img_names': img_names_list, 'gt_rect': gt_files_list}
    dict[video_name] = dict_collect


dumped = json.dumps(dict, cls=NumpyEncoder) 
with open('TNL2K_JE_test.json', 'w+') as f:
    json.dump(dumped, f)

print("==>> Done !")

file = open('TNL2K_JE_test.json','r',encoding='utf-8')
benchmark_info = json.load(file)
print(benchmark_info)