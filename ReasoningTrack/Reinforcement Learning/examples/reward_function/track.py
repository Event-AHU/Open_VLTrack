import re

def compute_iou(box1, box2):
    """
    计算两个矩形框的交并比（IOU）。
    
    参数:
    box1: 第一个矩形框的坐标，格式为[x1, y1, w1, h1]。
    box2: 第二个矩形框的坐标，格式为[x2, y2, w2, h2]。
    
    返回:
    iou: 两个矩形框的交并比。
    """
    
    # 计算两个矩形框的左上角和右下角坐标
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    left1, top1, right1, bottom1 = x1, y1, x1 + w1, y1 + h1
    left2, top2, right2, bottom2 = x2, y2, x2 + w2, y2 + h2
    
    # 计算交集的左上角和右下角坐标
    left_intersect = max(left1, left2)
    top_intersect = max(top1, top2)
    right_intersect = min(right1, right2)
    bottom_intersect = min(bottom1, bottom2)
    # print(left_intersect, top_intersect, right_intersect, bottom_intersect)
    # 计算交集的宽度和高度
    width_intersect = max(0, right_intersect - left_intersect)
    height_intersect = max(0, bottom_intersect - top_intersect)

    
    # 计算交集面积
    area_intersect = width_intersect * height_intersect
 
    # 计算两个矩形框的面积
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    
    # 计算并集面积
    area_union = area_box1 + area_box2 - area_intersect

    # 计算IOU
    iou = area_intersect / area_union
    
    return iou


from typing import Dict, List
from PIL import Image


def format_reward(predict: str) -> float:
    # pattern = r"^<think>(?:(?!</?think>)[\s\S]*?)</think>\s*<answer>(?:(?!</?answer>)[\s\S]*?)</answer><\|im_end\|>$"
    # pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    def check_tags(s):
        """
        检查字符串中是否存在 <think></think> 和 <answer></answer> 标志物。
        
        参数:
        s: 输入字符串
        
        返回:
        如果两个标志物都存在，返回 1；否则返回 0。
        """
        # 检查是否存在 <think>...</think>
        think_pattern = r"<think>.*?</think>"
        think_match = re.search(think_pattern, s, re.DOTALL)
        
        # 检查是否存在 <answer>...</answer>
        answer_pattern = r"<answer>.*?</answer>"
        answer_match = re.search(answer_pattern, s, re.DOTALL)

        # 检查是否出现 <d>...</d>
        d_pattern = r"<d>.*?</d>"
        d_match = re.search(d_pattern, s, re.DOTALL)
        
        # 如果两个标志物都存在，返回 1；否则返回 0
        if think_match and answer_match and d_match:
        # if think_match and answer_match:
            return 1
        else:
            return 0

    return check_tags(predict)
    # return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def reward_strict_format_dtagreward(predict: str) -> float:
    # 检查整体结构顺序：<think>...</think> 后跟 <d>...</d> 再跟 <answer>...</answer>
    pattern = re.compile(
        r'<think>.*?</think>.*?<d>(.*?)</d>.*?<answer>.*?</answer>',
        re.DOTALL  # 允许 . 匹配换行符
    )
    match = pattern.search(predict)
    if not match:
        return 0
    
    # 检查 <d> 标签内容是否为 yes/no（忽略大小写和前后空格）
    d_content = match.group(1).strip().lower()
    if d_content not in {'yes', 'no'}:
        return 0
    
    return 1

import sys
tracker_path = "/rydata/jinliye/RL/vltracking/EasyR1/LongTimeTracking/Trackingbaseline/backDUTrack/DUTrack"  # 替换为 tracker.py 所在的实际路径
sys.path.append(tracker_path)
from demo3 import track_and_visualize,batch_track_and_visualize
sys.path.append(tracker_path)


def iou_reward(predict: str, ground_truth: str, box :str,pic1, pic2, lan_desp) -> float:
    pattern = r'<answer>(.*?)</answer>'
    pattern2 = r'<d>(.*?)</d>'    
    match = re.search(pattern, predict, re.DOTALL)
    match2 = re.search(pattern2, predict, re.DOTALL)    
    if match2:
        d_content = match2.group(1).strip()  # 使用 strip() 去除多余的空白字符
        d = d_content
    else:
        d = "no"
    if d != "no" and d != "yes":
        d = "no"
    # 提取匹配的内容
    if match:
        answer_content = match.group(1).strip()  # 使用 strip() 去除多余的空白字符
        text = answer_content
    else:
        text = "tracking the object"
    init_bbox = box
    ground_truth = ground_truth
    if (type(init_bbox) is str):
        init_bbox = [float(coord) for coord in init_bbox.split(',')]
    if (type(ground_truth) is str):
        ground_truth = [float(coord) for coord in ground_truth.split(',')]
    TrackingResult = track_and_visualize(pic1,pic2,init_bbox,text)
    # 更新后的iou
    iou = compute_iou(TrackingResult[1]['target_bbox'], ground_truth)
    TrackingResult2 = track_and_visualize(pic1,pic2,init_bbox,lan_desp[88:])
    # 更新前的iou
    iou2 = compute_iou(TrackingResult2[1]['target_bbox'], ground_truth)

    d_reward = 0
    if d == "no" and iou2 >= iou:
        d_reward = 1
    if d == "yes" and iou2 < iou:
        d_reward = 1
    print("iou:",iou,"iou2:",iou2)
    if iou > 0.61:
        iou = 1
    else:
        iou = 0
    return iou,d_reward

def iou_reward_batch(predict: list, ground_truth: list, box: list, pic1: list, pic2: list, lan_desps: list) -> list:
    pattern = r'<answer>(.*?)</answer>'
    pattern2 = r'<d>(.*?)</d>'
    
    text_list = []  # 用于存储每个元素处理后的 text 结果
    d_list = []  # 用于存储每个元素处理后的 d 结果
    iou_list = []  # 用于存储每个元素的 iou 结果
    d_reward_list = []  # 用于存储每个元素的 d_reward 结果
    lan_desp = [lan_desp[88:] for lan_desp in lan_desps]
    # 处理 predict 列表
    for item in predict:
        match = re.search(pattern, item, re.DOTALL)
        match2 = re.search(pattern2, item, re.DOTALL)
        
        if match2:
            d_content = match2.group(1).strip()  # 使用 strip() 去除多余的空白字符
            d = d_content
        else:
            d = "no"
        if d != "no" and d != "yes":
            d = "no"
        
        # 提取匹配的内容
        if match:
            answer_content = match.group(1).strip()  # 使用 strip() 去除多余的空白字符
            text = answer_content
        else:
            text = "tracking the object"
        
        text_list.append(text)
        d_list.append(d)  # 如果需要 d 的结果，也可以存储起来

    # 处理 box 和 ground_truth 列表
    init_bbox_list = []
    gt_list = []
    for bbox, gt in zip(box, ground_truth):
        if isinstance(bbox, str):
            # 兼容多种分隔符：逗号、制表符、空格等
            coords = [c.strip() for c in re.split(r'[, \t]+', bbox) if c.strip()]
            init_bbox_list.append([float(coord) for coord in coords])
        elif isinstance(bbox, (list, tuple, np.ndarray)):
            init_bbox_list.append([float(coord) for coord in bbox])
        else:
            raise ValueError(f"Unsupported bbox type: {type(bbox)}")
        
        # 处理ground truth
        if isinstance(gt, str):
            coords = [c.strip() for c in re.split(r'[, \t]+', gt) if c.strip()]
            gt_list.append([float(coord) for coord in coords])
        elif isinstance(gt, (list, tuple, np.ndarray)):
            gt_list.append([float(coord) for coord in gt])
        else:
            raise ValueError(f"Unsupported gt type: {type(gt)}")

    # 跟踪和可视化
    TrackingResult = batch_track_and_visualize(pic1, pic2, init_bbox_list, text_list)
    TrackingResult2 = batch_track_and_visualize(pic1, pic2, init_bbox_list, lan_desp)

    # 计算 iou 和 d_reward
    for i in range(len(predict)):
        if TrackingResult[0]:
            iou = compute_iou(TrackingResult[1][i]['target_bbox'], gt_list[i])

            iou2 = compute_iou(TrackingResult2[1][i]['target_bbox'], gt_list[i])
        else:
            iou = 0
            iou2 = 0
        d_reward = 0
        if d_list[i] == "no" and iou2 >= iou:
            d_reward = 1
        if d_list[i] == "yes" and iou2 < iou:
            d_reward = 1
        if iou > 0.61:
            iou = iou
        else:
            iou = iou / 2
        iou_list.append(iou)
        d_reward_list.append(d_reward)
    
    # 返回结果
    return iou_list, d_reward_list


    

def yn_reward(predict: str, ground_truth: str, box :str,pic1, pic2, lan_desp) -> float:
    return 0

# groundtruth是第二张图片目标框
# boxes是第一张图片目标框
# lan_desps是文本信息
# predicts是模型回答结果
def compute_score(predicts: List[str], ground_truths: List[str], boxes: List[str], lan_desps: List[str], pic1s:List[Image.Image],pic2s:List[Image.Image],weight: List[float] = [0.1,0.1,0,0.3]) -> List[Dict[str, float]]:
# def compute_score(predicts: List[str], ground_truths: List[str],format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    # predict
    # '<think>Okay, let\'s see. The initial description is "the bald man in suit." In Frame 1, there\'s a guy with no hair, wearing a dark suit with a red and black pattern. He\'s standing on a street, and there\'s a crowd around him. The background has buildings, a car, and other people.\n\nNow looking at Frame 2, the same person is still there. But his position has changed. He\'s moved slightly to the right, maybe interacting with someone or something. The crowd seems more dense here, perhaps because he\'s the focal point. The environment still has the same elements— buildings, car, different people. Since the main change is his position and the surrounding crowd, the description should reflect these movements. The original text doesn\'t mention the movement or the crowd, so updating it to include those details would make it more accurate for Frame 2.\n</think><d>yes</d><answer>The bald man in suit moved slightly right, interacting with a crowd of various people.</answer>'
    # ground_truth
    # '376.9807,246.1965,107.4559,198.5894'
    # box
    # '262.0437,212.1914,172.7456,327.8086'
    # lan_des
    # '<image><image>response based on the system prompts, the initial language description is the bald man in suit'
    # type(pic1)
    # <class 'numpy.ndarray'>
    # pic1.shape
    for predict, ground_truth, box, lan_des, pic1,pic2  in zip(predicts, ground_truths,boxes,lan_desps,pic1s,pic2s):
        # breakpoint()
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        format2_score = reward_strict_format_dtagreward(predict)
        # def iou_reward(predict: str, ground_truth: str, pic1, pic2, lan_desp) -> float:
        iou_score,yn_score = iou_reward(predict, ground_truth,box,pic1,pic2,lan_des)

        # yn_score = yn_reward(predict, ground_truth,box,pic1,pic2,lan_des)
        scores.append(
            {
                "overall": weight[0] * format_score + weight[1] * format2_score + weight[2] * iou_score  + weight[3] * yn_score ,
                "format": format_score,
                "format_strict": format2_score,
                "iou":iou_score,
                "yn":yn_score,
            }
        )
    

    return scores



def compute_score_batch(predicts: List[str], ground_truths: List[str], boxes: List[str], lan_desps: List[str], pic1s:List[Image.Image],pic2s:List[Image.Image],weight: List[float] = [0.1,0.1,0,0.15]) -> List[Dict[str, float]]:
# def compute_score(predicts: List[str], ground_truths: List[str],format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    # predict
    # '<think>Okay, let\'s see. The initial description is "the bald man in suit." In Frame 1, there\'s a guy with no hair, wearing a dark suit with a red and black pattern. He\'s standing on a street, and there\'s a crowd around him. The background has buildings, a car, and other people.\n\nNow looking at Frame 2, the same person is still there. But his position has changed. He\'s moved slightly to the right, maybe interacting with someone or something. The crowd seems more dense here, perhaps because he\'s the focal point. The environment still has the same elements— buildings, car, different people. Since the main change is his position and the surrounding crowd, the description should reflect these movements. The original text doesn\'t mention the movement or the crowd, so updating it to include those details would make it more accurate for Frame 2.\n</think><d>yes</d><answer>The bald man in suit moved slightly right, interacting with a crowd of various people.</answer>'
    # ground_truth
    # '376.9807,246.1965,107.4559,198.5894'
    # box
    # '262.0437,212.1914,172.7456,327.8086'
    # lan_des
    # '<image><image>response based on the system prompts, the initial language description is the bald man in suit'
    # type(pic1)
    # <class 'numpy.ndarray'>
    # pic1.shape
    # breakpoint()
    iou_scores, yn_scores = iou_reward_batch(predicts, ground_truths, boxes, pic1s, pic2s, lan_desps)
    
    for predict, ground_truth, box, iou_score, yn_score in zip(predicts, ground_truths,boxes, iou_scores, yn_scores):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        format2_score = reward_strict_format_dtagreward(predict)
        
        scores.append(
            {
                "overall": weight[0] * format_score + weight[1] * format2_score + weight[2] * iou_score + weight[3] * yn_score,
                "format": format_score,
                "format_strict": format2_score,
                "iou": iou_score,
                "yn": yn_score,
            }
        )
    
    return scores


# ==========================
# 做测试
import importlib
import cv2
import argparse
import numpy as np
from datasets import load_dataset
import os

def load_and_process_dataset_sample(dataset_name, split, sample_index):
    """
    从Hugging Face数据集加载并处理单个样本
    
    参数:
        dataset_name: 数据集名称
        split: 数据集划分
        sample_index: 样本索引
    
    返回:
        tuple: (success, image1, image2, init_info)
            success: 是否成功加载
            image1: 第一张图像 (BGR格式)
            image2: 第二张图像 (BGR格式)
            init_info: 包含语言描述和初始边界框的字典
    """
    try:
        # 加载数据集
        dataset = load_dataset(dataset_name, split=split)
        sample = dataset[sample_index]
        # breakpoint()
        # 获取图像数据 (假设已经是numpy数组)
        image1_rgb = sample['pic1']  # 第一张图像 (RGB)
        image2_rgb = sample['pic2']  # 第二张图像 (RGB)
        image1_rgb= np.array(image1_rgb) 
        image2_rgb= np.array(image2_rgb) 
        # 验证图像数据
        if not isinstance(image1_rgb, np.ndarray) or not isinstance(image2_rgb, np.ndarray):
            print("Error: Images must be numpy arrays")
            return False, None, None, None
            
        if image1_rgb.ndim != 3 or image2_rgb.ndim != 3 or image1_rgb.shape[2] != 3 or image2_rgb.shape[2] != 3:
            print("Error: Images must be 3-channel RGB arrays")
            return False, None, None, None
            
        if image1_rgb.shape != image2_rgb.shape:
            print("Error: Two images must have the same shape")
            return False, None, None, None
        
        # 转换为BGR格式
        image1 = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2BGR)
        
        # 准备初始信息
        init_info = {
            'class': sample['problem'],  # 语言描述
            'init_bbox': sample['bbox1']  # 初始边界框
        }
        
        return True, image1, image2, init_info
        
    except Exception as e:
        print(f"Error loading dataset sample: {str(e)}")
        return False, None, None, None
    
# 加载并处理 10 个样本
def load_samples(dataset_name, split, num_samples=10):
    samples = []
    image1_list = []
    image2_list = []
    init_bbox_list = []
    init_text_description_list = []
    
    for i in range(num_samples):
        success, image1, image2, init_info = load_and_process_dataset_sample(dataset_name, split, i)
        if success:
            init_bbox = [float(x) for x in init_info['init_bbox'].split(',')]
            init_text_description = init_info['class']
            samples.append((image1, image2, init_bbox, init_text_description))
            # 添加到各自的列表中
            image1_list.append(image1)
            image2_list.append(image2)
            init_bbox_list.append(init_bbox)
            init_text_description_list.append(init_text_description)
        else:
            print(f"Failed to load sample {i}")
    
    return samples, image1_list, image2_list, init_bbox_list, init_text_description_list


if __name__ == '__main__':

# 示例调用
    dataset_name = "Jinliye/TNL2KLTRLDataset2"
    split = "train"
    predicts = ["<think></think><d>yes</d><answer>shshshsh</answer>"]*10
    ground_truth = ["2.3,4.5,12.4,55.2"]*10
    samples, image1_list, image2_list, init_bbox_list, init_text_description_list = load_samples(dataset_name, split, num_samples=10)
    # def compute_score(predicts: List[str], ground_truths: List[str], boxes: List[str], lan_desps: List[str], pic1s:List[Image.Image],pic2s:List[Image.Image],weight: List[float] = [0.1,0.1,0.5,0.3]) -> List[Dict[str, float]]:
    compute_score(predicts,ground_truth,init_bbox_list,init_text_description_list,image1_list, image2_list)

