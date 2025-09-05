import torch
from transformers import BertTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration

import torch.nn as nn
# qwen
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

from qwen_vl_utils import process_vision_info
from PIL import Image

from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

import base64
import requests

def image_to_base64(image_path):
    """将本地图片转为 Base64 Data URL"""
    with open(image_path, "rb") as f:
        return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"



save_name = '/wangx_nas/JLY/Code/LongTimeTracking/RLModels/easyr1/FULL_ioubf16_sft372/mysave/global_step_90/actor/huggingface'

# print("model loaded" + save_name)
cache_dir = "/rydata/jinliye/llmmodel"

system_prompt = "You are a visual tracking assistant that strictly analyzes only visual elements (ignore all text in images). Given an initial description and two consecutive frames (Frame 1 and Frame 2), first verify if the target object in Frame 1 matches the description, then determine if the description needs updating based on visual changes between frames (like position, shape, or color). Always respond in the exact format: <think>[your reasoning process]</think><d>yes/no</d><answer>[updated or original description for Frame 2]</answer>"

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("Waiting for debugger attach, press 'F5' to continue")
# debugpy.wait_for_client()

import time
class descriptgenRefiner(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    system_prompt = "You are a visual tracking assistant that strictly analyzes only visual elements (ignore all text in images). Given an initial description and two consecutive frames (Frame 1 and Frame 2), first verify if the target object in Frame 1 matches the description, then determine if the description needs updating based on visual changes between frames (like position, shape, or color). Always respond in the exact format: <think>[your reasoning process]</think><d>yes/no</d><answer>[updated or original description for Frame 2]</answer>"

    def __init__(self, blip_dir,bert_dir):
        super().__init__()
        self.system_prompt = system_prompt

        print(self.system_prompt)
        print(save_name)


    def extract_answers(self,answer_texts,tag):
        import re
        pattern = r'<{tag}>(.*?)</{tag}>'.format(tag=tag)
        processed_answers = []
        ans = answer_texts
        if tag == 'answer':
                match = re.search(pattern, ans, re.DOTALL)
                if match:
                    processed_answers.append(match.group(1).strip())
                else:
                    processed_answers.append("tracking the object")  # 默认值
        if tag == 'd':
                match = re.search(pattern, ans, re.DOTALL)
                if match:
                    processed_answers.append(match.group(1).strip())
                else:
                    processed_answers.append("no")  # 默认值
        if tag == 'think':
                match = re.search(pattern, ans, re.DOTALL)
                if match:
                    processed_answers.append(match.group(1).strip())
                else:
                    processed_answers.append("no thinking")  # 默认值

        return processed_answers
        

    def forward(self, image1,image2, cls):
        # start_time = time.perf_counter()
        img1_url = image_to_base64(image1)
        img2_url = image_to_base64(image2)

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model=save_name,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img1_url}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": img2_url}
                        },
                        {
                            "type": "text",
                            "text": cls
                        }
                    ]
                }
            ],
        )
        resp = chat_response.choices[0].message.content
        # print(resp)
        # print("=====")
        tag = self.extract_answers(resp,'d')
        ans= self.extract_answers(resp,'answer')
        think = self.extract_answers(resp,'think')
        # taglist,anslist,thinklist = self.gen_answers([prompt])

        if(tag[0] == 'yes'):
            descript = ans[0]
            # TODO: 更改“tracking the object”时候的描述，可能能提高性能,在actor26可以提升性能，在actor30性能降低（等什么时候满血再跑一次试试）
            if descript == "tracking the object":
                descript = cls
        else:
            descript = cls      
        # print("<begin>")
        # print(think)
        # print(tag)
        # print(descript)
        # # print(tracker.raw_text)
        # # print(image2_path)
        # print("</end>")
        # # 记录结束时间
        # end_time = time.perf_counter()

        # # 计算运行时间
        # elapsed_time = end_time - start_time

        # # 打印运行时间
        # print(f"代码运行时间：{elapsed_time:.6f} 秒")
        return descript,tag,think,ans

if __name__=='__main__':
    
    model = descriptgenRefiner("","")
    img1 = "/rydata/jinliye/RL/vltracking/EasyR1/LongTimeTracking/data/sampled_data2/sample0001/img1.jpg"
    img2 = "/rydata/jinliye/RL/vltracking/EasyR1/LongTimeTracking/data/sampled_data2/sample0001/img2.jpg"
    lan = "a ship"
    model.forward(img1,img2,lan)
    # breakpoint()


# class descriptgenRefiner(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, blip_dir,bert_dir):
#         super().__init__()
#         self.processor = BlipProcessor.from_pretrained(blip_dir)
#         self.model = BlipForConditionalGeneration.from_pretrained(blip_dir,torch_dtype=torch.float16).to("cuda")
#         self.tokenizer = BertTokenizer.from_pretrained(bert_dir)
        

#     def forward(self, image, cls):
#         if cls is None:
#             inputs = self.processor(image, return_tensors="pt").to("cuda", torch.float16)
#         else:
#             inputs = self.processor(image, cls, return_tensors="pt").to("cuda", torch.float16)
#         out = self.model.generate(**inputs)
#         descript = self.processor.decode(out[0], skip_special_tokens=True)
#         print("ahahahahha")
#         return descript

