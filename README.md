# Long-term Vision-Language Tracking 


> **ReasoningTrack: Chain-of-Thought Reasoning for Long-term Vision-Language Tracking**, Xiao Wang, Liye Jin, Xufeng Lou, Shiao Wang, Lan Chen, Bo Jiang, Zhipeng Zhang, arXiv:2508.05221
[[arXiv]](https://arxiv.org/abs/2508.05221) 
[[Code]](https://github.com/Event-AHU/Open_VLTrack)
## Abstract: 
Vision-language tracking has received increasing attention in recent years, as textual information can effectively address the inflexibility and inaccuracy associated with specifying the target object to be tracked. Existing works either directly fuse the fixed language with vision features or simply modify using attention, however, their performance is still limited. Recently, some researchers have explored using text generation to adapt to the variations in the target during tracking, however, these works fail to provide insights into the model's reasoning process and do not fully leverage the advantages of large models, which further limits their overall performance. To address the aforementioned issues, this paper proposes a novel reasoning-based vision-language tracking framework, named ReasoningTrack, based on a pre-trained vision-language model Qwen2.5-VL. Both SFT (Supervised Fine-Tuning) and reinforcement learning GRPO are used for the optimization of reasoning and language generation. We embed the updated language descriptions and feed them into a unified tracking backbone network together with vision features. Then, we adopt a tracking head to predict the specific location of the target object. In addition, we propose a large-scale long-term vision-language tracking benchmark dataset, termed TNLLT, which contains 200 video sequences. 20 baseline visual trackers are re-trained and evaluated on this dataset, which builds a solid foundation for the vision-language visual tracking task. Extensive experiments on multiple vision-language tracking benchmark datasets fully validated the effectiveness of our proposed reasoning-based natural language generation strategy.

## How to Download TNLLT dataset? 
![fig-1](./figures/TNLLT_samples.png)
Currently, the dataset can be downloaded from the BaiduYun: 
* **Baiduyun Drive:**

```
Full Dataset：
URL: https://pan.baidu.com/s/1Bsr3PENWaa9k_yCNpkUh_Q?pwd=1d6b
Code: 1d6b 

Example Sequence：
URL: https://pan.baidu.com/s/1onjHTESlh-V1vgR2AYgnLw?pwd=ed76
Code: ed76 
```




* **Dropbox**: 
```
To be released
```
```bash
1. cat TNLLT_part_* > TNLLT_restored.tar.gz
2. gunzip TNLLT_restored.tar.gz
3. md5sum -c TNLLT.tar.gz.md5 (optional)
4. tar -xvf TNLLT_restored.tar
```

## Tutorial for the Evaluation Toolkit: 
1. Download this github file: 
```bash
git clone https://github.com/Event-AHU/Open_VLTrack.git
```

2. Download annos from: [[Annos (word:bsf7)](https://pan.baidu.com/s/1oYdqdCLUnf5Ylu3QfcLcSQ?pwd=bsf7)]: 
```bash
unzip annos.zip and put it into Open_VLTrack/TNLLT_Evaluation_Toolkit/annos
```
> **Note**: 
> If there is a nested 'annos' folder after decompression, it should be removed.

3. Download the benchmark results from: [[Benchmark-Results (word:s48i)](https://pan.baidu.com/s/1Acx8tEWWdSquJWpx9AXdzA?pwd=s48i)]: 
```bash 
unzip tracking_results.zip and put it into Open_VLTrack/TNLLT_Evaluation_Toolkit/tracking_results
```
> **Note**: 
> If there is a nested 'tracking_results' folder after decompression, it should be removed.

4. Open the Matlab and run the script: 
```bash
run_tracker_performance_evaluation.m
```
> **Note**: 
> In the file `run_tracker_performance_evaluation.m`, you can
> 1. Change flag (line 25) for precision (1), normalized precision (2) or success rate (3).
> 2. Uncomment the line (line 167-line194) of `run_tracker_performance_evaluation` for the per-attribute performance plot.
> 3. In the file `utils/plot_draw_save.m`, you can change the color and line style of the plot.

5. Wait and see final results: 
![fig-1](./figures/SRPRNPR.png)


## Acknowledgement
This evaluation_toolkit code is modified based on the evaluation toolkit of [[LaSOT](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)]. 

