# OpenCompass 大模型评测解读及实战指南
> https://www.bilibili.com/video/BV1Gg4y1U7uc/
> https://github.com/InternLM/tutorial/blob/main/opencompass/opencompass_tutorial.md
## 关于评测的三个问题
1. 为什么需要评测
模型选型
大预言模型应用场景多，建立统一的评测标准才能更好的帮助我们去选择对模型进行选型
模型能力提升
对于开发者，评测的效果能让他们了解到模型的边界在哪
真实应用场景效果评测

2. 我们需要测什么
知识、推理、语言
长文本、智能体、多轮对话
情感、认知、价值观

3. 怎么测试大预言模型？
自动化客观评测
人机交互评测
基于大模型的大模型评测
例如基座模型评测的时候需要给一个instruct（即给个格式，让大语言模型按照格式回答），对话模型则直接像人一样提问和回答就好了
- 客观评测：无论模型怎么回答，只要能从回答中提取到我们想要的关键词，那就是正确的
- 主观评测：对一些主管的问题，如诗歌的谁写的更优，这种评测要做自动化的则需要用模型（如chatgpt）去评测模型

## OpenCompass 工具架构
![image](https://github.com/SaigeRing/InternLM-Camp/assets/154900748/29e60507-49a1-41ef-9749-155a89f7cc9e)
- 模型层：大模型评测所涉及的主要模型种类，OpenCompass  以基座模型和对话模型作为重点评测对象。
- 能力层：OpenCompass 从本方案从通用能力和特色能力两个方面来进行评测维度设计。在模型通用能力方面，从语言、知识、理解、推理、安全等多个能力维度进行评测。在特色能力方面，从长文本、代码、工具、知识增强等维度进行评测。
- 方法层：OpenCompass 采用客观评测与主观评测两种评测方式。客观评测能便捷地评估模型在具有确定答案（如选择，填空，封闭式问答等）的任务上的能力，主观评测能评估用户对模型回复的真实满意度，OpenCompass 采用基于模型辅助的主观评测和基于人类反馈的主观评测两种方式。
- 工具层：OpenCompass 提供丰富的功能支持自动化地开展大语言模型的高效评测。包括分布式评测技术，提示词工程，对接评测数据库，评测榜单发布，评测报告生成等诸多功能。
## 环境配置
  ```bash
  # intern-studio 开发机
  conda create --name opencompass --clone=/root/share/conda_envs/internlm-base
  source activate opencompass
  cd code
  git clone https://github.com/open-compass/opencompass
  cd opencompass
  pip install -e .
  ```
## 数据准备
  ```bash
  # 解压评测数据集到 data/ 处
  cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/code/opencompass/
  unzip OpenCompassData-core-20231110.zip
  # 将会在opencompass下看到data文件夹
  ```
## 查看支持的数据集和模型
  ```bash
  # 列出所有跟 internlm 及 ceval 相关的配置
  python tools/list_configs.py internlm ceval
  ```
  输出如下，InternLM2 已发布，会比教程中多出几种支持的模型。
  ```bash
  +--------------------------+--------------------------------------------------------+
  | Model                    | Config Path                                            |
  |--------------------------+--------------------------------------------------------|
  | hf_internlm2_20b         | configs/models/hf_internlm/hf_internlm2_20b.py         |
  | hf_internlm2_7b          | configs/models/hf_internlm/hf_internlm2_7b.py          |
  | hf_internlm2_chat_20b    | configs/models/hf_internlm/hf_internlm2_chat_20b.py    |
  | hf_internlm2_chat_7b     | configs/models/hf_internlm/hf_internlm2_chat_7b.py     |
  | hf_internlm_20b          | configs/models/hf_internlm/hf_internlm_20b.py          |
  | hf_internlm_7b           | configs/models/hf_internlm/hf_internlm_7b.py           |
  | hf_internlm_chat_20b     | configs/models/hf_internlm/hf_internlm_chat_20b.py     |
  | hf_internlm_chat_7b      | configs/models/hf_internlm/hf_internlm_chat_7b.py      |
  | hf_internlm_chat_7b_8k   | configs/models/hf_internlm/hf_internlm_chat_7b_8k.py   |
  | hf_internlm_chat_7b_v1_1 | configs/models/hf_internlm/hf_internlm_chat_7b_v1_1.py |
  | internlm_7b              | configs/models/internlm/internlm_7b.py                 |
  | ms_internlm_chat_7b_8k   | configs/models/ms_internlm/ms_internlm_chat_7b_8k.py   |
  +--------------------------+--------------------------------------------------------+
  +----------------------------+------------------------------------------------------+
  | Dataset                    | Config Path                                          |
  |----------------------------+------------------------------------------------------|
  | ceval_clean_ppl            | configs/datasets/ceval/ceval_clean_ppl.py            |
  | ceval_gen                  | configs/datasets/ceval/ceval_gen.py                  |
  | ceval_gen_2daf24           | configs/datasets/ceval/ceval_gen_2daf24.py           |
  | ceval_gen_5f30c7           | configs/datasets/ceval/ceval_gen_5f30c7.py           |
  | ceval_ppl                  | configs/datasets/ceval/ceval_ppl.py                  |
  | ceval_ppl_578f8d           | configs/datasets/ceval/ceval_ppl_578f8d.py           |
  | ceval_ppl_93e5ce           | configs/datasets/ceval/ceval_ppl_93e5ce.py           |
  | ceval_zero_shot_gen_bd40ef | configs/datasets/ceval/ceval_zero_shot_gen_bd40ef.py |
  +----------------------------+------------------------------------------------------+
  ```
- ## 启动评测
  在第一次运行时以 `--debug` 模式启动评估，并检查是否存在问题。
  （提前复制 `share` 文件夹下的 InternLM2 模型到 `model` 文件）  
  ```bash
  python run.py --datasets ceval_gen --hf-path /root/model/Shanghai_AI_Laboratory/internlm2-chat-7b/ --tokenizer-path /root/model/Shanghai_AI_Laboratory/internlm2-chat-7b/ --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1 --debug
  ```
- 评测结果如下：
  ```bash
  dataset                                         version    metric    mode    opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-7b
  ----------------------------------------------  ---------  --------  ------  -------------------------------------------------------------------------------------
  ceval-computer_network                          -          -         -       -
  ceval-operating_system                          -          -         -       -
  ceval-computer_architecture                     -          -         -       -
  ceval-college_programming                       4ca32a     accuracy  gen     18.92
  ceval-college_physics                           -          -         -       -
  ceval-college_chemistry                         e78857     accuracy  gen     0.00
  ceval-advanced_mathematics                      -          -         -       -
  ceval-probability_and_statistics                -          -         -       -
  ceval-discrete_mathematics                      -          -         -       -
  ceval-electrical_engineer                       ae42b9     accuracy  gen     18.92
  ceval-metrology_engineer                        ee34ea     accuracy  gen     50.00
  ceval-high_school_mathematics                   -          -         -       -
  ceval-high_school_physics                       -          -         -       -
  ceval-high_school_chemistry                     -          -         -       -
  ceval-high_school_biology                       -          -         -       -
  ceval-middle_school_mathematics                 -          -         -       -
  ceval-middle_school_biology                     -          -         -       -
  ceval-middle_school_physics                     -          -         -       -
  ceval-middle_school_chemistry                   -          -         -       -
  ceval-veterinary_medicine                       b4e08d     accuracy  gen     39.13
  ceval-college_economics                         f3f4e6     accuracy  gen     29.09
  ceval-business_administration                   c1614e     accuracy  gen     30.30
  ceval-marxism                                   -          -         -       -
  ceval-mao_zedong_thought                        51c7a4     accuracy  gen     70.83
  ceval-education_science                         591fee     accuracy  gen     62.07
  ceval-teacher_qualification                     4e4ced     accuracy  gen     77.27
  ceval-high_school_politics                      -          -         -       -
  ceval-high_school_geography                     -          -         -       -
  ceval-middle_school_politics                    -          -         -       -
  ceval-middle_school_geography                   -          -         -       -
  ceval-modern_chinese_history                    fc01af     accuracy  gen     65.22
  ceval-ideological_and_moral_cultivation         -          -         -       -
  ceval-logic                                     -          -         -       -
  ceval-law                                       a110a1     accuracy  gen     37.50
  ceval-chinese_language_and_literature           0f8b68     accuracy  gen     47.83
  ceval-art_studies                               2a1300     accuracy  gen     66.67
  ceval-professional_tour_guide                   4e673e     accuracy  gen     82.76
  ceval-legal_professional                        ce8787     accuracy  gen     30.43
  ceval-high_school_chinese                       -          -         -       -
  ceval-high_school_history                       -          -         -       -
  ceval-middle_school_history                     -          -         -       -
  ceval-civil_servant                             87d061     accuracy  gen     38.30
  ceval-sports_science                            -          -         -       -
  ceval-plant_protection                          -          -         -       -
  ceval-basic_medicine                            -          -         -       -
  ceval-clinical_medicine                         -          -         -       -
  ceval-urban_and_rural_planner                   95b885     accuracy  gen     58.70
  ceval-accountant                                002837     accuracy  gen     34.69
  ceval-fire_engineer                             bc23f5     accuracy  gen     12.90
  ceval-environmental_impact_assessment_engineer  c64e2d     accuracy  gen     38.71
  ceval-tax_accountant                            3a5e3c     accuracy  gen     42.86
  ceval-physician                                 6e277d     accuracy  gen     51.02
  ```
- 不开 `--debug`, 换 40G 显存开发机，避免 OOM，所有数据集都出结果。
  ```bash
  dataset                                         version    metric         mode      internlm2-chat-7b
  ----------------------------------------------  ---------  -------------  ------  ----------------------
  ceval-computer_network                          db9ce2     accuracy       gen       47.37
  ceval-operating_system                          1c2571     accuracy       gen       57.89
  ceval-computer_architecture                     a74dad     accuracy       gen       38.1
  ceval-college_programming                       4ca32a     accuracy       gen       18.92
  ceval-college_physics                           963fa8     accuracy       gen       5.26
  ceval-college_chemistry                         e78857     accuracy       gen       0
  ceval-advanced_mathematics                      ce03e2     accuracy       gen       0
  ceval-probability_and_statistics                65e812     accuracy       gen       11.11
  ceval-discrete_mathematics                      e894ae     accuracy       gen       18.75
  ceval-electrical_engineer                       ae42b9     accuracy       gen       18.92
  ceval-metrology_engineer                        ee34ea     accuracy       gen       50
  ceval-high_school_mathematics                   1dc5bf     accuracy       gen       0
  ceval-high_school_physics                       adf25f     accuracy       gen       31.58
  ceval-high_school_chemistry                     2ed27f     accuracy       gen       26.32
  ceval-high_school_biology                       8e2b9a     accuracy       gen       26.32
  ceval-middle_school_mathematics                 bee8d5     accuracy       gen       21.05
  ceval-middle_school_biology                     86817c     accuracy       gen       66.67
  ceval-middle_school_physics                     8accf6     accuracy       gen       52.63
  ceval-middle_school_chemistry                   167a15     accuracy       gen       80
  ceval-veterinary_medicine                       b4e08d     accuracy       gen       39.13
  ceval-college_economics                         f3f4e6     accuracy       gen       29.09
  ceval-business_administration                   c1614e     accuracy       gen       30.3
  ceval-marxism                                   cf874c     accuracy       gen       84.21
  ceval-mao_zedong_thought                        51c7a4     accuracy       gen       70.83
  ceval-education_science                         591fee     accuracy       gen       62.07
  ceval-teacher_qualification                     4e4ced     accuracy       gen       77.27
  ceval-high_school_politics                      5c0de2     accuracy       gen       21.05
  ceval-high_school_geography                     865461     accuracy       gen       42.11
  ceval-middle_school_politics                    5be3e7     accuracy       gen       38.1
  ceval-middle_school_geography                   8a63be     accuracy       gen       58.33
  ceval-modern_chinese_history                    fc01af     accuracy       gen       65.22
  ceval-ideological_and_moral_cultivation         a2aa4a     accuracy       gen       89.47
  ceval-logic                                     f5b022     accuracy       gen       13.64
  ceval-law                                       a110a1     accuracy       gen       37.5
  ceval-chinese_language_and_literature           0f8b68     accuracy       gen       47.83
  ceval-art_studies                               2a1300     accuracy       gen       66.67
  ceval-professional_tour_guide                   4e673e     accuracy       gen       82.76
  ceval-legal_professional                        ce8787     accuracy       gen       30.43
  ceval-high_school_chinese                       315705     accuracy       gen       21.05
  ceval-high_school_history                       7eb30a     accuracy       gen       75
  ceval-middle_school_history                     48ab4a     accuracy       gen       68.18
  ceval-civil_servant                             87d061     accuracy       gen       38.3
  ceval-sports_science                            70f27b     accuracy       gen       63.16
  ceval-plant_protection                          8941f9     accuracy       gen       68.18
  ceval-basic_medicine                            c409d6     accuracy       gen       57.89
  ceval-clinical_medicine                         49e82d     accuracy       gen       45.45
  ceval-urban_and_rural_planner                   95b885     accuracy       gen       58.7
  ceval-accountant                                002837     accuracy       gen       34.69
  ceval-fire_engineer                             bc23f5     accuracy       gen       12.9
  ceval-environmental_impact_assessment_engineer  c64e2d     accuracy       gen       38.71
  ceval-tax_accountant                            3a5e3c     accuracy       gen       42.86
  ceval-physician                                 6e277d     accuracy       gen       51.02
  ceval-stem                                      -          naive_average  gen       30.5
  ceval-social-science                            -          naive_average  gen       51.34
  ceval-humanities                                -          naive_average  gen       54.34
  ceval-other                                     -          naive_average  gen       46.53
  ceval-hard                                      -          naive_average  gen       11.63
  ceval                                           -          naive_average  gen       42.94
  ```
## 进阶-评测 LMDeploy 0.2.0 部署后在 C-Eval 数据集上的性能
  ```bash
  cp -r /root/share/model_repos/internlm2-chat-7b /root/model/Shanghai_AI_Laboratory/
  # lmdeploy convert internlm2-chat-7b  /root/model/Shanghai_AI_Laboratory/internlm2-chat-7b/ --dst_path /root/internlm2-chat-7b-turbomind
  export HF_MODEL=/root/model/Shanghai_AI_Laboratory/internlm2-chat-7b/
  export WORK_DIR=/root/model/Shanghai_AI_Laboratory/internlm2-chat-7b-4bit/
  lmdeploy lite auto_awq    $HF_MODEL   --calib-dataset 'ptb'   --calib-samples 128   --calib-seqlen 2048   --w-bits 4   --w-group-size 128   --work-dir $WORK_DIR
  ```
- 修改配置文件
  ```python
  from mmengine.config import read_base
  from opencompass.models.turbomind import TurboMindModel
  
  with read_base():
      # choose a list of datasets
      # from .datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
      from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
      # from .datasets.SuperGLUE_WiC.SuperGLUE_WiC_gen_d06864 import WiC_datasets
      # from .datasets.SuperGLUE_WSC.SuperGLUE_WSC_gen_7902a7 import WSC_datasets
      # from .datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
      # from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
      # from .datasets.race.race_gen_69ee4f import race_datasets
      # from .datasets.crowspairs.crowspairs_gen_381af0 import crowspairs_datasets
      # and output the results in a choosen format
      # from .summarizers.medium import summarizer
  
  
  datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
  
  internlm_meta_template = dict(round=[
      dict(role='HUMAN', begin='<|User|>:', end='\n'),
      dict(role='BOT', begin='<|Bot|>:', end='<eoa>\n', generate=True),
  ],
                                eos_token_id=103028)
  
  # config for internlm-chat-7b
  internlm_chat_7b = dict(
      type=TurboMindModel,
      abbr='internlm-chat-7b-turbomind',
      path="/root/model/Shanghai_AI_Laboratory/internlm2-chat-7b-4bit",
      engine_config=dict(session_len=512,
                         max_batch_size=2,
                         rope_scaling_factor=1.0),
      gen_config=dict(top_k=1,
                      top_p=0.8,
                      temperature=1.0,
                      max_new_tokens=100),
      max_out_len=100,
      max_seq_len=512,
      batch_size=2,
      concurrency=1,
      meta_template=internlm_meta_template,
      run_cfg=dict(num_gpus=1, num_procs=1),
  )
  
  models = [internlm_chat_7b]
  
  ```
  ```bash
  python run.py configs/eval_internlm2_chat_4bit_turbomind.py --debug
  ```
  评测正常进行，资源占用情况如下
  ```bash
  +------------------------------------------------------------------------------+
  | VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
  +-------------------------------------------+----------------------------------+
  | GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
  |===========================================+==================================|
  |   0  NVIDIA A100-SXM...  00000000:AD:00.0                 | 26152MiB / 40950MiB   41% /  25% |
  +-------------------------------------------+----------------------------------+
  ```
- 观察到很多 warning `total sequence length (453 + 100) exceeds `session_len` (512)`。所以中断任务，修改配置文件中的 `max_seq_len=1024`。重新运行 `python run.py configs/eval_internlm2_chat_4bit_turbomind.py`
  此时资源占用情况：
  ```bash
  +------------------------------------------------------------------------------+
  | VGPU-SMI 1.7.13       Driver Version: 535.54.03     CUDA Version: 12.2       |
  +-------------------------------------------+----------------------------------+
  | GPU  Name                Bus-Id           |        Memory-Usage     GPU-Util |
  |===========================================+==================================|
  |   0  NVIDIA A100-SXM...  00000000:AD:00.0                 | 26184MiB / 40950MiB   26% /  25% |
  +-------------------------------------------+----------------------------------+
  ```
- 评测结果
  ```bash
  dataset                                         version    metric    mode    internlm-chat-7b-turbomind
  ----------------------------------------------  ---------  --------  ------  ----------------------------
  ceval-computer_network                          db9ce2     accuracy  gen     57.89
  ceval-operating_system                          1c2571     accuracy  gen     73.68
  ceval-computer_architecture                     a74dad     accuracy  gen     57.14
  ceval-college_programming                       4ca32a     accuracy  gen     45.95
  ceval-college_physics                           963fa8     accuracy  gen     36.84
  ceval-college_chemistry                         e78857     accuracy  gen     33.33
  ceval-advanced_mathematics                      ce03e2     accuracy  gen     31.58
  ceval-probability_and_statistics                65e812     accuracy  gen     50
  ceval-discrete_mathematics                      e894ae     accuracy  gen     43.75
  ceval-electrical_engineer                       ae42b9     accuracy  gen     48.65
  ceval-metrology_engineer                        ee34ea     accuracy  gen     66.67
  ceval-high_school_mathematics                   1dc5bf     accuracy  gen     44.44
  ceval-high_school_physics                       adf25f     accuracy  gen     36.84
  ceval-high_school_chemistry                     2ed27f     accuracy  gen     36.84
  ceval-high_school_biology                       8e2b9a     accuracy  gen     26.32
  ceval-middle_school_mathematics                 bee8d5     accuracy  gen     36.84
  ceval-middle_school_biology                     86817c     accuracy  gen     76.19
  ceval-middle_school_physics                     8accf6     accuracy  gen     57.89
  ceval-middle_school_chemistry                   167a15     accuracy  gen     90
  ceval-veterinary_medicine                       b4e08d     accuracy  gen     43.48
  ceval-college_economics                         f3f4e6     accuracy  gen     45.45
  ceval-business_administration                   c1614e     accuracy  gen     45.45
  ceval-marxism                                   cf874c     accuracy  gen     84.21
  ceval-mao_zedong_thought                        51c7a4     accuracy  gen     79.17
  ceval-education_science                         591fee     accuracy  gen     68.97
  ceval-teacher_qualification                     4e4ced     accuracy  gen     84.09
  ceval-high_school_politics                      5c0de2     accuracy  gen     84.21
  ceval-high_school_geography                     865461     accuracy  gen     57.89
  ceval-middle_school_politics                    5be3e7     accuracy  gen     71.43
  ceval-middle_school_geography                   8a63be     accuracy  gen     75
  ceval-modern_chinese_history                    fc01af     accuracy  gen     78.26
  ceval-ideological_and_moral_cultivation         a2aa4a     accuracy  gen     89.47
  ceval-logic                                     f5b022     accuracy  gen     54.55
  ceval-law                                       a110a1     accuracy  gen     41.67
  ceval-chinese_language_and_literature           0f8b68     accuracy  gen     47.83
  ceval-art_studies                               2a1300     accuracy  gen     60.61
  ceval-professional_tour_guide                   4e673e     accuracy  gen     79.31
  ceval-legal_professional                        ce8787     accuracy  gen     52.17
  ceval-high_school_chinese                       315705     accuracy  gen     57.89
  ceval-high_school_history                       7eb30a     accuracy  gen     80
  ceval-middle_school_history                     48ab4a     accuracy  gen     81.82
  ceval-civil_servant                             87d061     accuracy  gen     55.32
  ceval-sports_science                            70f27b     accuracy  gen     73.68
  ceval-plant_protection                          8941f9     accuracy  gen     72.73
  ceval-basic_medicine                            c409d6     accuracy  gen     73.68
  ceval-clinical_medicine                         49e82d     accuracy  gen     36.36
  ceval-urban_and_rural_planner                   95b885     accuracy  gen     50
  ceval-accountant                                002837     accuracy  gen     51.02
  ceval-fire_engineer                             bc23f5     accuracy  gen     58.06
  ceval-environmental_impact_assessment_engineer  c64e2d     accuracy  gen     48.39
  ceval-tax_accountant                            3a5e3c     accuracy  gen     48.98
  ceval-physician                                 6e277d     accuracy  gen     57.14
  ```
  可以看到量化后总体得分还更高了。
