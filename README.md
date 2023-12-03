# [Mitigating Cross-Database Differences for Learning Unified HRTF Representation](https://ieeexplore.ieee.org/document/10248178)

[![GitHub](https://img.shields.io/github/stars/YutongWen/HRTF_field_norm)](https://github.com/YutongWen/HRTF_field_norm) [![IEEE Xplore](https://img.shields.io/badge/IEEE-10248178-E4A42C.svg)](https://ieeexplore.ieee.org/document/10248178) [![arXiv](https://img.shields.io/badge/arXiv-2307.14547-b31b1b.svg)](https://arxiv.org/abs/2307.14547)

This is the official implementation of the WASPAA paper "Mitigating Cross-Database Differences for Learning Unified HRTF Representation." 

[![Video](https://img.youtube.com/vi/NLxLF9mIc3U/hqdefault.jpg)](https://youtu.be/NLxLF9mIc3U)

This is also a follow-up work to our ICASSP 2023 paper "HRTF Field." [![IEEE Xplore](https://img.shields.io/badge/IEEE-10095801-E4A42C.svg)](https://ieeexplore.ieee.org/document/10095801) [![arXiv](https://img.shields.io/badge/arXiv-2210.15196-b31b1b.svg)](https://arxiv.org/abs/2210.15196) [![GitHub](https://img.shields.io/github/stars/yzyouzhang/hrtf_field)](https://github.com/yzyouzhang/hrtf_field) 

## How to run our code

norm way 4 is the proposed method,

norm way 5 is mix\_pos\_mix_ear, 

norm way 6 is mix\_ear\_individual\_pos, 

norm way 2 is equator. 

Please modify the output folder (`-o` dir) and  include the norm way info (`--norm_way` norm_way).

### Experiments

experiment 1

```bash
python3 hrtf_normalization/train.py -o hrtf_normalization/exp1 -n listen crossmod sadie bili -t ari --norm_way 2
```
experiment 2

```bash
python3 hrtf_normalization/train.py -o hrtf_normalization/exp2 -n ari crossmod hutubs cipic -t riec --norm_way 2
```
experiment 3

```bash
python3 hrtf_normalization/train.py -o hrtf_normalization/exp3 -n crossmod sadie bili -t listen --norm_way 2
```
experiment 4

```bash
python3 hrtf_normalization/train.py -o hrtf_normalization/exp4 -n hutubs 3d3a bili listen crossmod cipic ita sadie ari
```
experiment 5

```bash
python3 hrtf_normalization/train.py -o hrtf_normalization/exp5 -n listen 3d3a bili crossmod cipic ita sadie ari riec -t hutubs --norm_way 2
```

### SVM classification of different databases
To reproduce the result in Section 3 of our paper, simply run the following:

```bash
python3 database_classification.py
```

## Citation
```
@INPROCEEDINGS{wen2023mitigate,
  author={Wen, Yutong and Zhang, You and Duan, Zhiyao},
  booktitle={Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)}, 
  title={Mitigating Cross-Database Differences for Learning Unified {HRTF} Representation}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/WASPAA58266.2023.10248178}}

@INPROCEEDINGS{zhang2023hrtf,
  author={Zhang, You and Wang, Yuxiang and Duan, Zhiyao},
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={{HRTF} Field: Unifying Measured {HRTF} Magnitude Representation with Neural Fields}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095801}}
```