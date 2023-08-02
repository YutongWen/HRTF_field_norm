# HRTF_field_norm
This is the official implementation of the paper "Mitigating Cross-Database Differences for Learning Unified HRTF Representation" [[arXiv]](https://arxiv.org/abs/2307.14547) 
# how to train

norm way 4 is the proposed method,
norm way 5 is mix_pos_mix_ear, 
norm way 6 is mix_ear_individual_pos, 
norm way 2 is equator. 
please modify the output folder to include the norm way info (-o dir)
# experiments
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

# SVM Classification of Different Databases
In this section, we classify which database a given HRTF originates from using a kernel SVM. We train the kernel SVM on the subset of twelve positions which were found to be shared across eight of the selected databases; the remaining two databases do not contain these positions in common. In this task, a total of 144 subjects were utilized, with 18 subjects (the smallest database size) extracted from each database. This sampling methodology was adopted to reduce bias toward any database. The total number of HRTFs in this experiment is thus 12 (positions) × 18 (subjects) × 2 (ears) = 432 for each database.