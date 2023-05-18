# HRTF_field_norm

# how to train
# norm way 4 is proposed
# norm way 5 is mix_pos_mix_ear
# norm way 6 is mix_ear_individual_pos
# norm way 2 is equator
# please modify the output folder to include the norm way info (-o dir)
# exp1
python3 hrtf_normalization/train.py -o hrtf_normalization/exp1 -n listen crossmod sadie bili -t ari --norm_way 2
# exp2
python3 hrtf_normalization/train.py -o hrtf_normalization/exp2 -n ari crossmod hutubs cipic -t riec --norm_way 2
# exp3
python3 hrtf_normalization/train.py -o hrtf_normalization/exp3 -n crossmod sadie bili -t listen --norm_way 2
# exp4
python3 hrtf_normalization/train.py -o hrtf_normalization/exp4 -n hutubs 3d3a bili listen crossmod cipic ita sadie ari
# exp5
python3 hrtf_normalization/train.py -o hrtf_normalization/exp5 -n listen 3d3a bili crossmod cipic ita sadie ari riec -t hutubs --norm_way 2