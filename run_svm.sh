# export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/chase-master/python/src
# python3 -m exp.classifier_gridsearch_main /home/ziqizhang/chase/output /home/ziqizhang/chase/data/ml/td/labeled_data_part1.csv _td-tdf "" 0 none,kb,sfm
python3 -m exp.classifier_gridsearch_main /home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/chase-master/output /home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/hate-speech-and-offensive-language-master/data/labeled_data_all.csv _td-tdf "" 0 none,kb,sfm
