#!/bin/bash
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/chase-master/python/src
# input=/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
# input=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/hate-speech-and-offensive-language-master/data/labeled_data_all.csv
input=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/chase-master/data/ml/public/dt/labeled_data_all_2classes_only.csv

# output=/home/zqz/Work/chase/output
output=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/chase-master/output
# emg_model=/home/zqz/Work/data/GoogleNews-vectors-negative300.bin.gz
emg_model=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/Google-Embedding/GoogleNews-vectors-negative300.bin
emg_dim=300
# emt_model=/home/zqz/Work/data/Set1_TweetDataWithoutSpam_Word.bin
emt_model=/home/sjonnal3/Hate_Speech_Detection/Applied_Machine_Learning/glove.twitter.27B/glove.27B.25d.w2vformat.txt
emt_dim=300
data=dt
targets=2
word_norm=0

#cnn conv concat, Pack(1,2,3) and Gamback(2,3,4) baseline
#SETTINGS=("input=$input output=$output oov_random=0 dataset=$data model_desc=b_p_sub_conv[1,2,3](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=0 dataset=$data model_desc=b_g_sub_conv[2,3,4](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=1 emb_model=$emg_model emb_dim=$emg_dim dataset=$data model_desc=b_pg1_sub_conv[1,2,3](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=2 emb_model=$emg_model emb_dim=$emg_dim dataset=$data model_desc=b_pg2_sub_conv[1,2,3](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=1 emb_model=$emg_model emb_dim=$emg_dim dataset=$data model_desc=b_gg1_sub_conv[2,3,4](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=2 emb_model=$emg_model emb_dim=$emg_dim dataset=$data model_desc=b_gg2_sub_conv[2,3,4](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=1 emb_model=$emt_model emb_dim=$emt_dim dataset=$data model_desc=b_pt1_sub_conv[1,2,3](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=2 emb_model=$emt_model emb_dim=$emt_dim dataset=$data model_desc=b_pt2_sub_conv[1,2,3](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=1 emb_model=$emt_model emb_dim=$emt_dim dataset=$data model_desc=b_gt1_sub_conv[2,3,4](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" 
#"input=$input output=$output oov_random=2 emb_model=$emt_model emb_dim=$emt_dim dataset=$data model_desc=b_gt2_sub_conv[2,3,4](conv1d=100),maxpooling1d=4,flatten,dense=$targets-softmax" )

# #www model, lstm model
SETTINGS=(
# "input=$input output=$output oov_random=0 dataset=$data model_desc=www_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=1 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
"input=$input output=$output oov_random=2 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax"
# "input=$input output=$output oov_random=2 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl0_,dropout=0.2,conv1d=50-4,maxpooling1d=4,lstm=50-True,gmaxpooling1d,dense=$targets-softmax"
 
# #"input=$input output=$output oov_random=0 dataset=$data model_desc=wwwbase_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=0 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwbaseggl0_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=1 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwbaseggl1_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=2 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwbaseggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data model_desc=www_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=1 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=2 dataset=$data emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=1  model_desc=wwwwn1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1ggl0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=1 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1ggl1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=2 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1ggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=0 dataset=$data word_norm=1 model_desc=wwwwn1base_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=0 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1baseggl0_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=1 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1baseggl1_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=2 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1baseggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=1  model_desc=wwwwn1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1ggl0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=1 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1ggl1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=2 dataset=$data word_norm=1 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn1ggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=2  model_desc=wwwwn2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2ggl0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=1 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2ggl1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# "input=$input output=$output oov_random=2 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2ggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=0 dataset=$data word_norm=2 model_desc=wwwwn2base_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=0 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2baseggl0_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=1 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2baseggl1_,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# #"input=$input output=$output oov_random=2 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2baseggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-False,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=2 model_desc=wwwwn2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=0 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2ggl0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=1 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2ggl1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
# "input=$input output=$output oov_random=2 dataset=$data word_norm=2 emb_model=$emg_model emb_dim=$emg_dim model_desc=wwwwn2ggl2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dropout=0.2,dense=$targets-softmax" 
)
#"input=$input output=$output oov_random=0 dataset=$data emb_model=$emt_model emb_dim=$emt_dim model_desc=wwwt0_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
#"input=$input output=$output oov_random=1 dataset=$data emb_model=$emt_model emb_dim=$emt_dim model_desc=wwwt1_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" 
#"input=$input output=$output oov_random=2 dataset=$data emb_model=$emt_model emb_dim=$emt_dim model_desc=wwwt2_,dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=$targets-softmax" )

#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5](dropout=0.2,conv1d=100,maxpooling1d=v),flatten,dense=500,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5](dropout=0.2,conv1d=100,maxpooling1d=v),flatten,dense=1000,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5](dropout=0.2,conv1d=100,maxpooling1d=v),flatten,dense=1000,dense=200,dense=2-softmax")

#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),lstm=200-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4,lstm=100-True,gmaxpooling1d),dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4,lstm=200-True,gmaxpooling1d),dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-3,maxpooling1d=4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),flatten,dense=100,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,lstm=200-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,flatten,dense=100,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-3,maxpooling1d=4,conv1d=50-3,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4,maxpooling1d=4,conv1d=50-4,flatten,dense=2-softmax" 
#SETTINGS=("input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,conv1d=50-5,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-3,maxpooling1d=4,conv1d=50-3,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4,maxpooling1d=4,conv1d=50-4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,conv1d=50-5,flatten,dense=2-softmax" )


#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4-2,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5-2,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-6-2,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]{1}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]{1,2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]{1}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]{1,2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4]{1}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4]{1,2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"

#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[2]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]<3>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax")
# SETTINGS=(
# "input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[3],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
# "input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[4],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
# "input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[5],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
# "input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[3,4,5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
# below is line 114 from original code
# "input=$input output=$output dataset=dt model_desc=f_(conv1d=100-[3,4,5],so),lstm=100-True,gmaxpooling1d,dense=2-softmax"
# "input=$input output=$output dataset=dt model_desc=f_(conv1d=100-[3,4,5],so),lstm=100-True,gmaxpooling1d,dense=1-softmax" 
# )
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2,3,4]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[5]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[5]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[5]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax")
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" g
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]{2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax" )

#model_desc="b sub_conv[2,3,4](dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"



IFS=""

echo ${#SETTINGS[@]}
c=0
for s in ${SETTINGS[*]}
do
    printf '\n'
    c=$[$c +1]
    echo ">>> Start the following setting at $(date): "
    echo $c
    line="\t${s}"
    echo -e $line
    # python3 -m ml.classifier_dnn ${s}
    python3 -m ml.classifier_dnn_1 ${s}
    echo "<<< completed at $(date): "
done



