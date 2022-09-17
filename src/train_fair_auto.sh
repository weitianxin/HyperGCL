cuda=$1
aug=$2
dataset_list=( german bail credit)
# ./train_fair.sh 0 edge 0.3 0 dropout=0.2 t=0.5 aug_ratio=0.25
# dataset_list=( bail )
# ./train_fair.sh 0 edge 0.2 0 dropout=0.3 t=0.3 aug_ratio=0.25
# dataset_list=( german )
# ./train_fair.sh 0 edge 0.3 0 dropout=0.2 t=0.5 aug_ratio=0.25
# dataset_list=( credit )
lr=0.001
wd=0
# hyper-parameter optimal
runs=5
epochs=200
p_epoch=100
aug_ratio=0.25
t=0.5
dropout=0.2
train_prop=0.1
p_lr=1e-3
# m_l=$3
# g_l=$4
metric=1
method=AllDeepSets
# method=AllSetTransformer
for m_l in 0.1
do
for g_l in 1
do
for train_prop in 0.1
do
for dname in ${dataset_list[*]} 
do
        echo =============
        echo "m_l $m_l g_l $g_l"
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_fair_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 2 \
            --MLP_num_layers 1 \
            --feature_noise 0.0 \
            --heads 4 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr $lr \
            --t $t \
            --p_lr $p_lr \
            --p_epochs $p_epoch \
            --aug_ratio $aug_ratio \
            --aug $aug \
            --dropout $dropout \
            --train_prop $train_prop \
            --m_l $m_l \
            --metric $metric \
            --g_l $g_l
    echo "Finished training on ${dname}"
done
done
done
done
echo "Finished all training for AllDeepSets!"