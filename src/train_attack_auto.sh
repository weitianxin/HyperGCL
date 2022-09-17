cuda=$1
aug=$2
dataset_list=( cora NTU2012 citeseer \
    ModelNet40 house-committees-100 )
# dataset_list=( cora )
lr=0.001
wd=0
# hyper-parameter optimal
runs=20
epochs=500
p_epoch=100
aug_ratio=0.3
t=0.3
dropout=0.2
train_prop=0.1
p_lr=1e-3
m_l=$3
g_l=$4
method=AllDeepSets
attack_list=( minmax net remove )
# method=AllSetTransformer
for train_prop in 0.1 0.01
do
for dname in ${dataset_list[*]} 
do
for attack in ${attack_list[*]} 
do
    echo "${attack}"
    if [ "$dname" = "cora" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 4 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
    echo "Finished training on ${dname}"
    elif [ "$dname" = "citeseer" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "pubmed" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_cora" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_dblp" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "zoo" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0.00001 \
            --epochs $epochs \
            --runs $runs \
            --cuda $cuda \
            --lr 0.01 \
            --t $t \
            --p_lr $p_lr \
            --p_epochs $p_epoch \
            --aug_ratio $aug_ratio \
            --aug $aug \
            --dropout $dropout \
            --train_prop $train_prop \
            --m_l $m_l \
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "20newsW100" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "Mushroom" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NTU2012" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "ModelNet40" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "yelp" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname}"
    elif [ "$dname" = "house-committees-100" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 1.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname} with noise 0.6"
    elif [ "$dname" = "walmart-trips-100" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 1.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_attack_auto.py \
            --method $method \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
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
            --g_l $g_l \
            --attack $attack
        echo "Finished training on ${dname} with noise 0.6"   
    fi
done
done
done
echo "Finished all training for AllDeepSets!"