aug=edge
cuda=$1
dataset_list=( cora citeseer pubmed ModelNet40 coauthor_cora coauthor_dblp \
    NTU2012 zoo Mushroom 20newsW100 \
    house-committees-100 walmart-trips-100 yelp )
lr=0.001
wd=0
runs=20
epochs=500
p_epoch=200
aug_ratio=0.3
t=0.3
dropout=0.2
train_prop=0.1
p_lr=0
g_lr=1e-3
step=1
m_l=0
mode="InfoNCE"
method=AllDeepSets
for train_prop in 0.1
do
for dname in ${dataset_list[*]} 
do
    if [ "$dname" = "cora" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
    echo "Finished training on ${dname}"
    elif [ "$dname" = "citeseer" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "pubmed" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_cora" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.3 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_dblp" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --epochs 1000 \
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
            --g_lr $g_lr \
            --g_l 1 \
            --step $step \
            --m_l $m_l \
            --a_l 0.3 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "zoo" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "20newsW100" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "Mushroom" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NTU2012" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "ModelNet40" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "yelp" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --epochs 1500 \
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
            --g_lr $g_lr \
            --g_l 5 \
            --step $step \
            --m_l $m_l \
            --a_l 0.1 \
            --mode $mode
        echo "Finished training on ${dname}"
    elif [ "$dname" = "house-committees-100" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.3 \
            --mode $mode
        echo "Finished training on ${dname} with noise 1.0"
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.3 \
            --mode $mode
        echo "Finished training on ${dname} with noise 0.6"
    elif [ "$dname" = "walmart-trips-100" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_auto.py \
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
            --g_lr $g_lr \
            --g_l 10 \
            --step $step \
            --m_l $m_l \
            --a_l 0.5 \
            --mode $mode
        echo "Finished training on ${dname} with noise 0.6"   
    fi
done
done
echo "Finished all training!"