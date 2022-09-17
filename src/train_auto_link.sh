aug=edge
cuda=$1
dataset_list=( cora citeseer ModelNet40 pubmed )
lr=0.001
wd=0
# hyper-parameter optimal
runs=5
epochs=500
p_epoch=200
aug_ratio=0.3
t=0.3
dropout=0.2
train_prop=0.1
p_lr=1e-3
g_lr=1e-3
g_l=$2
step=$3
deg=0
m_l=0
a_l=$4
d_l=0
aug_two=0
method=AllDeepSets
for train_prop in 0.8
do
for dname in ${dataset_list[*]} 
do
    if [ "$dname" = "cora" ]; then
        echo =============
        echo "${g_lr} ${m_l} ${g_l} ${step} ${t} ${dropout} ${aug_ratio}"
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --m_l $m_l \
            --hard 1 \
            --d_l $d_l \
            --aug_two $aug_two \
            --a_l $a_l
    echo "Finished training on ${dname}"
    elif [ "$dname" = "citeseer" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --m_l $m_l \
            --d_l $d_l \
            --aug_two $aug_two \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "pubmed" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --m_l $m_l \
            --d_l $d_l \
            --aug_two $aug_two \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_cora" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --m_l $m_l \
            --d_l $d_l \
            --aug_two $aug_two \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_dblp" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "zoo" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "20newsW100" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "Mushroom" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NTU2012" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "ModelNet40" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "yelp" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname}"
    elif [ "$dname" = "house-committees-100" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname} with noise 0.6"
    elif [ "$dname" = "walmart-trips-100" ]; then
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:${method} (default), Dataset: ${dname}"  
        python train_link_auto.py \
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
            --g_l $g_l \
            --step $step \
            --deg $deg \
            --d_l $d_l \
            --m_l $m_l \
            --a_l $a_l
        echo "Finished training on ${dname} with noise 0.6"   
    fi
done
done
echo "Finished all training!"