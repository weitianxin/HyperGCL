

dataset_list=( cora citeseer  \
    NTU2012 ModelNet40 house-committees-100 )
t_attack=$1
cuda=$2
for dname in ${dataset_list[*]} 
do
    python generate_attack.py --dname $dname --cuda $cuda --t_attack $t_attack
done