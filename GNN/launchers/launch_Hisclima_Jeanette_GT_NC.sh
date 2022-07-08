#!/usr/bin/env bash
loss=NLL
ngfs=( 64,64,64,64 )
layers_MLPs=( 64,64,64,64 )
epochs=( 6000 )
trys=( 1 )
# # COLS
conjugates=( ROW )
# COLS
name_data=graph_k10_wh0ww0jh1jw1_maxwidth0.5_minradio1 # NORMAL GRAPH
data_path=/data/HisClima/JeanneteAndAlbatrossGT/graphs_preprocessed # PATH TO THE DATA
# models=( edgeconv transformer )
models=( edgeconv ) #TYPE OF NETWORKS. SEE "MODELS"
steps=800,5000
gamma_step=0.5
alpha_FPs=( 5 )
metrics=( loss )
mlp_dos=( 0 )
do_adjs=( 0.3 )
base_lr=0.01
test_lst=/data/HisClima/HisClimaProd/DLA/HisClima_0/testJeannette.lst
train_lst=/data/HisClima/HisClimaProd/DLA/HisClima_0/trainvalJeannette.lst
val_lst=/data/HisClima/HisClimaProd/DLA/HisClima_0/valJeannette.lst
cd ..
for mlp_do in "${mlp_dos[@]}"; do
for do_adj in "${do_adjs[@]}"; do
for alpha_FP in "${alpha_FPs[@]}"; do
for conjugate in "${conjugates[@]}"; do
for ngf in "${ngfs[@]}"; do
for layers_MLP in "${layers_MLPs[@]}"; do
for epoch in "${epochs[@]}"; do
for try in "${trys[@]}"; do
for model in "${models[@]}"; do
for metric in "${metrics[@]}"; do
python main.py --batch_size 32 \
--data_path ${data_path}/${name_data}/ \
--epochs ${epoch} --seed ${try} --trans_prob 0 \
--work_dir works/${conjugate}/work_graph_${conjugate}_${loss}_${ngf}ngfs_${try}_${model}_${name_data}_alpha_FP_${alpha_FP}_val_6points3_6pointsedge_${metric}_DO_corregido_domlp${mlp_do}_adj${do_adj}_mlp${layers_MLP}_PL \
--test_lst ${test_lst} \
--train_lst ${train_lst} \
--do_val --val_lst ${val_lst} --show_val 100 --metric ${metric} --mlp_do ${mlp_do} --do_adj ${do_adj} \
--load_model True --show_test 1000 --model ${model} --alpha_FP ${alpha_FP} --layers_MLP ${layers_MLP} \
--layers ${ngf} --adam_lr ${base_lr} --conjugate ${conjugate} --classify EDGES --g_loss ${loss} --steps ${steps} --gamma_step ${gamma_step}
done
done
done
done
done
done
done
done
done
done
conjugates=( COL )
name_data=graph_k10_wh0ww4jh1jw1_maxwidth0.5_minradio0.1 # COL CELL
for mlp_do in "${mlp_dos[@]}"; do
for do_adj in "${do_adjs[@]}"; do
for alpha_FP in "${alpha_FPs[@]}"; do
for conjugate in "${conjugates[@]}"; do
for ngf in "${ngfs[@]}"; do
for epoch in "${epochs[@]}"; do
for try in "${trys[@]}"; do
for model in "${models[@]}"; do
for metric in "${metrics[@]}"; do
for layers_MLP in "${layers_MLPs[@]}"; do
python main.py --batch_size 32 \
--data_path ${data_path}/${name_data}/ \
--epochs ${epoch} --seed ${try} --trans_prob 0 \
--work_dir  works/${conjugate}/work_graph_${conjugate}_${loss}_${ngf}ngfs_${try}_${model}_${name_data}_alpha_FP_${alpha_FP}_val_6points3_6pointsedge_${metric}_DO_corregido_domlp${mlp_do}_adj${do_adj}_mlp${layers_MLP}_PL \
--test_lst ${test_lst} \
--train_lst ${train_lst} \
--do_val --val_lst ${val_lst} --show_val 100 --metric ${metric} \
--load_model True --show_test 1000 --model ${model} --alpha_FP ${alpha_FP} --mlp_do ${mlp_do} --do_adj ${do_adj} --layers_MLP ${layers_MLP} \
--layers ${ngf} --adam_lr ${base_lr} --conjugate ${conjugate} --classify EDGES --g_loss ${loss} --steps ${steps} --gamma_step ${gamma_step}
done
done
done
done
done
done
done
done
done
done
##################HEADER
ngfs=( 64,64,64,64 )
epochs=( 2 )
name_data=graph_k10_wh0ww0jh1jw1_maxwidth0.5_minradio0
gamma_step=0.5
base_lr=0.001
for ngf in "${ngfs[@]}"; do
for epoch in "${epochs[@]}"; do
for try in "${trys[@]}"; do
for model in "${models[@]}"; do
python main.py --batch_size 32 \
--data_path ${data_path}/${name_data}/ \
--epochs ${epoch} --seed ${try} \
--work_dir  works/HEADER/work_graph_${loss}_${ngf}ngfs_${try}_${model}_${name_data}_PL2 \
--test_lst ${test_lst} \
--train_lst ${train_lst} \
--do_val --val_lst ${val_lst} \
--load_model True --show_test 500 --model ${model} \
--layers ${ngf} --adam_lr ${base_lr} --classify HEADER --g_loss ${loss} --gamma_step ${gamma_step}
done
done
done
done
cd launchers
