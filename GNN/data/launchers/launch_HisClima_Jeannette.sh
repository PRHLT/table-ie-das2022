#!/usr/bin/env bash
path=/data/HisClima/JeanneteAndAlbatrossGT #PATH TO SAVE THE GRAPHS
data_path="/data/HisClima/HisClimaProd/DLA/HisClima_0/all_corregido" # PATH TO THE DATA TO LOAD / PAGES
hisclima=true
###Normal
minradio=0
min_num_neighrbors=10
weight_radius_w=0
weight_radius_h=0
j_h=1
j_w=1
### NORMAL GRAPH
max_width_line=0.5
name_path="k${min_num_neighrbors}_wh${weight_radius_h}ww${weight_radius_w}jh${j_h}jw${j_w}_maxwidth${max_width_line}_minradio${minradio}"
dir_dest=${path}/graph_${name_path}
dir_dest_processed=${path}/graphs_preprocessed/graph_${name_path}
cd ..
python create_graphs.py --dir ${data_path} --dir_dest ${dir_dest} --minradio ${minradio} --min_num_neighbours ${min_num_neighrbors} \
--weight_radius_w ${weight_radius_w} --weight_radius_h ${weight_radius_h} --mult_punct_w ${j_h} --mult_punct_h ${j_w} \
--max_width_line ${max_width_line} --hisclima "true" --do_spans "false" --data_path_tablelabel ""  --reading_order "false"
python preprocess.py ${dir_dest} ${dir_dest_processed}

## ROWS
minradio=0.1
min_num_neighrbors=10
weight_radius_w=4
weight_radius_h=0 
j_h=1
j_w=1
name_path="k${min_num_neighrbors}_wh${weight_radius_h}ww${weight_radius_w}jh${j_h}jw${j_w}_maxwidth${max_width_line}_minradio${minradio}"
dir_dest=${path}/graph_${name_path}
dir_dest_processed=${path}/graphs_preprocessed/graph_${name_path}
python create_graphs.py --dir ${data_path} --dir_dest ${dir_dest} --minradio ${minradio} --min_num_neighbours ${min_num_neighrbors} \
--weight_radius_w ${weight_radius_w} --weight_radius_h ${weight_radius_h} --mult_punct_w ${j_h} --mult_punct_h ${j_w} \
--max_width_line ${max_width_line} --hisclima "true" --do_spans "false" --data_path_tablelabel ""  --reading_order "false"
python preprocess.py ${dir_dest} ${dir_dest_processed}


## COLS
minradio=0.1
min_num_neighrbors=10
weight_radius_h=4
weight_radius_w=0
j_h=1
j_w=1
name_path="k${min_num_neighrbors}_wh${weight_radius_h}ww${weight_radius_w}jh${j_h}jw${j_w}_maxwidth${max_width_line}_minradio${minradio}"
dir_dest=${path}/graph_${name_path}
dir_dest_processed=${path}/graphs_preprocessed/graph_${name_path}
python create_graphs.py --dir ${data_path} --dir_dest ${dir_dest} --minradio ${minradio} --min_num_neighbours ${min_num_neighrbors} \
--weight_radius_w ${weight_radius_w} --weight_radius_h ${weight_radius_h} --mult_punct_w ${j_h} --mult_punct_h ${j_w} \
--max_width_line ${max_width_line} --hisclima "true" --do_spans "false" --data_path_tablelabel ""  --reading_order "false"
python preprocess.py ${dir_dest} ${dir_dest_processed}

cd launchers
