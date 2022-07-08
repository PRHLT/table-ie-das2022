path_col= # PATH TO THE RESULTS OF "results.txt" FROM COL DETECTION SYSTEM
path_row= # PATH TO THE RESULTS OF "results.txt" FROM ROW DETECTION SYSTEM
path_header= #PATH TO THE RESULTS OF "results.txt" FROM header DETECTION SYSTEM
nexp=1_das
min_w=0.5 # weight to cut. 0.5 by default

#Path to the HYPCOORD file as explained in main page of this repo.
path_hyp_coords=/data2/jose/projects/TableUnderstanding/information_extraction/hypotesisCoords_GT

use_gt=false
USE_GT_COL=${use_gt}
USE_GT_ROW=${use_gt}
USE_GT_H=${use_gt}
path_save= # PATH TO SAVE THE DATA

python extract_files_PAGE.py --path_hyp_coords ${path_hyp_coords} --path_header ${path_header} --path_row ${path_row} --path_col ${path_col} --path_save ${path_save} --nexp ${nexp} --min_w ${min_w} --USE_GT_COL ${USE_GT_COL} --USE_GT_ROW ${USE_GT_ROW} --USE_GT_H ${USE_GT_H}
# metrica
# exp_name=NER_exp${nexp}_GT
exp_name=NER_exp${nexp}

# Path to the GT file  as explained in main page of this repo.
path_IE_GT=/data/HisClima/HisClimaProd/IE/GT_IE/Jeannette/GToldformat.txt

path_hyp_file=${path_save}/${exp_name}
echo ${path_hyp_file}/hyp_file.txt ${path_IE_GT}
./evaluateIE.o ${path_hyp_file}/hyp_file.txt ${path_IE_GT} # execute evaluateIE.o
mv FN.txt ${path_hyp_file}/FN.txt
mv FP.txt ${path_hyp_file}/FP.txt