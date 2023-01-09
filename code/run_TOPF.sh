#!/bin/bash
# v3.2

# # activate venv
# source /data/project/movies_extrct_diff/Python/movie_pred/myenv7/bin/activate

# activate venv
eval "$(conda shell.bash hook)" # this line is necessary to conda activate via bash script
conda activate topfeval


python3.9 run_TOPF.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}

# deactivate
conda deactivate




