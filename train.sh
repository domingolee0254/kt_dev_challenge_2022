python code/train.py -dd '/home/work/team01/food-kt/data' -sd '/home/work/team01/food-kt/ckpt' \
    -m 'tf_efficientnet_b4_ns' -is 384 -av 0 -e 100 -we 10 -bs 32 -nw 4 \
    -l 'smoothing_ce' -ls  0.5 -ot 'adamw' -lr 1e-3 -sc 'cos_base' -wd 0.05 \
    -cm True -mp 0.5 -cms 51 \
    --wandb False --amp True