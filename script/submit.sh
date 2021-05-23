git checkout parallel_srtr

# titanxp ()
for OPTIM in "Adam" "AdamW" "Adamax" "RMSprop" "SGD";
do
    echo $OPTIM

    NAME="base_${OPTIM}"
    rm -rf log/"titanxp_${NAME}"
    python run_with_submitit.py \
        --backbone resnet50 \
        --partition titanxp --batch_size 16 \
        --optimizer ${OPTIM} \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --weight_decay 0.0 \
        --clip_max_norm 0.1 \
        --verb_loss_coef 1 \
        --noun_loss_coef 1 \
        --epochs 15 \
        --num_workers 4 \
        --job_dir log/"titanxp_${NAME}"

    # 2080ti 8
    NAME="base_${OPTIM}"
    rm -rf log/"2080ti_${NAME}"
    python run_with_submitit.py \
        --backbone resnet50 \
        --partition 2080ti --batch_size 8 \
    --optimizer ${OPTIM} \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --weight_decay 0.0 \
        --clip_max_norm 0.1 \
        --verb_loss_coef 1 \
        --noun_loss_coef 1 \
        --epochs 15 \
        --num_workers 4 \
        --job_dir log/"2080ti_${NAME}"
done