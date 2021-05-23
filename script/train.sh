git checkout parallel_srtr

# sandia (RTX3090 24GB x1)
NAME="base"
rm -f log/${NAME}.log dist/${NAME}
rm -rf log/${NAME}

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --backbone resnet50 \
    --batch_size 32 \
    --optimizer Adam \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --weight_decay 0.0 \
    --clip_max_norm 0.1 \
    --verb_loss_coef 1 \
    --noun_loss_coef 1 \
    --epochs 50 \
    --num_workers 4 \
    --dist_url file://${PWD}/dist/${NAME} \
    --output_dir log/${NAME} \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%m:%d-%T)" "$line"; done \
    | tee -a log/${NAME}.log