CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py  --backbone vgg16_bn --batch_size 128 \
--num_verb_embed 504 --num_role_queries 190 --gt_role_queries --output_dir "verb_dual_transformer_fcn/" \
--dist_url file:///home/hyeonjun/detr/dist/distributed_verb_dual_transformer_fcn \
| tee verb_dual_transformer_fcn.log