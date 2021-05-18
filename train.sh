python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py  --batch_size 128 --use_verb_decoder --num_verb_embed 504 --num_role_queries 190 --gt_role_queries --output_dir "verb_dual_w_decoder/" | tee verb_dual_w_decoder.log