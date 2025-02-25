
pkill -f eval_arx5

python scripts/eval_arx5.py \
    -i ../diffusion_policy/data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5-reformat/checkpoints/latest.ckpt \
    -o data/outputs_v4/umi_fewdata_lang/data500_umi_mix_dataaug-dataaugtype3-lr1e-5-5-reformat \
    --policy_port 8768 --frequency 5 --steps_per_inference 4 \
    --task_name cup