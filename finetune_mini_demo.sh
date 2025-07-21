# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
export CUDA_VISIBLE_DEVICES='0,1';

export LAYERNORM_TYPE=fast_layernorm;
export USE_DEEPSPEED_EVO_ATTENTION=false;
export WANDB_DISABLE_STATS=true;
export CUTLASS_PATH="/mnt/rna01/liuzhiyuan/zyliu/cutlass";
# export TORCH_LOGS="+dynamo";
# export TORCHDYNAMO_VERBOSE=1;
# wget -P ./bin/ckpt https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt
# wget -P ./bin/ckpt https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_mini_default_v0.5.0.pt


checkpoint_path="./bin/ckpt/protenix_mini_default_v0.5.0.pt"

# 
PYTHONPATH=./ torchrun --nproc_per_node=2 ./runner/train.py \
--model_name "protenix_mini_default_v0.5.0" \
--run_name protenix_mini_finetune \
--seed 42 \
--base_dir ./output \
--dtype bf16 \
--project protenix \
--use_wandb true \
--diffusion_batch_size 32 \
--eval_interval 50 \
--log_interval 1 \
--checkpoint_interval 50 \
--blocks_per_ckpt none \
--ema_decay 0.999 \
--train_crop_size 256 \
--max_steps 15000 \
--warmup_steps 1000 \
--lr 0.001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ${checkpoint_path} \
--load_ema_checkpoint_path ${checkpoint_path} \
--data.batch_size 2 \
--data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
--data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list dataset/train_pdbs.txt \
--data.test_sets recentPDB_1536_sample384_0925