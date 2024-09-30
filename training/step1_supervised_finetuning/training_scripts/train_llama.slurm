#!/bin/bash
#SBATCH -A ECS24003
#SBATCH --time=01:59:00
#SBATCH -o training_log_output/3.1_8b-a100-%J.o
#SBATCH -e training_log_output/3.1_8b-a100-%J.e
#SBATCH -N 4
#SBATCH -n 12
#SBATCH -p gpu-a100
#SBATCH --mail-user=sli@tacc.utexas.edu
#SBATCH --mail-type=all

mv /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/train.json \
/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/ds_train_ascii.json

mv /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/eval.json \
/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/ds_eval_ascii.json

mv /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/tk23train.json \
/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/train.json

mv /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/tk23eval.json \
/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/data/eval.json

module load cuda/12.0
export CUDA_HOME=/opt/apps/cuda/12.0
export NCCL_HOME=/opt/apps/cuda12_0/nccl/2.17.1/
export CUDNN_HOME=/opt/apps/cuda12_0/cudnn/8.8.1 
export LD_LIBRARY_PATH=/usr/lib64:/opt/apps/cuda/12.0/lib64:/opt/intel/compilers_and_libraries_2020.1.217/linux/bin/intel64:/opt/apps/intel19/python3/3.9.7/lib:/opt/apps/pmix/3.2.3/lib:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/libfabric/lib:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib/release:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib:/opt/intel/debugger_2020/libipt/intel64/lib:/opt/intel/compilers_and_libraries_2020.1.217/linux/daal/lib/intel64_lin:/opt/intel/compilers_and_libraries_2020.1.217/linux/tbb/lib/intel64_lin/gcc4.8:/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries_2020.1.217/linux/ipp/lib/intel64:/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin:/opt/apps/gcc/9.4.0/lib64:/opt/apps/gcc/9.4.0/lib

source /work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/venv/bin/activate

export PATH=/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/pdsh/bin:/:$PATH
export HF_HOME=/tmp/huggingface_cache
HUGGINGFACE_TOKEN_file="/work/07980/sli4/ls6/code/DeepSpeedChat/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/huggingface_token"
HUGGINGFACE_TOKEN=$( cat $HUGGINGFACE_TOKEN_file )
srun huggingface-cli login --token $HUGGINGFACE_TOKEN

model_name_or_path="/scratch/07980/sli4/data/gpt/meta-llama/Meta-Llama-3.1-8B-Instruct"
num_train_epochs=4
lora_dim=0
max_seq_len=1024
per_device_train_batch_size=6

current_time=$(date "+%Y%m%d-%H%M%S")

OUTPUT=$1
ZERO_STAGE=$2
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
if [ "$OUTPUT" == "" ]; then
   OUTPUT="./output_Llama-3.1-8B-Instruct_epoch$num_train_epochs""_train_batch_size$per_device_train_batch_size""_seq$max_seq_len""_lora$lora_dim""_zero$ZERO_STAGE""_$current_time"
fi
mkdir -p $OUTPUT

#ibrun python3 training_scripts/dump_hostfile.py --output_dir ./

NODEFILE=/tmp/hostfile
scontrol show hostnames  > $NODEFILE
cat $NODEFILE
NNODES=$(< $NODEFILE wc -l)
GPU_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
python3 training_scripts/dumpHostfile.py --slots $GPU_PER_NODE
cat "./hostfile"


echo $PYTHONPATH
which python
deepspeed --hostfile ./hostfile main_cp.py \
   --sft_only_data_path local/jsonfile  \
   --model_name_or_path $model_name_or_path \
   --per_device_train_batch_size $per_device_train_batch_size \
   --per_device_eval_batch_size 4 \
   --max_seq_len $max_seq_len \
   --learning_rate "9.65e-6" \
   --weight_decay 0. \
   --num_train_epochs $num_train_epochs  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim $lora_dim \
   --lora_module_name "layers." \
   --output_dir $OUTPUT \
   --print_loss \
   --save_interval 2000 \
   --gradient_checkpointing \
   --enable_tensorboard \
   --tensorboard_path "${OUTPUT}/step1_tensorboard" \
   --load_dir ./output_Llama-3.1-8B-Instruct_epoch4_train_batch_size6_seq1024_lora0_zero3_20240923-164356 \
   --ckpt_id step_6000 \
   |& tee $OUTPUT/training.log
