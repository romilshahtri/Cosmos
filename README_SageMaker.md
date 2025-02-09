# SageMaker for Cosmos
- Utilizing the steps mentioned in [here](https://github.com/romilshahtri/Cosmos/blob/main/cosmos1/models/diffusion/nemo/post_training/README.md#1-prepare-a-dataset) to run a SM training job.

## Running local docker
```
$ cd ~/Cosmos
$ git submodule update --init --recursive
$ export HF_TOKEN=(huggingface token)
$ make docker-interactive
(docker) $ export RAW_DATA="cosmos1/models/diffusion/assets/nemo_diffusion_example_data"
(docker) $ export CACHED_DATA="./cached_data" && mkdir -p $CACHED_DATA
(docker) $ python cosmos1/models/diffusion/nemo/download_diffusion_nemo.py
(docker) $ huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir cosmos1/models/diffusion/assets/ --include "*.mp4*"
(docker) $ export PYTHONPATH=$PYTHONPATH:/workspace/Cosmos
(docker) $ python cosmos1/models/diffusion/nemo/post_training/prepare_dataset.py --dataset_path $RAW_DATA --output_path $CACHED_DATA --prompt "A video of sks teal robot." --height 480 --width 640 --num_chunks 500
(docker) $ NVTE_FUSED_ATTN=0 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node=4 cosmos1/models/diffusion/nemo/post_training/general.py \
    --yes \
    --factory cosmos_diffusion_7b_text2world_finetune \
    data.path=$CACHED_DATA \
    trainer.max_steps=1000 \
    optim.config.lr=1e-6 \
    trainer.strategy.tensor_model_parallel_size=4
```

* Note: Cosmos requires unreleased versions of `Megatron-LM` and `NeMo` which can be found [here](externals/README.md) and added to submodules here. SageMaker requires `batch_test` which can be found from [here](https://github.com/TRI-ML/batch_test.git).

## Run SageMaker SM Job:
- Make changes to `configs/post-train.yaml` as per requirements
```
$ export HF_TOKEN=(huggingface token)
$ python3 scripts/launch_sagemaker_train.py --user romilshah --profile default --build-type update --cfg_path configs/post-train.yaml --region us-west-2 --arn arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess --s3-remote-sync s3://tri-ml-sandbox-16011-us-west-2-datasets/sagemaker/s3_remote_sync/ --version 251 --instance-type p4d --instance-count 1 --base-job-name romilshah-cosmos
```
