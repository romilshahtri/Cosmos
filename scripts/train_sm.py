# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import yaml
import boto3
from botocore.exceptions import NoCredentialsError

import nemo_run as run
from huggingface_hub import snapshot_download
from nemo.collections import llm
from nemo.collections.diffusion.models.model import DiT7BConfig, DiT14BConfig
from nemo.collections.diffusion.train import pretrain, videofolder_datamodule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


config_file = os.environ.get('CONFIG_FILE')
print(f"Loading config file from: {config_file}")
with open(config_file, 'r') as ff:
    params = yaml.safe_load(ff)
print(f"Parameters from config file: {params}")


def download_from_s3(s3_path, local_dir):
    """Helper function to download files from S3 to a local directory"""
    s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)
    s3_client = boto3.client('s3')
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    try:
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_key)
        for content in response.get('Contents', []):
            key = content['Key']
            local_file_path = os.path.join(local_dir, os.path.relpath(key, s3_key))
            local_file_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            s3_client.download_file(s3_bucket, key, local_file_path)
            print(f"Downloaded {key} to {local_file_path}")
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_models():
    """Function to download models from the S3 bucket to the local directory"""
    print(f"Downloading models from {params['s3_model']} to {params['model_local_dir']}...")
    download_from_s3(f"{params['s3_model']}/models--google-t5--t5-11b", f"{params['model_local_dir']}/models--google-t5--t5-11b")
    download_from_s3(f"{params['s3_model']}/models--nvidia--Cosmos-1.0-Prompt-Upsampler-12B-Text2World", f"{params['model_local_dir']}/models--nvidia--Cosmos-1.0-Prompt-Upsampler-12B-Text2World")
    download_from_s3(f"{params['s3_model']}/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8", f"{params['model_local_dir']}/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8")
    if params['factory'] == "cosmos_diffusion_7b_text2world_finetune":
        download_from_s3(f"{params['s3_model']}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World", f"{params['model_local_dir']}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World")
    elif params['factory'] == "cosmos_diffusion_14b_text2world_finetune":
        download_from_s3(f"{params['s3_model']}/models--nvidia--Cosmos-1.0-Diffusion-14B-Text2World", f"{params['model_local_dir']}/models--nvidia--Cosmos-1.0-Diffusion-14B-Text2World")
def download_dataset():
    """Function to download the dataset from the S3 bucket to the local directory"""
    print(f"Downloading dataset from {params['s3_dataset']} to {params['dataset_local_dir']}...")
    download_from_s3(params['s3_dataset'], params['dataset_local_dir'])
def postprocess_dataset():
    ## Run the following command to preprocess the data
    print("Run the following command to preprocess the data")
    os.system(f"python /opt/ml/code/cosmos1/models/diffusion/nemo/post_training/prepare_dataset.py --tokenizer_dir {params['model_local_dir']}/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8 --dataset_path {params['dataset_local_dir']} --output_path {os.environ.get('CACHED_DATA')} --prompt 'A video of sks teal robot.' --height 480 --width 640 --num_chunks 5")
def download_processed_dataset():
    # download_from_s3(params['s3_dataset'], os.environ.get("CACHED_DATA"))
    download_from_s3(params['s3_dataset'], params['dataset_local_dir'])


@run.cli.factory(target=llm.train)
def cosmos_diffusion_7b_text2world_finetune() -> run.Partial:
    print(f"Running finetuning for 7b")
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(DiT7BConfig)

    # Trainer setup
    recipe.trainer.max_steps = params['max_steps']
    recipe.optim.config.lr = params['lr']

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = params['tensor_model_parallel_size']
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = params['data_path']

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        f"{params['model_local_dir']}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World", "nemo"
    )  # path to diffusion model checkpoint
    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_diffusion_7b_text2world_finetune"
    
    return recipe


@run.cli.factory(target=llm.train)
def cosmos_diffusion_14b_text2world_finetune() -> run.Partial:
    print(f"Running finetuning for 14b")
    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(DiT14BConfig)

    # Trainer setup
    recipe.trainer.max_steps = params['max_steps']
    recipe.optim.config.lr = params['lr']

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = params['tensor_model_parallel_size']
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    # Activation Checkpointing
    recipe.model.config.recompute_granularity = "full"
    recipe.model.config.recompute_method = "uniform"
    recipe.model.config.recompute_num_layers = 1

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = params['data_path']

    # Checkpoint load
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        f"{params['model_local_dir']}/models--nvidia--Cosmos-1.0-Diffusion-14B-Text2World", "nemo"
    )  # path to diffusion model checkpoint

    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = "nemo_experiments/cosmos_diffusion_14b_text2world_finetune"
    
    return recipe


if __name__ == "__main__":
    os.system("ls -al /opt/ml/code")
    os.system("ls -al /opt/ml/checkpoints")
    os.system("ls -al /opt/ml/input/data/training")
    download_models()

    sys.argv.extend(["--yes"])
    
    if params['factory'] == "cosmos_diffusion_7b_text2world_finetune":
        run.cli.main(
                     llm.train, 
                     default_factory=cosmos_diffusion_7b_text2world_finetune
                     )
    elif params['factory'] == "cosmos_diffusion_14b_text2world_finetune":
        run.cli.main(
                     llm.train, 
                     default_factory=cosmos_diffusion_14b_text2world_finetune
                     )
