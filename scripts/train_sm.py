import os
import argparse
import boto3
from botocore.exceptions import NoCredentialsError

import nemo_run as run
from huggingface_hub import snapshot_download
from nemo.collections import llm
from nemo.collections.diffusion.models.model import DiT7BConfig, DiT14BConfig
from nemo.collections.diffusion.train import pretrain, videofolder_datamodule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


def cosmos_diffusion_finetune(factory: str, max_steps: int, lr: float, data_path: str, tensor_model_parallel_size: int):
    # Choose model configuration
    if factory == 'cosmos_diffusion_7b_text2world_finetune':
        model_cfg = DiT7BConfig
    elif factory == 'cosmos_diffusion_14b_text2world_finetune':
        model_cfg = DiT14BConfig

    # Model setup
    recipe = pretrain()
    recipe.model.config = run.Config(model_cfg)

    # Trainer setup
    recipe.trainer.max_steps = max_steps
    recipe.optim.config.lr = lr

    # Tensor / Sequence parallelism
    recipe.trainer.strategy.tensor_model_parallel_size = tensor_model_parallel_size
    recipe.trainer.strategy.sequence_parallel = True
    recipe.trainer.strategy.ckpt_async_save = False

    # FSDP
    recipe.trainer.strategy.ddp.with_megatron_fsdp_code_path = True
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "MODEL_AND_OPTIMIZER_STATES"
    recipe.trainer.strategy.ddp.overlap_param_gather = True
    recipe.trainer.strategy.ddp.overlap_grad_reduce = True
    recipe.model.config.use_cpu_initialization = True

    # Activation Checkpointing (for 14B model only)
    if factory == 'cosmos_diffusion_14b_text2world_finetune':
        recipe.model.config.recompute_granularity = "full"
        recipe.model.config.recompute_method = "uniform"
        recipe.model.config.recompute_num_layers = 1

    # Data setup
    recipe.data = videofolder_datamodule()
    recipe.data.path = data_path

    # Checkpoint load
    snapshot_id = f"nvidia/Cosmos-1.0-Diffusion-{factory.upper()}-Text2World"
    recipe.resume.restore_config = run.Config(RestoreConfig, load_artifacts=False)
    recipe.resume.restore_config.path = os.path.join(
        snapshot_download(snapshot_id, allow_patterns=["nemo/*"]), "nemo"
    )
    recipe.resume.resume_if_exists = False

    # Directory to save checkpoints / logs
    recipe.log.log_dir = f"nemo_experiments/{factory}"

    return recipe

def download_from_s3(s3_path, local_dir):
    """Helper function to download files from S3 to a local directory"""
    # Extract the S3 bucket and key from the S3 path
    s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)
    # Create an S3 client
    s3_client = boto3.client('s3')
    # Check if the local directory exists, if not, create it
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    # Try downloading the S3 file(s)
    try:
        # List objects in the S3 path to handle files within directories
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_key)
        for content in response.get('Contents', []):
            key = content['Key']
            local_file_path = os.path.join(local_dir, os.path.relpath(key, s3_key))
            local_file_dir = os.path.dirname(local_file_path)
            # Ensure the local directory structure exists
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            # Download the file
            s3_client.download_file(s3_bucket, key, local_file_path)
            print(f"Downloaded {key} to {local_file_path}")
    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_models(args):
    """Function to download models from the S3 bucket to the local directory"""
    print(f"Downloading models from {args.s3_model} to {args.model_local_dir}...")
    print(os.system("df -h"))
    download_from_s3(f"{args.s3_model}/models--google-t5--t5-11b", f"{args.model_local_dir}/models--google-t5--t5-11b")
    # download_from_s3(f"{args.s3_model}/models--nvidia--Cosmos-1.0-Guardrail", f"{args.model_local_dir}/models--nvidia--Cosmos-1.0-Guardrail")
    print(os.system("df -h"))
    download_from_s3(f"{args.s3_model}/models--nvidia--Cosmos-1.0-Prompt-Upsampler-12B-Text2World", f"{args.model_local_dir}/models--nvidia--Cosmos-1.0-Prompt-Upsampler-12B-Text2World")
    print(os.system("df -h"))
    download_from_s3(f"{args.s3_model}/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8", f"{args.model_local_dir}/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8")
    print(os.system("df -h"))
    if args.factory == "cosmos_diffusion_7b_text2world_finetune":
        download_from_s3(f"{args.s3_model}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World", f"{args.model_local_dir}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World")
    elif args.factory == "cosmos_diffusion_14b_text2world_finetune":
        download_from_s3(f"{args.s3_model}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World", f"{args.model_local_dir}/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World")
    # ## Run the following command to download the models
    # print("Run the following command to download the models")
    # os.system("python /opt/ml/code/cosmos1/models/diffusion/nemo/download_diffusion_nemo.py")

def download_dataset(args):
    """Function to download the dataset from the S3 bucket to the local directory"""
    print(f"Downloading dataset from {args.s3_dataset} to {args.dataset_local_dir}...")
    download_from_s3(args.s3_dataset, args.dataset_local_dir)
    # ## Run the following command to download the sample videos used for post-training
    # print("Run the following command to download the sample videos used for post-training")
    # os.system("huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir /opt/ml/code/cosmos1/models/diffusion/assets/ --include '*.mp4*'")

def postprocess_dataset(args):
    ## Run the following command to preprocess the data
    print("Run the following command to preprocess the data")
    os.system(f"python /opt/ml/code/cosmos1/models/diffusion/nemo/post_training/prepare_dataset.py --tokenizer_dir {args.s3_model}/models--nvidia--Cosmos-1.0-Tokenizer-CV8x8x8 --dataset_path {args.dataset_local_dir} --output_path {os.environ.get('CACHED_DATA')} --prompt 'A video of sks teal robot.' --height 480 --width 640 --num_chunks 5")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--factory", type=str, choices=["cosmos_diffusion_7b_text2world_finetune", "cosmos_diffusion_14b_text2world_finetune"], required=True, help="Model configuration to use (7b or 14b)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--tensor-model-parallel-size", type=int, required=True, help="Tensor model parallel size")
    
    parser.add_argument("--s3-dataset", type=str, default="s3://tri-ml-sandbox-16011-us-west-2-datasets/cosmos-1/datasets/Cosmos-NeMo-Assets", help="S3 Path to the dataset")
    parser.add_argument("--dataset-local-dir", type=str, default="/opt/ml/code/cosmos1/models/diffusion/assets", help="Local directory to store dataset")
    args = parser.parse_args()
    parser.add_argument("--s3-model", type=str, default="s3://tri-ml-sandbox-16011-us-west-2-datasets/cosmos-1/checkpoints/Cosmos-NeMo-Assets/default", help="S3 Path to the dataset")
    parser.add_argument("--model-local-dir", type=str, default="/opt/ml/input/data/training", help="Local directory to store dataset")
    
    args = parser.parse_args()
    
    download_models(args) # download the models
    download_dataset(args) # download the sample videos used for post-training
    postprocess_dataset(args) # postprocess the dataset


    # Call the finetuning function
    cosmos_diffusion_finetune(
        factory=args.factory,
        max_steps=args.max_steps,
        lr=args.lr,
        data_path=args.data_path,
        tensor_model_parallel_size=args.tensor_model_parallel_size
    )

if __name__ == "__main__":
    main()
