import argparse
import time
import os
import subprocess
from datetime import datetime
from pathlib import Path

import boto3
from botocore.config import Config
from sagemaker import Session as sm_Session
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import FileSystemInput

try:
    from sagemaker.batch_queueing.queue import Queue
    print(f"Loading SageMaker batch queueing.")
    is_sm_queue = True
except Exception as e:
    print(f"Could not load SageMaker batch queueing: {e}.")
    print(f"Trying to install")
    os.system("bash scripts/setup_sm_batch.sh")
    from sagemaker.batch_queueing.queue import Queue
    print(f"Loading SageMaker batch queueing.")
    is_sm_queue = True
except:
    is_sm_queue = False

is_sm_queue = True

NAME = "cosmos"
INSTANCE_MAPPER = {
    "p4d": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge"
}


def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(user, instance_type, version="251", build_type="full", profile="default", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent.parent
    if instance_type in ("p4d", "p4de", "p5"):
        algorithm_name = f"{user}-{NAME}-{version}"
        dockerfile_base = docker_dir / f"docker/Dockerfile_sm_{version}"
        dockerfile_update = docker_dir / "docker/Dockerfile_update"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if build_type is None:
        return fullname

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    if build_type == "full":
        print("Building container")
        commands = [
            # Log in to Sagemaker account to get image.
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            (
                f"aws --region {region} --profile {profile} ecr describe-repositories --repository-names {algorithm_name} || "
                f"aws --region {region} --profile {profile} ecr create-repository --repository-name {algorithm_name}"
            ),
        ]
    elif build_type == "update":
        print("Updating container")
        commands = [
            f"docker build -f {dockerfile_update} --build-arg BASE_DOCKER={algorithm_name} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]
    else:
        raise ValueError(f"Unknown build_type: {build_type}")

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-type", choices=["full", "update"], help="Build image from scratch")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--user", required=True, help="User name")
    
    parser.add_argument("--cfg_path", help="Location of config file", default="configs/post-train.yaml")

    # AWS profile args
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--profile", default="default", help="AWS profile to use")
    parser.add_argument("--arn", default=None, help="If None, reads from SAGEMAKER_ARN env var")
    parser.add_argument(
        "--s3-remote-sync", default=None, help="S3 path to sync to. If none, reads from S3_REMOTE_SYNC env var"
    )

    # Instance args
    parser.add_argument("--instance-count", default=1, type=int, help="Number of instances")
    parser.add_argument("--instance-type", default="p4de", choices=list(INSTANCE_MAPPER.keys()))
    parser.add_argument("--version", default="251", type=str, help="Choose from: (230, 251)")
    parser.add_argument("--spot-instance", action="store_true")

    parser.add_argument('--base-job-name', type=str)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='s3')

    # Jobs Queue
    parser.add_argument("--fss-identifier", default="default", help="Share identifier for FSS queue")
    parser.add_argument("--priority", default=1, type=int, help="Priority of the job")

    args = parser.parse_args()
    main_after_setup_move(args)


def main_after_setup_move(args):
    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        args.arn = os.environ["SAGEMAKER_ARN"]
    
    if args.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        args.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]

    image_uri = get_image(
        args.user,
        args.instance_type,
        args.version,
        region=args.region,
        build_type=args.build_type,
        profile=args.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sm_Session(
        boto_session=boto3.session.Session(
            region_name=args.region,
            profile_name=args.profile
        )
    )

    if args.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()

    role = args.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split(["/"][-1])
    
    boto3_config = Config(
        region_name = args.region
    )

    # client = boto3.client("sts", config=boto3_config)
    # account = client.get_caller_identity()["Account"]
    account = '124224456861' # client.get_caller_identity()["Account"]
    # account = subprocess.getoutput(
    #     f"aws --region {args.region} --profile {args.profile} sts get-caller-identity --query Account --output text"
    # )

    # session = boto3.session.Session()
    session = boto3.session.Session(region_name=args.region)
    region = session.region_name

    ##########
    # Configure the training
    ##########
    base_job_name = args.base_job_name # f"{args.user.replace('.', '-')}-{NAME}"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
        job_name = "-".join([base, date_str])
        return job_name

    job_name = get_job_name(base_job_name)

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)
    
    tags = [
        {
            "Key": "tri.project", 
            "Value": "MM:PJ-0077",
        },
        {
            "Key": "tri.owner.email",
            "Value": "romil.shah.ctr@tri.global",
        },
    ]

    max_run = 5 * 24 * 60 * 60
    max_wait = 5 * 24 * 60 * 60 if args.spot_instance else None
    keep_alive_period_in_seconds = 60 * 60 if not args.spot_instance else None  

    entry_point = "scripts/train_sm.py"
    instance_type = "local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type]
    instance_count = args.instance_count
    train_use_spot_instances = args.spot_instance
    
    checkpoint_s3_uri = os.path.join(
        f's3://tri-ml-sandbox-16011-us-west-2-datasets/cosmos-1/output-checkpoints-{args.user}', job_name)
    checkpoint_s3_uri = None if args.local else checkpoint_s3_uri

    checkpoint_local_path = "/opt/ml/checkpoints"
    checkpoint_local_path = None if args.local else checkpoint_local_path 

    if args.input_source == 'local':
        input_mode = 'File'
        train_fs = 'file:///data'
    elif args.input_source == 'lustre':
        input_mode = 'File'
        file_system_id = 'fs-01158facd398d97f2'
        directory_path = '/gpd7vbev'
        train_fs = FileSystemInput(
            file_system_id=file_system_id,
            file_system_type='FSxLustre',
            directory_path=directory_path,
            file_system_access_mode='ro'
        )
    elif args.input_source == 's3':
        input_mode = 'FastFile'
        train_fs = 's3://tri-ml-sandbox-16011-us-west-2-datasets/cosmos-1/datasets-processed/HxW-480x640-chunks-50'
    else:
        raise ValueError(f'Invalid input source {args.input_source}')
    
    inputs = {
        'training': train_fs,
    }
    hyperparameters = {
        # "factory": f"{args.factory}",
        # "data-path": f"{args.data_path}",
        # "lr": f"{args.optim_config_lr}",
        # "max-steps": f"{args.trainer_max_steps}",
        # "tensor-model-parallel-size": f"{args.tensor_model_parallel_size}"
    }
    distribution={
        "torch_distributed": {
            "enabled": True,
        }
    }
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1",
        "WANDB_API_KEY": os.environ.get('WANDB_API_KEY', None),
        # "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
        # "TORCH_CPP_LOG_LEVEL": "INFO",
        "CONFIG_FILE": args.cfg_path,
        "FI_EFA_FORK_SAFE": "1",
        "NVTE_FUSED_ATTN": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }

    security_group_ids = {
        'us-east-1': [
            'sg-0afb9fb0e79a54061'
        ],
        'us-west-2': [
            'sg-029d1d476bc087e31',
        ],
    }
    subnets = {
        'us-east-1': [
            'subnet-07bf42d7c9cb929e4',
            'subnet-0e260ba29726b9fbb',
        ],
        'us-west-2': [
            'subnet-0610f766a4cd5cdae', 
            'subnet-029adfb9e225d68f8',
            'subnet-01cc1bfeaf20155b5',
        ]
    }

    print()
    print(security_group_ids[region])
    print(subnets[region])
    print()

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'SM Queue:                       {is_sm_queue}-{args.priority}-{args.fss_identifier}')
    print(f'AWS region:                     {region}')
    print(f'AWS profile:                    {args.profile}')
    print(f'AWS account:                    {account}')
    print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    print(f'Configuration file:             {hyperparameters}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print(f'Instance type:                  {instance_type}')
    print('#############################################################')
    print()
    print()

    estimator = PyTorch(
        entry_point=entry_point,
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        train_use_spot_instances=train_use_spot_instances,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=output_s3,
        distribution=distribution,
        max_run=max_run,
        max_wait=max_wait,
        input_mode=input_mode,
        debugger_hook_config=False,
        environment=environment,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        tags=tags,
        subnets=subnets[region],
        security_group_ids=security_group_ids[region],
        train_volume_size=1000,
    )

    if is_sm_queue:
        queue_name  = f"fss-{INSTANCE_MAPPER[args.instance_type]}-{args.region}".replace('.', '-')
        queue = Queue(queue_name)
        print(f"Starting training job on queue: {queue.queue_name}")

        queued_jobs = queue.map(
            estimator,
            inputs=[inputs['training']],
            job_names=[job_name],
            priority=args.priority,
            share_identifier=args.fss_identifier,
        )
        print(f"Queued jobs: {queued_jobs}")
    else:
        estimator.fit(inputs=inputs)


if __name__ == "__main__":
    main()
