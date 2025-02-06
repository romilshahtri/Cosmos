#!/bin/bash
echo "Getting batch_test"
echo "Install sm_batch"
pushd externals/batch_test
pip install boto3
aws configure add-model --service-model file://install//batch-2016-08-10.normal.json --service-name sm_batch
pip install --upgrade pip
pip install install/sagemaker-2.226.2.dev0.tar.gz
pip install nest_asyncio
popd
