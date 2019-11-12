# Hurricane-nas
## How to use:

* Evaluate the pretrained model:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12389 dali_evaluate.py -j 4 --hardware vpu`

* train the model from scratch:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12390 dali_train.py -j 4 --filename test --hardware cpu`

## Visualize the architectures in Table 2:

![Image description](/images/bestmodels.png)

