# ndcn
Neural Dynamics on Complex Networks

## Install libs:
```
conda create --name ndcn 
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch 
conda install networkx 
conda install matplotlib 
conda install scipy
conda install scikit-learn
```
<!-- --network community --dump --sampled_time equal --baseline ndcn --gpu -1 --weight_decay 1e-4 -->

## Running examples
### NDCN for mutualistic interation dynamics 
Python files: mutualistic_dynamics.py
```
python mutualistic_dynamics.py  --T 5 --network grid --dump --sampled_time irregular --baseline ndcn --viz --gpu -1 --weight_decay 1e-2
```
--network *** for underlining graph with choices=['grid', 'random', 'power_law', 'small_world', 'community']<br /> 
--sampled_time ** for irregularlly-sampled graph dynamics or regularly sampled ones with choices=['irregular', 'equal']<br /> 
--baseline ** chooses any model from choices=['ndcn', 'no_embed', 'no_control', 'no_graph', 'lstm_gnn', 'rnn_gnn', 'gru_gnn']<br /> 
Please refer to the code for the detailed parameter choices


### Similar commands for heat-diffusion dynamics or gene regulatory dynamics
Python files: heat_dynamics.py and gene_dynamics.py
```
python heat_dynamics.py  --T 5 --network grid --dump --sampled_time irregular --baseline ndcn --viz --gpu -1 --weight_decay 1e-3
python gene_dynamics.py  --T 5 --network grid --dump --sampled_time irregular --baseline ndcn --viz --gpu -1 --weight_decay 1e-4
```
