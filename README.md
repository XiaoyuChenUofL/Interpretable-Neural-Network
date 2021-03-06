# Interpretable-Neural-Network
- Source code for paper "INN: An Interpretable Neural Network for AI Incubation in Manufacturing" accepted by ACM Transactions on Intelligent Systems and Technology.
- *Authors*: Xiaoyu Chen, Yingyan Zeng, Sungku Kang, Ran Jin

## Dependencies
- `tensorflow>=2.0`
- `tqdm`
- `keras_bert`
- `keras_radam`
- `numpy`
- `pandas`
- `matplotlib`
- `graphviz`
- `scikit-learn`
- `tpot`
- `auto-sklearn` (only support linux OS)

## Simulation Data
- `simu_data/*.npy` generated by `Simulation_generation_WA.py`

## Main Entrance
- `python IntNN_simulation_WA.py` to test the INN model on the simulation data set.
- `simulation_WA_NN.py` contains the INN network specifications.
