# README

## Requirements

- **Python**: 3.11.7
- **PyTorch**: 2.4.1
- **TensorBoard**: 2.18.0

## Steps

### 1. Generate Data
Run the following command to generate data:

```bash
python3 data_generation.py <number_of_samples> <training_split>
```

(Note: <number_of_samples> and <training_split> are optional. Default values are set to 10000 and 0.8 respectively)

### 2. Train network

```bash
python3 train.py
```

Network is trained on 10 epochs and a batch size of 1024.

### 3. Test network

```bash
python3 test.py
```
### 4. View output in TensorBoard

Execute the following command in your terminal to generate the training/testing loss values as a function of the epoch:

```bash
tensorboard --logdir=runs
```

Connect to TensorBoard by going to http://localhost:6006/ in your browser to view the results.