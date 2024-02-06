Finetuning Pre-Trained BERT for Named Entity Recognition using RLHF

## Environment Setup
The code is built on Python 3.10 and [Hugging Face Transformers Library](https://github.com/huggingface/transformers) with customized data processor and trainer.
Other package requirements are listed in `requirements.txt`.
You are suggested to run the code in an isolated virtual [conda](https://www.anaconda.com/) environment.
Suppose you have already install conda in your device, you can create a new environment and activate it with
```bash
conda create -n 310 python=3.10
conda activate 310
```
Then, you can install the required packages with
```bash
pip install -r requirements.txt
```

Alternatively, you can also use other Python version manager or virtual environments such as [pyenv](https://github.com/pyenv/pyenv) or [docker](https://www.docker.com/) to you prefer.

## Data
The dataset used for this assignment is a subset of [CoNLL 2003](https://aclanthology.org/W03-0419.pdf), the most famous benchmark dataset for NER.
For this assignment, we subsampled 1000 training data points, and 100 points for validation and test.
The data is already pre-processed for you and stored in the `./data/` directory as `.json` files.

## Run

If you are using a Unix-like system such as Linux or MacOS, you can run the code through the provided `run.sh` file.
You first need to edit `run.sh` to complete your name and GTID, and then run
```bash
./run.sh [GPU ID]
```
for example, 
```bash
./run.sh 0
```
if you want to GPU-0 to accelerate your training.
If you leave `GPU ID` blank, the model will be trained on CPU.

~~For MacOS running on M* chips, running `bash ./run.sh` will automatically take advantage of [mps accelaration](https://developer.apple.com/metal/pytorch/). You can disable this behavior by adding `--no_mps` argument into the Python call in the `sh` file.~~
This feature is deprecated as mps sometimes returns incorrect results.


## Reference

With default hyper-parameters, each training epoch takes roughly 30s to run on Mac M1 CPU, ~~18s on mps,~~ and 2s on Nvidia A5000/RTX4090 GPU.
