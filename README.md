# DAIR-BioMed
This reporepository holds DAIR-BioMed, an open-source toolkit for multi-modal representation learning in AI-driven biomedical research. We focus on multi-modal information, i.e. knowledge graphs and biomedical texts for drugs, proteins and single cells and wide applications including drug-target interaction prediction, molecular property prediction, cell-type prediction, molecule-text retrieval, molecule-text generation, and drug-response prediction. Researchers can compose a large number of deep learning models including  LLMs like BioMedGPT and CellLM to facilitate downstream tasks. Easy-to-use APIs and commands are provided to accelerate life science research.

### News!

- [04/23] 🔥The pre-alpha BioMedGPT model and Open-DAIR-BioMed are available!

### Features

- **3 different modalities for drugs, proteins and cell-lines**,  i.e. molecular structure, knowledge graphs and biomedical texts. We present a unified and easy-to-use pipeline to load, process and fuse the multi-modal information .

- **15+ deep learning models** for the multi-modal information of  ranging from CNN, GNN to Transformers. We present the pre-alpha version of **BioMedGPT**, a pre-trained multi-modal molecular fundation model that assosiates the 2D molecular graph and texts with 1.6B parameters. We also present **CellLM**, a single cell foundation model with 50M parameters.
- **10+ downstream tasks** ranging from AIDD tasks like drug-target interaction prediction and molecule property training to cross-modal tasks like molecule captioning and text-based molecule generation.  
- **20+ datasets** that are most popular in AI-driven biomedical research. Reproductible benchmarks with abundant model combinations and comprehensive evaluations are provided.
- **3+ knowledge graphs** with extensive domain expertise. We present **BMKGv1**, a knowledge graph containing 6, 917 drugs, 19, 992 proteins and 2,223,850 relationships with text descriptions. We provide  APIs to load and process these graphs and link drugs or proteins to these graphs based on structural information.

### Installation

To support minimum usage of DAIR-BioMed, run the following command:

```bash
conda create -n DAIR-BioMed python=3.8
conda activate DAIR-BioMed
conda install -c conda-forge rdkit
pip install torch

# torch_geometric, please follow instructions in https://github.com/pyg-team/pytorch_geometric to install the correct version of pyg
pip install torch_cluster-1.6.0-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.14-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
pip install torch-geometric

pip install transformers 
pip install ogb
```

**Note** that additional packages may be required for specific downstream tasks.

### Quick Start

Here we provide a quick example of training DeepDTA for drug-target interaction prediction on Davis dataset. For more models, datasets, and tasks, please refer to our [scripts](./dair_biomed/scripts) and [documents](./docs).

#### Step1: Data Preparation

Install the Davis dataset [here](https://drive.google.com/drive/folders/1pz4QZEmcZrBU5JAJliyMNvMrBFeXN4SN?usp=sharing) and and run the following:

```
mkdir datasets
cd datasets
mkdir dti
mv [your_path_of_davis] ./dti/davis
```

#### Step2: Training and Evaluation

Run:

```bash
cd ../dair_biomed
bash scripts/dti/train_baseline_regression.sh
```

The results will be like the following (running takes around 40 minutes on an Nividia A100 GPU)

```bash
INFO - __main__ - MSE: 0.2198, Pearson: 0.8529, Spearman: 0.7031, CI: 0.8927, r_m^2: 0.6928
```

### Contaction

As a pre-alpha version release,  we are looking forward to user feedbacks to help us improve our framework. If any questions or suggestions, please open an issue or contact [dair@air.tsinghua.edu.cn](mailto:dair@air.tsinghua.edu.cn).

