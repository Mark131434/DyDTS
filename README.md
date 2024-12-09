# Dynamic Topic Boundary Refinement (DTBR) for Dialogue Topic Segmentation

## Installation

To use this repository, clone the repository and install the required dependencies:

### Clone the repository

```bash
git clone https://github.com/your-username/ATBR-DTS.git
cd ATBR-DTS
```

### Install dependencies

We recommend using a virtual environment (e.g., venv, conda) to install the dependencies.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Description

DialSeg711 is a real-world dataset consisting of 711 English dialogues, sourced from MultiWOZ and KVRET. It exhibits an average of 4.9 topic segments and 5.6 utterances per segment. Doc2Dial is a synthetic dataset comprising over 4,100 English conversations grounded in 450+ documents across four domains. It presents an average of 3.7 topic segments and 3.5 utterances per segment.

#### Details of Dialogue Datasets

| Datasets                         | DialSeg711  | Doc2Dial  |
|-----------------------------------|-------------|------------|
| #samples                          | 711         | 4100       |
| #Avg. Topic Segments/Dialogue     | 4.9         | 3.7        |
| #Avg. Utterances/Topic Segments   | 3.7         | 3.5        |


### 2. Data Preparation

Prepare your dialogue data in the required format. The dataset should consist of a series of utterances, where each dialogue is represented as a sequence of text. The dataset is available right [here](https://drive.google.com/drive/folders/11HSQWJR8qurD8K_ezgo6HqtcULl18UJq?usp=sharing)

```bash
python data_prepare.py --data_dir data/dialseg711 --file_name 711.pkl --output_dir processed_711_data --model_name  sup-simcse-bert-base-uncased
```

### 3. Training

To train the model on your dataset:

```bash
python train.py --data_dir processed_711_data --model_name sup-simcse-bert-base-uncased --output_dir model_711_trained
```

### 4. Evaluation

To evaluate the model's performance, we provide evaluation scripts and [model](https://drive.google.com/drive/folders/16JPkKNrKHrKYxr6okOyVO0F8w9fI0J6-) for calculating various metrics, such as Pk and WD, based on the segmented output:

```bash
python inference.py --data_dir data/dialseg711 --model_name sup-simcse-bert-base-uncased --output_dir model_711
```

## Performance

Our method outperforms existing baseline models on the DialSeg711 and Doc2Dial datasets, achieving state-of-the-art results. For detailed performance metrics, please refer to the Results section in the original paper.

## Contributing

We welcome contributions to improve the ATBR method. Feel free to fork the repository and submit pull requests for:

- Bug fixes
- Feature enhancements
- Improvements to the documentation

## Contact

For any questions, feel free to open an issue or contact the project maintainers.

