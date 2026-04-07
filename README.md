# Wide and Deep Anime Recommender System

A personalized anime recommendation system built with PyTorch, implementing the Wide & Deep Learning architecture on the MyAnimeList 2023 dataset. The model learns user and anime features to generate top-N anime recommendations for each user.



## Overview

This project is based on the Wide & Deep Learning paper by Cheng et al. (Google, 2016), originally developed for the Google Play app recommendation system. The model combines two components trained jointly:

- **Wide component** - a linear model that memorizes direct feature interactions from historical data
- **Deep component** - a multi-layer neural network that generalizes to unseen feature combinations using embeddings

Together, they produce recommendations that are both relevant (memorization) and diverse (generalization).



## Dataset

**Source:** [MyAnimeList Dataset 2023 on Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset)

Three CSV files are downloaded and merged to form the training data:

| File | Description |
|---|---|
| `anime-dataset-2023.csv` | Anime metadata - genres, studios, status, popularity, etc. |
| `users-details-2023.csv` | User profiles - gender, birthday, days watched, etc. |
| `users-score-2023.csv` | User-anime rating interactions |

Merged dataset size: ~6.3 GB. Train/test split: 80/20.



## Model Architecture

```
Wide Features                          Deep Features
[Username, anime_id,                   Categorical: [Gender, Genres, Studios]
 Genres, Studios, Status]              Continuous:  [Age, Days Watched, Dropped,
        |                                            Rewatched, Popularity]
  Linear Layer (-> 1)                       |                    |
        |                            Embedding Layers     (already scaled)
        |                                   |                    |
        |                            Concatenate embeddings + continuous
        |                                   |
        |                          Dense Layer (-> 128) + ReLU + Dropout(0.3)
        |                          Dense Layer (-> 64)  + ReLU + Dropout(0.3)
        |                          Dense Layer (-> 1)
        |                                   |
        +------------ Add -----------------+
                         |
                  Final Prediction
                   (Anime Rating)
```

- Wide layer: `nn.Linear(5 -> 1)`
- Deep embeddings: one `nn.Embedding` layer per categorical feature
- Deep MLP: hidden units `[128, 64]`, ReLU activations, Dropout rate 0.3
- Wide and deep outputs are summed to produce the final rating prediction
- GPU support via automatic CUDA detection



## Pipeline

**1. Data Loading and EDA**
Load all three datasets and inspect shapes, data types, and null/duplicate counts.

**2. Data Cleanup**
- Remove duplicate records
- Drop rows with missing usernames in the user score dataset
- Derive `Age` from `Birthday` using `datetime`; filter out ages below 5 or above 100

**3. Data Visualization**
- User ratings distribution (with `log1p` transformation applied to reduce skew)
- User age distribution

**4. Column Pruning**

| Dataset | Dropped Columns |
|---|---|
| Anime | `Other name`, `Synopsis`, `Image URL` |
| User Details | `Mal ID`, `Birthday`, `Location` |
| User Score | `user_id`, `Anime Title` |

Anime entries with `Status = "Not yet aired"` are also removed.

**5. Dataset Merging**

```
user_score + user_details  -->  merged on Username
merged + anime             -->  merged on anime_id
```

Post-merge nulls are filled: `Gender` with `"NA"`, numeric watch stats with `0`, `Age` with the column mean.

**6. Feature Engineering**

Wide features (label-encoded):
```
Username, anime_id, Genres, Studios, Status
```

Deep features:
- Categorical (label-encoded, then embedded in the model): `Gender`, `Genres`, `Studios`
- Continuous (MinMaxScaled): `Age`, `Days Watched`, `Dropped`, `Rewatched`, `Popularity`

**7. Preprocessing**
- `LabelEncoder` applied to all categorical columns
- `MinMaxScaler` applied to continuous deep features
- Target variable (`rating`) scaled independently with a separate `MinMaxScaler`
- Both scalers saved with `joblib` for reuse at inference time

**8. Training**
- Data converted to `torch.float32` tensors
- `TensorDataset` and `DataLoader` with batch size 256 and shuffling
- Loss: `nn.MSELoss`
- Optimizer: `Adam` with `lr=0.0001` and `weight_decay=1e-4`
- Epochs: 20
- Checkpoints saved during training



## Results

Evaluated on the 20% held-out test set:

| Metric | Value |
|---|---|
| RMSE | 0.127 |
| MAE | 0.087 |
| R2 Score | 0.105 |



## Inference

The `recommend_anime()` function takes a username, runs the model over all anime associated with that user, and returns the top-N entries ranked by predicted rating.

Sample users tested: `10518`, `39745`



## Repository Structure

```
wide-deep-recommender-system/
|
|-- wide_deep_recommender.ipynb                # Full notebook: EDA, preprocessing,
|                                              # training, evaluation, and inference
|
|-- wide_and_deep_model_final.pth              # Saved full model (architecture + weights)
|
|-- wide_and_deep_model_state_dict_final.pth   # Saved model weights only (state dict)
|
|-- README.md
```



## Getting Started

### Prerequisites

```
python >= 3.8
torch
pandas
numpy
scikit-learn
seaborn
matplotlib
joblib
tqdm
```

### Installation

```bash
git clone https://github.com/NehaBais/wide-deep-recommender-system.git
cd wide-deep-recommender-system
pip install torch pandas numpy scikit-learn seaborn matplotlib joblib tqdm
```

### Dataset Setup

Download the three CSV files from [Kaggle](https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset) and place them in the project root directory.

### Run the Notebook

```bash
jupyter notebook wide_deep_recommender.ipynb
```

### Load the Saved Model

```python
import torch

# Option 1: Load the full model
model = torch.load('wide_and_deep_model_final.pth')
model.eval()

# Option 2: Load weights into an initialized model instance
model = WideAndDeep(
    wide_input_dim=5,
    embedding_dims=[...],
    hidden_units=[128, 64],
    deep_input_continuous_dim=5
)
model.load_state_dict(torch.load('wide_and_deep_model_state_dict_final.pth'))
model.eval()
```


## Reference

Cheng, H. T., Koc, L., Harmsen, J., Shaked, T., Chandra, T., Aradhye, H., ... & Shah, H. (2016).
Wide & Deep Learning for Recommender Systems.
Proceedings of the 1st Workshop on Deep Learning for Recommender Systems.
https://arxiv.org/abs/1606.07792

