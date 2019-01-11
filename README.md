# Kaggle-HPA
PyTorch implementation for [Kaggle Human Protein Atlas Image Classification Challenge](https://www.kaggle.com/c/human-protein-atlas-image-classification)


## Training
* Model
	- `ResNet18` backbone (pretrained on ImageNet)
	- Small decoder for image reconstruction
* Input size: (512, 512, 4) (RGBY)
* Augmentation
	- Random horizontal flip
	- Random vertical flip
	- Random affine transformation (rotation, translation, scale)
* Batch size: 40
* Optimizer: SGD
* Learning rate: cosine annealing from 5e-2 to 4e-4 (1 cycle)
* Weight decay: 1e-4
* Loss functions
	- Binary Cross Entropy (BCE)
		+ Log-damped class frequency pos_weights, without oversampling rare data
	- Mean Square Error (MSE) for image reconstruction (unsupervised)
		+ To learn a good feature representation as an auxiliary signal
	- Prediction for # of class co-occurrence (COOC)
		+ Get higher confidence of predictions with higher threshold
* 5 folds (random split), 20 epochs for each fold


## Final prediction
* The mean of the predictions of 5-fold model with 8 TTA (transpose + flip)
* Threshold: 0.4
* Score (with data leak): public LB 0.568 / private LB 0.516 (**138th / 2236**)


## Requirements
* pytorch 0.4.1
* torchvision 0.2.1
* numpy
* opencv
* scikit-learn
* pandas
* tqdm

`pip install -r requirements.txt`


## Usage

### Data
* Download data from [Kaggle HPA competition](https://www.kaggle.com/c/human-protein-atlas-image-classification/data)
* Extract train/test.zip
* Download [HPAv18 external data](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984) and `HPAv18RGBY_WithoutUncertain_wodpl.csv`
* Put the external data into the corresponding `external` directory
* Modify the path appropriately in `config.json`
* Run `PYTHONPATH=. python loaders/hpa_loader.py` first to generate training/validation data lists

### To train/test the model, create final submission
`python [train, test, merge].py -h` for more details
