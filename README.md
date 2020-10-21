# Utilising Deep Learning Techniques for Effective Zero-Day Attack Detection


This work aims at proposing an autoencoder implementation for detecting zero-day attacks. 


## Citation
To cite the paper please use the following format;

````
@article{Hindy2020,
  doi = {10.3390/electronics9101684},
  url = {https://doi.org/10.3390/electronics9101684},
  year = {2020},
  month = Oct,
  publisher = {{MDPI}},
  volume = {9},
  number = {10},
  pages = {1684},
  author = {Hanan Hindy and Robert Atkinson and Christos Tachtatzis and Jean-Noël Colin and Ethan Bayne and Xavier Bellekens},
  title = {Utilising Deep Learning Techniques for Effective Zero-Day Attack Detection},
  journal = {Electronics}
}

````

## General Parameters

| Argument       | Usage        				 	     | Default       |  Values and Notes	          |
| ---------------|:-------------------------------------:|:-------------:|:-------------------|
| --model | Autoencoder or OneClass SVM | Autoencoder | Use 'svm' for One-Class SVM |
| --normal_path      | CICIDS2017 Dataset Monday File Path     | DataFiles/CIC/biflow_Monday-WorkingHours_Fixed.csv  |  |
| --attack_paths     | CICIDS2017 Dataset Folder Path     | DataFiles/CIC/  |  |
| --dataset_path 	 | KDD or NSL csv path 	        | DataFiles/KDD/kddcup.data_10_percent_corrected | For the NSL-KDD dataset: DataFiles/NSL/KDDTrain+.txt|
| --output  | The output file name | Result.csv ||
| --epochs  | The number of ephochs | 50 || 
| --archi   | Autoencoder architecture | U15,U9,U15 | UX represents a layer with X neurons, D, represents a dropout layer |
| --dropout | Dropout value that is used with D layer in archi | 0.05 | | 
| --regu    | Regularisation | l2 | l1, l2 or l1l2 |
| --l1_value    | L1 Regularisation Value | 0.01 | |
| --l2_value    | L2 Regularisation Value | 0.0001 | |
| --correlation_value | Correlation value to drop features | 0.9 | |
| --nu | One-Class SVM nu value | 0.01 | | 
| --kern | One-Class SVM kernel | rbf | | 
| --loss | Loss function | mse | NSL-KDD: mae | 

## How to Run the repository:

```
Clone this repository
run main.py [specify the parameters as required]

