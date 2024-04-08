
# SecNet: A Second Order Neural Network for MI-EEG

This repository contains the code for the paper: 

**Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).**  
*SecNet: A second order neural network for MI-EEG.*  
Information Processing & Management, 62(3), 104012.

## Overview

SecNet is a neural network model designed to enhance motor imagery (MI) decoding from EEG signals by leveraging second-order information.

## Requirements

- Python 3.10 
- PyTorch 2.2.2 
- Numpy 1.26.3 (Numpy 2.x.x may cause errors in EEG processing)
- Other dependencies can be found in `requirements.txt`.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SecNet-mi/SecNet.git
   cd SecNet
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Download Dataset and Prepare Data
The data used in this study are already publicly available at the following: OpenBMI in http://dx.doi.org/10.5524/100542;  BCI competition IV 2a in https://bbci.de/competition/iv/. 
1. Prepare your EEG dataset in the required format.
```
RawData/
├── BCICIV_2a/A01E.gdf...
└── OpenBMI/
    ├── session1/sess01_subj01_EEG_MI.mat ...
    └── session2/sess02_subj01_EEG_MI.mat ...
```
After downloaded the dataset, you need to run: 
```bash
python DataLoader\LoadData.py
```

### To run the model:

2. Train the model using the provided script:
   ```bash
   python ho.py 
   ```

For more details on usage, check the documentation in the codebase.

## Citation

If these codes help you, please cite:

```
@article{liang2025secnet,
  title={SecNet: A second order neural network for MI-EEG},
  author={Liang, Wei and Allison, Brendan Z and Xu, Ren and He, Xinjie and Wang, Xingyu and Cichocki, Andrzej and Jin, Jing},
  journal={Information Processing \& Management},
  volume={62},
  number={3},
  pages={104012},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
