# audio-denoise-addon-v2
This is a machine learning driven audio denoiser used in browser. Pytorch for model training, TVM for deployment.

# run
change the dataset path and dump path in ZENNet/train/train.py
dump path is a directory will recovered audio file is stored, so that the quality of the model can be checked.


enter the python environment

run
```
pip install -e .
```

then run
```
python3
```

then enter
```
from  ZENNet.train.train import train
train()
```
