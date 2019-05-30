# Stock_Prediction_with_RNN
Predict stock market prices using RNN model with multilayer LSTM cells + optional multi-stock embeddings.

1. you can install tensorflow, numpy

```
pip install tensorflow
pip install numpy
```

2. you might use a csv file on the folder panda.csv which has the information of value of stock

3. run prediction.py

`python prediction.py`

4. use tensorboard on logs
```
cd logs/
tensorboard --logdir=.
```
5. if you run the prediction.py, you can create tensorboard file as well as model on model folder

Based on the information on 5 columns below, you can get the info of prediction 18.8679 which is pretty similar with 18.79 on 7th row * 5th column

<img width="256" alt="Screen Shot 2019-05-07 at 4 48 37" src="https://user-images.githubusercontent.com/42028366/57299316-350cf300-710f-11e9-95e3-628d2fbb21f8.png">
<img width="356" alt="Screen Shot 2019-05-07 at 4 48 44" src="https://user-images.githubusercontent.com/42028366/57299318-350cf300-710f-11e9-8b23-1c8097a77551.png">

