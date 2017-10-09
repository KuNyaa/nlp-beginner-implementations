# 文本分类Plus

- ## FNN

###        n gram

| training data size | gram |      layers       | dropout keep prob | batch size | epochs | accuracy |
| :----------------: | :--: | :---------------: | :---------------: | :--------: | :----: | :------: |
|       前50000       |  2   |    16269,1024     |        0.4        |     64     |  150   |  58.76%  |
|         全部         |  2   | 41925,2048,256,32 |        0.5        |     64     |  100   |  62.22%  |



然后是加了L2正则化的softmax regression

| training data size | gram | layers  | dropout keep prob | regularization rate | batch size | epochs | accuracy |
| :----------------: | :--: | :-----: | :---------------: | :-----------------: | :--------: | :----: | :------: |
|         全部         |  1   | 16113,5 |        1.0        |        0.002        |     64     |  150   |  58.88%  |
|         全部         |  3   | 51043,5 |        1.0        |       0.0005        |     64     |  150   |  62.07%  |



###        word embedding + initialize randomly

| embedding size |      layers      | dropout keep prob | regularization rate | batch size | epochs | accuracy |
| :------------: | :--------------: | :---------------: | :-----------------: | :--------: | :----: | :------: |
|      128       | 7168,1024,256,32 |        0.6        |       0.0005        |     64     |  150   |  60.49%  |
|      128       |      7168,5      |        1.0        |       0.0002        |     64     |  100   |  61.34%  |
|      128       |      7168,5      |        1.0        |        0.001        |     64     |  100   |  61.49%  |

###   

### word embedding + initialize with GloVe

to be done.



- ## CNN 

###        word embedding + initialize randomly

| embedding size | filter sizes | num filters per size | dropout keep prob | batch size | epochs | accuracy |
| :------------: | :----------: | :------------------: | :---------------: | :--------: | :----: | :------: |
|      128       |   3,4,5,6    |          64          |        0.6        |     64     |   10   |  58.36%  |
|      128       |   3,4,5,6    |          64          |        0.6        |     64     |  100   |  61.94%  |
|      256       |   3,4,5,6    |         128          |        0.6        |     64     |  150   |  62.24%  |
|      256       |  3,4,5,6,7   |         128          |        0.5        |     64     |  100   |  62.03%  |
|      128       |    3,5,7     |         128          |        0.4        |    128     |  100   |  62.55%  |
|      128       |    3,4,5     |         128          |        0.5        |     64     |  200   |  61.59%  |
|      256       |    3,5,7     |         128          |        0.4        |     64     |  200   |  62.03%  |
|      128       |   3,4,5,6    |         128          |        0.3        |     64     |  150   |  62.29%  |
|      128       |   3,4,5,6    |         128          |        0.4        |    128     |  150   |  62.28%  |
|      128       |   2,3,5,7    |         128          |        0.4        |    128     |  100   |  62.42%  |
|      128       |  3,4,5,6,7   |         256          |        0.3        |    128     |  200   |  61.95%  |



​	发现dropout已经hold不住了，于是加了L2正则化。



| embedding size | filter sizes | num filters per size | dropout keep prob | regularization rate | batch size | epochs | accuracy |
| :------------: | :----------: | :------------------: | :---------------: | :-----------------: | :--------: | :----: | :------: |
|      128       |    3,5,7     |         128          |        0.4        |        0.001        |    128     |  100   |  63.29%  |
|      128       |    3,4,5     |         128          |        0.5        |        0.001        |     64     |  200   |  63.54%  |
|      128       |    3,4,5     |         128          |        0.6        |        0.005        |     64     |  200   |  58.66%  |



PS: 最后一个是regularization rate太大

后续数据还在跑



### word embedding + initialize with GloVe

to be done.



- ## RNN

to be done.