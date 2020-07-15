# Residual-SE-Network

Residual Squeeze-and-Excitation Network for Fast Image Deraining

Jun Fu, Jianfeng Xu, Kazuyuki Tasaka, Zhibo Chen

https://arxiv.org/abs/2006.00757

Modified and add BASNet to suit reflection removal.

Requirements:(All network reimplements are same of similar)

* 1.Pytorch 1.3.0
* 2.Torchvision 0.2.0
* 3.Python 3.6.10
* 4.glob
(Dataset)
* 5.PIL
* 6.tqdm(For training)
* 7.Opencv-Python
* 8.tensorboardX


Dataset Modified:

Line 25,26,27

imgpath='/public/zebanghe2/cycledomain/dataset/mix'

transpath='/public/zebanghe2/cycledomain/dataset/transmission'

maskpath='/public/zebanghe2/cycledomain/dataset/mask'

In this project,maskpath is invalid if you just use ResSENet.

Train

python train.py

Epoch Number:Line 96

Batch Size:Line 91

Test
python test.py

If any problem, please ask in issue.

Jerry He
