import torch
import matplotlib.pyplot as plt
import torchvision.transforms as tfs
from PIL import Image
import time

def getAffinity_Matrix(img):
    img = img.permute(1,2,0)#进行维度置换（14,14,3）
    # [width, height]
    #创建一个H*W的方阵，作为亲和矩阵的初始阵，为何是H*W上文已说明
    affinity = torch.zeros(img.shape[0]*img.shape[1], img.shape[0]*img.shape[1])
    print(affinity.shape)
    #将图片变为一维，但是3通道。所以img1.shape为（196,3）
    img1 = img.reshape(-1, img.shape[-1])
    # 计算向量的模 元素自点乘 将三通道相加 开根号 Tensor(196,)
    # 每个像素点都是3个分量构成 将通道分量结合
    img_ = torch.sqrt((img1[:,:]**2).sum(dim=-1))
    img2 = img.reshape(-1, img.shape[-1])
    for idx in range(affinity.shape[1]):
        affinity[idx, :] = torch.mul(img1[idx, :], img2[:, :]).sum(dim=-1)
        affinity[idx, :] = affinity[idx, :]/img_[idx] #求余弦值函数
    for idx in range(affinity.shape[0]):
        #continue
        affinity[:, idx] = affinity[:, idx]/img_[idx]
    print(affinity)
    return affinity

def display(affinity):
    plt.imshow(affinity)
    plt.colorbar()
    plt.savefig("affinity.jpg")
    plt.show()

def process(img_root, rate=16):
    img = Image.open(img_root)#打开图像
    size = 224//rate
    img = img.resize((size,size))#缩放到16*16
    plt.imshow(img)
    plt.savefig("tmp.jpg")#保存缩放以后的图
    img = tfs.ToTensor()(img)#将图片转为Tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#划分一个位置来放置Tensor
    img = img.to(device)#将Tensor放到划分好的GPU上
    return img#注意得到的是（3,14,14）的Tensor
img=torch.randn(size=(3,4,4))
# img = process(img)
affinity = getAffinity_Matrix(img)
display(affinity)
