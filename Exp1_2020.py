"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2

###
#please coding here for solving Task [I].
file_name = "/Users/LT/Desktop/IDE/preview.jpg"
img=cv2.imread(file_name)#按照BGR顺序
cv2.imshow("img",img)#显示图像
b,g,r=cv2.split(img)#通道分离
cv2.imshow("rr",r)#通道图单独显示红色
cv2.waitKey(0)#窗口等待任意键盘按键输入,0为一直等待,其他数字为毫秒数
plt.hist(img[:,:,2].ravel(),bins=256,color='r')#ravel()把多维数组转化为一维数组
plt.xlabel('bins = 256 red levels')
plt.ylabel('Counted pixel numbers in each level')
plt.title('Red Histogram')
plt.show()#matplotlib based

hist = cv2.calcHist([img], [2], None, [256], [0.0,255.0])#使用opencv的方法
plt.plot(hist,color='r')
plt.show()

hist2,x=np.histogram(img[:,:,2].ravel(),bins=256,range=(0,256))#使用numpy
plt.plot(hist2,color='r')
plt.show()



###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

###
#please coding here for solving Task[II]
import scipy
from scipy import ndimage
#创建4个小窗口显示不同sigma的结果
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
result=np.zeros(img.shape,dtype=np.uint8)
result2=np.zeros(img.shape,dtype=np.uint8)
result3=np.zeros(img.shape,dtype=np.uint8)
for i in range(3):  # 对图像的每一个通道都应用高斯滤波
    result[:,:,i]=scipy.ndimage.gaussian_filter(img[:,:,i],sigma=1)
    result2[:,:,i]=scipy.ndimage.gaussian_filter(img[:,:,i],sigma=3)
    result3[:,:,i]=scipy.ndimage.gaussian_filter(img[:,:,i],sigma=9)
ax1.imshow(img)
ax2.imshow(result)
ax3.imshow(result2)
ax4.imshow(result3)
plt.show()




"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]

mean = (2, 2,2)
cov = np.eye(3)#使用numpy.eye()来直接生成一个对角矩阵
x = np.random.multivariate_normal(mean, cov, (600, 600))
plt.hist(x.ravel(), bins=200, color='r')#统计直方图
plt.show()

#mean：均值，n维分布的平均值，是一个一维数组长度为N.在标准正态分布里对应的就是图形的峰值。
#cov：分布的协方差矩阵，它的形状必须是（n,n），也就是必须是一个行数和列数相等的类似正方形矩阵，
# 它必须是对称的和正半定的，才能进行适当的采样。
#返回值：一个n维数组。如果指出了形状尺寸也就是前边的size，则生成的样本为指定的形状，
# 如果没有提供size，则生成的样本形状为（n,）
