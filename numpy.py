import numpy as np
from numpy import pi
import sys
# 轴，也就是维度，对于任意多维数组，可通过数括号来确定是哪条轴。
# 例如[[1,2][3,4]] 有2层括号，我们把每层括号叫做轴，最外那层是第一个轴
# 第一轴，axix = 0，它有2个成员，长度是2；下一层括号叫做第二轴，axix=1，它们
# 均有2个成员，长度是2

# 数组类ndarray，别名array。注意：numpy.array这与标准Python库类不同array.array，后者只处理一维数组并提供较少的功能。
# ndarray重要属性：
# ndarray.ndim 轴数
# ndarray.shape 数组的大小，这是一个整数元组，表示每个维度中数组的大小。例如二维用(m,n)表示
# reshape是指定数组的大小重新构造数组结构
# ndarray.size 数组的元素总数，等于各维元素个数乘积
# ndarray.dtype 描述数组中元素类型的对象,可以使用标准Python类型创建或指定dtype。此外，NumPy还提供自己的类型。numpy.int32，numpy.int16和numpy.float64就是一些例子
# ndarray.itemsize 数组中每个元素的大小（以字节为单位）。相当于ndarray.dtype.itemsize。
# ndarray.data 包含数组实际元素的缓冲区。通常，我们不需要使用此属性，因为我们将使用索引工具访问数组中的元素。

# a = np.arange(15).reshape(3,5)   #arange返回一维数组
# print(a)
# print(a.shape)
# print(a.ndim)
# print(a.dtype.name)
# print(a.itemsize)
# print(a.size)
# print(type(a))

# 数组创建
# a = np.array([4,5,6])  #通过列表创建数组
# a = np.array(4,5,6)     #wrong
# print(a)
# print(a.dtype)    #dype和dype.name返回结果一样
# b = np.array([(1,2,3),(4,5,6)])
# print(b)
# 也可以在创建时显式指定数组的类型
# c = np.array([4,5,6],dtype = complex) 
# print(c)

# 创建全是0的数组
# a = np.zeros((1,3))
# print(a)
# 创建全是1的数组
# b = np.ones((2,3,2),dtype = np.int16)
# print(b)
# 创建随机数组
# c = np.empty((2,3),dtype = np.int16)
# print(c)

# a = np.arange(0,2,0.3)   #类似range，但它返回数组而不是列表。步长若不写，默认为1
# print(a)
# 当arange与浮点参数一起使用时，由于有限的浮点精度，通常不可能预测所获得的元素的数量。
# 出于这个原因，通常最好使用linspace设置我们想要接收的元素数量
# b = np.linspace(0,2,9)       #与arang不同，最后一个参数是要接收的元素数量，它是根据数量来设置步长的
# print(b)
# x = np.linspace(0,2*pi,100)
# print(x)
# f = np.sin(x)


# 数组打印
# a = np.arange(6)
# print(a)            #一维数组打印为行
# b = np.arange(12).reshape(4,3)
# print(b)            #二维数组打印为矩阵
# c = np.arange(24).reshape(2,3,4)
# print(c)             #三维数据打印为矩阵列表
# 数组太大而无法打印，NumPy会自动跳过数组的中心部分并仅打印角落
# print(np.arange(10000))
# print(np.arange(10000).reshape(100,100))
# 要使numpy打印整个阵列，可以更改打印选项set_printoptions。
# np.set_printoptions(threshold=sys.maxsize)

# ----------------------数组或矩阵的基本运算----------------------
# 数组上的算术运算符应用于元素。***创建一个新数组并填充结果
# 加减
# a = np.array([20,30,40,50])
# b = np.arange(4)
# c = a - b
# print(c)
# d = a + b
# print(d)
# 乘方
# e = b ** 2
# print(e)
# 三角函数
# f = np.sin(a)
# print(f)
# 布尔运算
# g = a < 35
# print(g)
# ****注意：乘法*不是矩阵乘法，而是对应元素相乘;
# ****矩阵相乘用@或者dot函数
# h = a * b
# print(h)
# i = a @ b   #或者i = a.dot(b)
# print(i)
# 矩阵也有+=或者*=的运算，只是这会修改现有阵列而不是创建新阵列(其实本质已经创建了新矩阵只不过将值赋给了本身)
# a*=3
# print(a)
# b+=a
# print(b)
# 当使用不同类型的数组进行操作时，结果数组的类型对应于更一般或更精确的数组（称为向上转换的行为）
# a = np.ones(3, dtype=np.int32)
# b = np.linspace(0,pi,3)
# print(b.dtype.name)
# c = a + b
# print(c)
# d = np.exp(c*1j)           #??
# print((d,d.dtype.name))

# 数组的一些统计操作
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(a)
# print((a.sum(),a.min(),a.max()))
# 指定axis 参数，可以沿数组的指定轴应用操作
# print(a.sum(axis = 1))        #a.min(axis=0)  a.max(axis=1)
# print(a.cumsum(axis = 1))       #cumsum是按轴求梯形累加和

# 通用函数，也叫做数学函数（exp，sin，cos等）
# a = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(np.exp(a))
# print(np.sqrt(a))

# *************************索引，切片和迭代*******************
# 数组的索引，切片和迭代与列表的相同，不同轴之间用逗号隔开
# def f(x,y):
    # return 10*x+y   
# b = np.fromfunction(f,(5,4),dtype = int)      #fromfunction函数是将数组下标作为函数参数，计算出值创建出矩阵
# print(b)
# print(b[0:2,1])

# 当提供的索引少于轴的数量时，缺失的索引被认为是完整的切片
# print(b[-1])   #b[-1] == b[-1,:]

# numpy允许使用点来省略轴，点...表示许多冒号
# x[1,2,...]等价于x[1,2,:,:,:]
# x[...,3]等价于x[:,:,:,:,3]
#x[4,...,5,:]等价于x[4,:,:,5,:]
# 小练习：
# c = np.array( [[[  0,  1,  2],              
                # [ 10, 12, 13]],
               # [[100,101,102],
                # [110,112,113]]])
# c[1,...] = ??
# c[...,2] = ??

# 对多维数组进行迭代是针对第一个轴完成的;
# 要一个个元素全部取出来，要看数组有几个轴，就是有几重循环；
# 也可以使用flat属性 作为数组的所有元素的迭代器，取出所有元素
# c = np.array( [[[  0,  1,  2],              
                # [ 10, 12, 13]],
               # [[100,101,102],
                # [110,112,113]]])
# for element in c.flat:
    # print(element)

# *************************形状操纵******************
# 数组的形状由沿每个轴的元素数量给出
# a = np.floor(10*np.random.random((3,4)))  #floor是向下取整函数
# print(a)
# 各种命令更改矩阵的形状
# print(a.ravel())    #ravel改为一维数组

# print(a.reshape(6,2)) 

# print(a.T)      #转置矩阵

# a.resize(6,2)   #reshape函数返回带有修改形状的新数组，而 ndarray.resize方法修改数组本身
# print(a)

# 如果在重新整形操作中将尺寸指定为-1，则会自动计算其他尺寸（例如总共12个元素，已知每行6个，便可以求出另一个参数是2行）
# a.reshape(3,-1)
# array([[ 2.,  8.,  0.,  6.],
       # [ 4.,  5.,  1.,  1.],
       # [ 8.,  9.,  3.,  6.]])

# ***********************************堆叠不同数组??????********************************
# a = np.floor(10*np.random.random((2,2)))
# b = np.floor(10*np.random.random((2,2)))
# print(a)
# print(b)
# 注意，vstack和hstack函数只有一个参数，所以((a,b))
# c = np.vstack((a,b))     #往纵向堆叠
# print(c)  
# d = np.hstack((a,b))     #往横向堆叠
# print(d)

#??????????????column_stack可将1维数组作为列堆叠到2维数组中，类似于hstack
# 注意，column_stack函数只有一个参数，所以((a,b))
# d = np.column_stack((a,b))  
# print(d)
# m = np.array([4,3])
# n = np.array([2,8])
# print(np.column_stack((m,n)))
# print(np.hstack((m,n)))     #这俩结果是不同的

# from numpy import newaxis
# print(m[:,newaxis])          #允许1维数组变为2维数组
# i = np.column_stack((m[:,newaxis],n[:,newaxis]))
# j = np.hstack((m[:,newaxis],n[:,newaxis]))
# print(i)
# print(j)                     #此时这俩函数结果是一样的

# ???????r_和c_是通过沿一个轴堆叠号码创建阵列有用的
# print(np.r_[1:4,0,4])

# ******************数组拆分****************


# ******************花式索引****************
# 使用索引数组进行索引
# a = np.arange(4)**2
# i = np.array([0,2,1,3])
# print(a[i])

# 索引数组i是多维时，单个索引数组指的是a的第一个维度。
# 换句话说，可将索引数字直接替换成a第一个维度的相应元素，i的其他括号不变
# a = np.array([4,2,3,1,2,7,8,9,6,4,8,1,3,4])
# i = np.array( [ [ 3, 4], [ 9, 7 ] ] )      
# print(a[i])                                # a[i]数组与i同形状


# a = np.array( [ [0,0,0],
                    # [255,0,0], 
                    # [0,255,0], 
                    # [0,0,255],
                    # [255,255,255]])
# i = np.array( [ [ 0, 1, 2, 0 ],
                    # [ 0, 3, 4, 0 ]  ] )
# print(a[i])


# a = np.arange(12).reshape(3,4)
# print(a)
# i = np.array( [ [0,1],                        # 第一维的索引
                # [1,2] ] )
# j = np.array( [ [2,1],                        # 第二维的索引
                # [3,3] ] )
# print(a[i,j])                                     # i 和 j必须形状相同
                                                # 必须理解：a[i,j]的机制是数组i和数组j相同位置的对应数字两两组成一对索引，
                                                            # 然后用这对索引在数组a中进行取值。比如数组i的索引(0,0)处的值为0，
                                                            # 数组j的索引(0,0)处的值为2，它们组成的索引对是(0,2)，在数组a中对应的值是2。
                                                            # 在这样的机制下，理所当然要求数组i和数组j需要有相同的形状，否则将无法取得相应的索引对。
                                                            # 又因为数组i和数组j分别是数组a的两个轴(axis)上的索引数组，所以最终的结果也就和数组i/j的形状相同
                                         
# print(a[i,2])                                    #数组i是数组a第一个轴的索引数组，a[i,2]中的数字2表示数组a的第二个轴的索引，
                                                   #数组i中的每个数字都与2组成索引对，也就是([ [(0,2), (1,2)], [(1,2),(2,2)] ])，然后依据这些索引对和相应的形状取数组a中的值。

# print(a[:,j])                                    # 对数组a第一个轴进行完整切片，得到(0,1,2)，然后每个值都与数组j中的元素两两组成索引对，也就是组成3个二维索引对，然后根据索引对取数组a中的值

#可将i，j放入列表进行索引
# l = [i,j]                                       #l = i,j
# print(a[l])                                     #等价于a[i,j];但是目前python不建议这么做
#建议使用'arr[tuple(seq)]' 而不是'arr[seq]'
# s = np.array([i,j])
# t = a[tuple(s)]
# print(t)                  #这是对的，相当于a[i,j]
# 但不能将i和j放入数组来实现这一点，因为这个数组将被解释为索引a的第一个维度
# print(a[s])              #这会报错

# 使用数组索引的另一个常见用法是搜索与时间相关的系列的最大值??????????
# time = np.linspace(20, 145, 5)
# data = np.sin(np.arange(20)).reshape(5,4)
# print(time)
# print(data)
# ind = data.argmax(axis=0)              #argmax是求出平行于axis=0方向的数据的最大值0轴的索引
# print(ind)
# time_max = time[ind]
# data_max = data[ind, range(data.shape[1])]
# print(time_max)
# print(data_max)
# np.all(data_max == data.max(axis=0))       #all（）方法，直接比对a矩阵和b矩阵的所有对应的元素是否相等。
                                           #any（）方法是查看两矩阵是否有一个对应元素相等。
                                           #事实上，all（）操作就是对两个矩阵的比对结果再做一次与运算，而any则是做一次或运算
# 用数组索引作为分配给的目标
# a = np.arange(5)
# a[[1,3,4]] = 0
# print(a)
# 当索引列表包含重复时，分配会多次完成，留下最后一个值：
# a = np.arange(5)
# a[[0,0,2]]=[1,2,3]
# print(a)
# 注意是否要使用Python的 +=构造，因为它可能不会按预期执行：
# a = np.arange(5)
# a[[0,0,2]]+=1                    #第0个元素只增加了一次
# print(a)

# **************************使用布尔数组进行索引**********************
# a = np.arange(12).reshape(3,4)
# b = a > 4
# print(b)
# print(a[b])
# 此属性在分配中非常有用：
# a[b] = 0
# print(a)

# 使用布尔索引生成Mandelbrot集?????????
# import numpy as np
# import matplotlib.pyplot as plt
# def mendelbrot(h,w,maxit = 20):
    # y,x = np.ogrid[ -1.4:1.4:h*1j,-2:0.8:w*1j ]
    # c = x+y*1j
    # z = c
    # divtime = maxit + np.zeros(z.shape,dtype = int)
    # for i in range(maxit):
        # z = z ** 2 + c
        # diverge = z* np.conj(z) > 2**2
        # div_now = diverge & (divtime == maxit)
        # divtime[div_now] = i
        # z[diverge] = 2
    # return divtime

# plt.imshow(mendelbrot(400,400))
# plt.show()    

# a = np.arange(12).reshape(3,4)
# b1 = np.array([False,True,True])
# b2 = np.array([True,False,True,False])
# print(a[b1,:])
# print(a[b1])
# print(a[:,b2])
# print(a[b1,b2])

# ************************************ix********************************????????????????????
# a = np.array([2,3,4,5])
# b = np.array([8,5,4])
# c = np.array([5,4,6,8,3])
# ax,bx,cx = np.ix_(a,b,c)
# print(ax)
# print(bx)
# print(cx)
# print(ax.shape, bx.shape, cx.shape)
# result = ax+bx*cx
# print(result)
# print(result[3,2,4])
# print(a[3]+b[2]*c[4])
