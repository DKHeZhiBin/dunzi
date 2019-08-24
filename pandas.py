import numpy as np
import pandas as pd
s = pd.Series([1,2,3,4])

# 行索引，列索引 统称为标签；
# 为方便起见，系列称为向量，DataFrame称为矩阵
# print(s)

dates = pd.date_range('20190801',periods=6)
# print(dates)

b = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD')) #6行4列的随机数
# print(b)

c = pd.DataFrame({'A':[1,4,2,3],
'B':pd.Timestamp('20190104'),
'C':pd.Series(1,index=list(range(4)),dtype='float32'),
'D':np.array([3]*4,dtype='int32'),
'E':pd.Categorical(["test","train","test","train"]),
'F':'foo'})
# print(c)
# 显示头几行数据
# print(c.head(2))
# 显示倒数几行的数据
# print(c.tail(2))
# 显示行索引信息（行标签）
# print(c.index)
# 显示列字段信息（列标签）
# print(c.columns)

# 当您的DataFrame列具有不同数据类型时，这可能是一项昂贵的操作.NumPy数组对整个数组有一个dtype，而pandas DataFrames每列有一个dtype。当你调用DataFrame.to_numpy()，pandas会找到可以容纳 DataFrame中所有 dtypes 的NumPy dtype。这可能最终成为object，这需要将每个值都转换为Python对象。
# print(c.to_numpy())

# 显示数据的快速统计摘要
# print(c.describe())

# 转置数据矩阵（行变列，列变行）
# print(c.T)

# 根据行（列）名进行排序，改变行/列的位置，
# axis=0,按行排序；axis=1,按列排序
# print(c.sort_index(axis=1,ascending=False))

# 根据指定列名对里面的数据按值排序
# print(c.sort_values(by='A'))

# 选择一个列，产生一个Series，相当于df.A 
# print(c['A'])
# 注意：行无法通过索引标签获取行的数据,只能通过下面的获取横截面loc方法获取行数据或者直接根据位置获取行数据c[1]，例如
# x = pd.DataFrame(np.random.randn(2, 4),index=['a','b'],columns=list('ABCD'))
# print(x['a'])  ！！！！这是获取不到值的
# print(x.loc['a'])  只有这样才获取的到值

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
# 对行进行切片。
# print(df[1:3])
# print(df['20190802':'20190805'])

####################################按标签获取元素loc,at##################
# 使用标签获取横截面（行）
# print(df.loc[dates[0]])  #注意，此处loc是[]

# 按标签选择多列
# print(df.loc[:,['A','B']])

# loc第一个元素是行切片，第二个元素是列切片
# print(df.loc['20190802':'20190805', ['A', 'B']])

# 获取标量值,即获取矩阵元素值
# print(df.loc['20190802','B']) #或者df.at[dates[0],'B']
# 注意，以下是错误的，标签必须是非整型的！！！
# p = pd.DataFrame(np.random.randn(2, 2),index=[1,2],columns=['a','b'])
# print(p.at[2,'b'])

####################################按位置（传递的整数的位置）获取元素 iloc,iat才可以，其他的像loc，at无法使用位置获取元素##################
# 指定确切行/列
# print(df.iloc[3])
# print(df.iloc[[0,2],[1]])

# 行列切片
# print(df.iloc[0:2,1:3])

# 明确获取某个元素值
# print(df.iloc[1,1])

# 快速访问标量
# print(df.iat[1,1])

#################################布尔索引###############################################
# 保留在A列上值大于0的行
# print(df[df.A>0])

# 筛选值大于0的元素，不满足的用NaN表示
# print(df[df>0])

# df2 = df.copy()
# df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
# print(df2)
# print(df2[df2['E'].isin(['two','four'])])

# panda的一些设定：
# 1、设置新列会自动根据索引对齐数据。
# s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20190801', periods=6))
# print(s1)
# 课外实践：可以动手试试，打乱索引顺序，会发现panda依然可以根据索引去对位置
# df['F'] = s1
# print(df)
# 若对不上索引，则相应位置会记为NaN
# s1 = pd.Series([1, 2, 3, 4, 5, 6], index=[1,3,5,7,8,9])
# df['F'] = s1
# print(df)

# 按标签设置值：
# df.at[dates[2],'C'] = 1
# df.at[dates[2],'C'] = 'saber' #这是不可以的，可见panda要求同列元素类型必须相同
# print(df)

# 按位置设置值：
# df.iat[1,2] = 0
# print(df)

# 使用NumPy数组进行设置值
# df.loc[:,'D'] = np.array([5]*len(df))
# print(df)

# 设置相反数
# df2 = df.copy()    #可见panda的强大功能，别想太复杂
# df2[df2>0] = -df2 #一定要懂布尔索引的意思； 俩矩阵，对应元素赋值
# print(df2>0) #对应每个元素是否存留
# print(df2.A>0) #对应每一行是否存留
# print(df2)

# 缺少数据,np.nan表示缺失的数据。默认情况下，它不包含在计算中
df1 = df.reindex(index=dates[0:4],columns=list(df.columns)+['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
# print(df1)
# 删除任何缺少数据的行。
# a = df1.dropna(how='any') #返回删除后的矩阵，自身并不变
# print(df1)
# print(a)
# 填写缺失的数据。
# b = df1.fillna(value=5) #返回填充后的矩阵，自身并不变
# print(df1)
# print(b)
# 获取值所在的布尔掩码nan
# print(pd.isna(df1))

# print(df.mean())  #行平均值
# print(df.mean(1)) #列平均值

# 使用具有不同维度且需要对齐的对象进行操作。此外，pandas会自动沿指定维度进行广播。
# s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2) #shift是右移
# print(s)
# print(df)
# print(df.sub(s,axis='index'))  #矩阵减去向量（矩阵每一行的元素减去向量对应的元素）

# 将函数应用于数据：
# print(df)
# print(df.apply(np.cumsum))   #np.cumsum按照所给定的轴参数返回元素的梯形累计和，axis=0，按照行累加。axis=1，按照列累加。axis不给定具体值，就把numpy数组当成一个一维数组。
# print(df.apply(lambda x: x.max()-x.min()))   #每一列最大减去最小

# s = pd.Series(np.random.randint(0,7,size=10))
# print(s)
# print(s.value_counts())

# 字符串方法
# s = pd.Series(['A','B','C','Aaba',np.nan,'CABA','dog','cat'])
# print(s.str.lower())  #str指代系列中的所有字符串对象

# 合并？
# df = pd.DataFrame(np.random.randn(10,4))
# print(df)
# pieces = [df[:3],df[3:7],df[7:]]   #分解
# print(pieces)
# print(pd.concat(pieces))    #将pandas对象连接在一起concat()

# 根据笛卡尔乘积进行sql样式连接
# left = pd.DataFrame({'key':['foo','foo'],'lval':[1,2]})
# right = pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})
# print(left)
# print(right)
# print(pd.merge(left,right,on='key'))

# 追加行
# df = pd.DataFrame(np.random.randn(8,4),columns=['A','B','C','D'])
# print(df)
# s = df.iloc[3]
# print(df.append(s,ignore_index=True))

# **组的作用**
# 1、根据某些标准将数据拆分为组
# 2、将功能独立应用于每个组
# 3、将结果组合成数据结构
# df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                         # 'foo', 'bar', 'foo', 'foo'],
                   # 'B': ['one', 'one', 'two', 'three',
                         # 'two', 'two', 'one', 'three'],
                   # 'C': np.random.randn(8),
                   # 'D': np.random.randn(8)})
# 作用2：
# print(df.groupby('A').sum())
# 多列分组形成分层索引:
# print(df.groupby(['A','B']).sum())

# 堆栈?
# tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                              # 'foo', 'foo', 'qux', 'qux'],
                             # ['one', 'two', 'one', 'two',
                              # 'one', 'two', 'one', 'two']]))
# print(tuples) 
# index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])  #多级索引,names是各层级名称
# print(index)
# df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
# df2 = df[:4]
# print(df2)
# 该stack()方法“压缩”DataFrame列中的级别。
# stacked = df2.stack()
# print(stacked)
# stack()的逆操作,unstack()默认情况下取消堆叠最后一级
# print(stacked.unstack())
# print(stacked.unstack(1))
# print(stacked.unstack(0))

# *************数据透视表*********************
# df = pd.DataFrame({'A':['one','one','two','three']*3,
                   # 'B':['A','B','C']*4,
                   # 'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   # 'D': np.random.randn(12),
                   # 'E': np.random.randn(12)})
# print(df)             
# 数据透视表（Pivot Table）是一种交互式的表，可以进行某些计算，如求和与计数等。所进行的计算与数据跟数据透视表中的排列有关。
# 之所以称为数据透视表，是因为可以动态地改变它们的版面布置，以便按照不同方式分析数据，也可以重新安排行号、列标和页字段。
# 每一次改变版面布置时，数据透视表会立即按照新的布置重新计算数据。另外，如果原始数据发生更改，则可以更新数据透视表。
# 一定要理清表的转化逻辑，他是将指定的行/列的值作为标签
# 类似于------------------------------------------
      # |\ 日|  日期  |  日期   | 日期  |  日期  |
      # | \期|  那列  |  那列   | 那列  |  那列  |
      # |时\ |  的值  |  的值   | 的值  |  的值  |
      # |间 \|   1    |   2     |   3   |    4   |
      # |----------------------------------------|    
      # |时间|        |         |       |        |
      # |那列|        |         |       |        |
      # |的值|        |         |       |        |
      # |  1 |        |         |       |        |
      # |----------------------------------------|
# print(pd.pivot_table(df,values='D',index=['A','B'],columns=['C']))

# *************************时间序列*************************
# 重采样
# rng = pd.date_range('1/1/2012',periods=60,freq='1min')
# ts = pd.Series(np.random.randint(0,500,len(rng)),index=rng)
# print(ts)
# print(ts.resample('5min').sum())   #理解该例子

# 时区代表：
# rng = pd.date_range('1/8/2019 00:00', periods=5, freq='D')
# ts = pd.Series(np.random.randn(len(rng)), rng)
# print(ts)
# ts_utc = ts.tz_localize('UTC')  #UTC:协调世界时，又称世界统一时间、世界标准时间、国际协调时间
# print(ts_utc)

# 转换为另一个时区
# print(ts_utc.tz_convert('US/Eastern'))

# 在时间跨度表示之间转换
# rng = pd.date_range('1/8/2019', periods=5, freq='M')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# ps = ts.to_period()  #转化为以时期作为标签索引
# print(ps)
# tsp = ps.to_timestamp()  #转化为以时间戳作为标签索引
# print(tsp)

# ***************************分类*****************************
# df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                   # "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})
# 将原始成绩转换为分类数据类型。
# df["grade"] = df["raw_grade"].astype("category")     #将类别信息存储到category变量中
# print(df["grade"])
# 将类别重命名为更有意义的名称（分配到 Series.cat.categories就位！）。
# df["grade"].cat.categories = ["very good","good","very bad"]
# print(df["grade"])
# 重新排序类别并同时添加缺少的类别（默认情况下返回新方法）。Series .catSeries
# df["grade"] = df["grade"].cat.set_categories(["very bad","bad","medium","good","very good"])
# print(df["grade"])
# 排序是按类别中的每个顺序排序(cat.set_categories设的值顺序)，而不是词汇顺序。
# df.sort_values(by="grade")
# 按分类列分组还显示空类别。
# print(df.groupby("grade").size())

# *****************************绘图?*******************************
ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2000', periods=1000))
# ts = ts.cumsum()    #cumsum返回给定axis上的累计和
# ts.plot()    #在ipython交互式脚本上可实现将图打印出来

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
# df = df.cumsum()
# plt.figure()
# df.plot()
# plt.legend(loc='best') 

# ****************************获取数据(重要，必须记住)******************************
# csv文件
# df.to_csv('D://Users//Documents//超高校级的绝望//1.csv')  #写入csv文件
# print(pd.read_csv('D://Users//Documents//超高校级的绝望//1.csv'))  #读取csv文件

# hdf文件                  Hierarchical Data Format可以存储不同类型的图像和数码数据的文件格式，并且可以在不同类型的机器上传输，
                           #同时还有统一处理这种文件格式的函数库。大多数普通计算机都支持这种文件格式。
# 从HDF5存储中读取。
# pd.read_hdf('foo.h5', 'df')
 
# 写入excel文件
# df.to_excel('foo.xlsx', sheet_name='Sheet1')
# 从excel文件中读取。
# pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])
