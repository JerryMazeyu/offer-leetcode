# 剑指offer（Python版)
## NO1 单例模式的实现
### 什么是单例模式？
单例模式是设计模式中最容易实现的一种模式，而设计模式又是面向对象编程非常重要的一环，那么这就不奇怪为什么在考题中经常会用考这道题目了。  
单例模式是在一个类中只能有一个实例，一个类只返回一个对象和一个引用这个对象的方法。在现实的应用场景中，这经常用来控制全局的配置或者充当信号量的作用使进程之间能够通信。  
### 如何实现单例模式？
#### 方法一
`python`模块是天然的单例模式，因为每次执行一个模块后会产生一个`pyc`文件，下次直接会从这个`pyc`中加载实例，所以很简单的：
##### singleton.py
```python
class Singleton(object):
	def __init__(self):
		pass

singleton1 = Singleton()
```
让这个`py`文件用模块的方式调用，比如在`main.py`中引用一下这个文件：  
 
```python
from singleton import singleton1
```
#### 方法二
`python`的`__new__()`方法也是实现单例模式的好方法，`__new__()`方法是在实例花对象时最先被执行的magic方法，利用这个特性，我们可以重载`__new__()`方法。  

```python
import threading  # 多线程模块
class Singleton(object):
	_instance_lock = threading.Lock()  # 加载一个锁的对象
	def __init__(self):
		pass 
	def __new__(cls):
		if not hasattr(Singleton, "_instance"):  # hassattr(obj, name)这是一个检测一个对象有没有name属性的api，如果True的话，就说明实例已经被创建了
			with Singleton._instance_lock:  # 尝试获取线程锁
				if not hasattr(Singleton, "_instance"):  # 获取到了锁的情况下，发现还没有实例，那么赶紧创建一个实例
          			Singleton._instance = object.__new__(cls)
      	return Singleton._instance
      
      
       
obj1 = Singleton()
obj2 = Singleton()
print(obj1,obj2)
```
这其中的逻辑是这样的：构建实例时首先会运行`__new__()`，然后查看是否已经有实例，如果有就直接返回创建的实例即可，但是如果没有的话不可以马上就创建新的实例，这是因为再多线程的情况下，有可能会出现数据读取的错乱，甲乙同一时刻创建这个实例的时候，这会同时产生两个实例。所以接下来要去访问这个锁，保证的是创建的过程中，只能是只有一个进程正在创建。这样的解法也是很棒了。
输出的结果为：  

```python
<__main__.Singleton object at 0x1088fd438> 
<__main__.Singleton object at 0x1088fd438>
```
可以看出每一个实例都是一个实例。  
## NO2 数组中重复的数字
在一个长度为n的数组里的所有数字都在0～n-1的范围里。数组中某些数字是重复的，但不知道有几个数组重复了，也不知道每个数字重复了几次。请找出数组中**任意**一个重复的数字。例如，如果数组为[2, 3, 1, 0, 2, 5, 3]返回2或者3。
### 大概的思路
* 首先我想到的是：构造一个哈希表，里面的value存放频数，就像桶排序那样，这种方法一定是可行的。但是这并不优，因为需要额外的空间去存放一个dict。  
* 其次比较容易去想到的是排序，那么原地快排的时间复杂复杂度会是O(nlogn)，这看起来也不像是最优的解法，因为有一种直觉，（对，就是直觉！）他只找出任意的一个，所以会有一些trick。  
* 作者就是狠啊，他提出一个非常nice的算法，我尝试遵循他最初的想法走下去，大概是这样的：先假设一种最好的情况，就是没有一个重复的数字，那么此时如果将数字排序，会出现的结果是数组的下标和数组的值相等。抓住这个想法，我们的目的是找出O(n)的时间复杂度、O(1)的空间复杂度的算法。那么我们只能遍历一次数组，然后每次尝试着把不好的情况像理想的情况靠拢。
* 具体来说，用一些组图去表示会比较好：
* ![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-11-14-172140.jpg)
* 从这张图里面其实可以看出，慢慢的序列就被排的越来越整齐，这样也就达到了找重复数字的作用。  

### 上代码
```python 
def find_same_num(num):
    cur = 0  # 定义指针指向首位字符串
    while cur < len(num)-1:  # 只能遍历一遍
        if num[cur] != cur:  # 如果下标和指针所指代的位置不同
            if num[num[cur]] != num[cur]:  # 而且指针指向的数字和他所在的下标也不同
                tmp = num[cur]  # 那么交换
                num[cur], num[tmp] = num[tmp], num[cur]
            else:
                return num[cur]  # 如果相同，那么你就找到了重复数字
        else:
            cur += 1  # 慢慢遍历
    return None
```