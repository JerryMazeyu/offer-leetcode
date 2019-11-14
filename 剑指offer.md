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

