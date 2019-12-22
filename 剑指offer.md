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

## NO3 不修改数组找出重复的数字
问题：  
在一个**长度为n+1**的数组里的所有数字都在0～n的范围内，所以数组至少存在一个重复的数字。请找出任意一个重复的数字。  
这个题目是上一个的变形版本，上一个的大致思路是不断的变换数组的位置，将数组和下标对应的索引进行不断的靠近，这样就比较容易判断是谁重复出现了。但是在不能改变数组的情况下，这种办法显然不再适用，那么新的思路诞生在如下：  
如果每一次我都统计一下0～n的数字出现的频数，如果这个频数>n，那么可以认为在0～n中存在着重复的数字。那么我们可以用二分法去处理这个问题，首先要做的是统计出这组数字的范围，然后和平时的二分法一样的流程，只不过需要加一下统计频数的步骤了。  
这里有一点需要说明的是，这种方法并不是所有问题的通解。在一段数组有重复的和数组本身范围长度相同这种情况很常见，比如[0, 1, 1, 3]，但是不能说明这里就没有重复数组。但是本题说明了长度n+1，则可以用这种解法解决。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-11-24-053854.jpg)
### 上代码
```python
import queue

def find_min_max(arr):
	"""找到极差的范围"""
    assert len(arr) > 0
    min = max = arr[0]
    for i in arr:
        if i > max:
            max = i
        if i < min:
            min = i
    return min, max

def find_frequency(arr:list, range_:tuple):
	"""查看是否频数和频率对的上"""
    lo, hi = range_[0], range_[1]
    count = 0
    for i in arr:
        if i >= lo and i <= hi:
            count += 1
    return count == hi - lo + 1

def split_range(range_):
	"""将范围用二分法拆分"""
    lo, hi = range_[0], range_[1]
    mid = round((lo + hi + 0.001)/2)
    return (lo, mid-1), (mid, hi)


def main(arr):
	"""主函数"""
    q = queue.Queue()
    range_ = find_min_max(arr)
    q.put(range_)
    while True:
        tmp = q.get()
        left, right = split_range(tmp)
        if not find_frequency(arr, left):
            q.put(left)
            if left[0] == left[1]:
                return left[0]
        if not find_frequency(arr, right):
            q.put(right)
            if right[0] == right[1]:
                return right[0]
```
这里面采用了一个BFS的思路去代替原始的二分查找，原始的二分查找可以这样去做：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-11-24-053919.jpg)  

```python
def main(arr):
    range_ = find_min_max(arr)
    lo, hi = range_[0], range_[1]
    while (lo < hi):
        mid = round((lo+hi+0.001)/2)
        if not find_frequency(arr, (lo, mid-1)):
            hi = mid-1
            continue
        if not find_frequency(arr, (mid, hi)):
            lo = mid+1
            continue
    return lo
```
这样的方式更加简洁，主要看当时想到那种思路了。
### 总结一下
这道题说的就是如何去做二分查找的变体，很多时候并不直接使用二分法而是用二分法的思想去让问题减治或者分治，这样才便找到了突破口。
## NO4 从尾到头打印链表
兄弟萌，这是一个有趣的问题。很多同学看到这种问题后会想到改变指针去做，比如把每一个指针都反向的指，这样不就可以了吗。不过打印很多时候是一个只读操作，不能去改变原来的数据结构，那么就会思考这样一个问题：  
既然从头到尾打印链表如此的容易，我是不是让他们逆过来就好了呢？那么便很容易想到了后进先出的栈了，而想到了栈你就应该想到了递归对吗，因为计算机执行递归的时候就是用栈实现的呀。那么很容易的给出两个写法：
### 方法一 利用栈
python里面列表就可以很方便的实现栈了，那么我们的思路也就很简单，不多说上代码： 
  
```python
class ListNode(object):
    def __init__(self, val=None):
        self.val = val
        self.next = None

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)

def reverse_print(head):
    res = []
    while hasattr(head, 'next'):
        res.append(head.val)
        head = head.next
    while res!=[]:
        print(res.pop(), end="->")
    print("None")
```
### 方法二 递归
```python
def recursive_reverse_print(head):
    if head.next == None:
        print(head.val)
    else:
        recursive_reverse_print(head.next)
        print(head.val)
```
### 比较两种方法
看起来递归会容易，但是其实在现实情况中，不断的递归会导致栈溢出，其实还是用循环代替递归会好一些，但是第一种方法用了O(n)的空间去存放这个结果，所以各有利弊吧！

## NO5 重建二叉树
给出前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果都不含重复的数字。例如：输出前序遍历[12473568],中序遍历[47215386],则重建二叉树如下：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-11-25-044707.jpg)

这题明显比前面的题目更唬人一点，这是因为什么呢？反正不知道大家看到这题目的感受，我个人来讲的话，感觉得到递归的思想，就是找出左子树和右子树，重复的做一件事就OK了。但是我一时间不知道所谓的“这件事”是哪件事，我要做什么才能让这个树可以递归的进行下去。这是一个非常抽象的东西，我没办法去了解到，索性回归递归的套路，一步一步的考虑，化难为简，这是我所有写代码的思路。首先我们要明白的是，递归有三个要素：  
1. 递归出口  
2. 相互之间的关系  
3. 每一步如何做才能使问题规模缩小  

一步一步去想，首先，递归出口是什么呢？那么如果只有一个元素的时候，或者什么都没有的时候，这个时候肯定是不适合继续递归了吧，当然了，你如果觉得2个元素或者3个元素作为递归出口也可以，只不过0和1的情况你也要考虑到。
其次，每一步我要做点啥？我的想法是，首先，我要把左子树和右子树包含的元素找出来，**这时候假设我这个时候递归已经写完了，那么会出现的情况是我的左子树和右子树的元素只要传进递归函数的参数里，跳出来的就是两颗我需要的树了。**那么对于这两棵树，我要做的就仅仅是把我的根的左子树指向左子树，右孩子指向右子树就行了。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-11-25-044747.jpg)
### 代码如下：
```python
class TreeNode(object):
    def __init__(self, val=None):
        self.val = val
        self.lchild = None
        self.rchild =None

def rebuild_binarytree(pre_order:list, mid_order:list):
    assert len(pre_order) >= 0
    assert len(pre_order) == len(mid_order)
    if len(pre_order) == 0: # 递归出口
        return
    if len(pre_order) == 1: # 递归出口
        return TreeNode(pre_order[0])
    else:
        root_val = pre_order[0] # 找到根节点
        root = TreeNode(root_val)
        root_location = mid_order.index(root_val)
        left_pre = pre_order[1:root_location+1] # 左右子树的两种遍历方式找到
        right_pre = pre_order[root_location+1:]
        left_mid = mid_order[0:root_location]
        right_mid = mid_order[root_location+1:]
        root.lchild = rebuild_binarytree(left_pre, left_mid) # 递归
        root.rchild = rebuild_binarytree(right_pre, right_mid)
        return root
```   
## NO6 二叉树的下一个节点
给定一棵二叉树和其中的一个节点，如何找出中序遍历序列的下一个节点？树中的节点除了有两个分别指向左、右子节点的指针，还有一个指向父节点的指针。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-11-25-044814.jpg)  

这个树的中序遍历是 **[d b h e i a f c g]**  
这个问题看到之后有两个想法，第一个就是想办法找出root，然后中序遍历，然后找到后面一个元素。但这种方法在树比较庞大时必定会产生巨大资源浪费，所以我们可以想到了第二种方式，第二种方式大概的思路是这样的：首先我们要找出三个代表性的点：b h 和 i。下面分别去分析：  
1. b是有右子树的，在有右子树的情况下，我们应该找的是右子树上面最左侧的点。那么我们就沿着b的右子树一直向左搜索就可以了。  
2. h没有右子树，但是他是e的左子树，那说明下一个就是他的祖先节点，类似的节点还有d和f。  
3. i没有右子树，他又不是e的左子树，那么我们就要一直向上寻找，知道找到某个祖先的左节点是i所在的树，这样的话这个祖先就是i的下一个节点了。  
通过这种分类讨论，我们把节点分成了三种情况，这样基本就可以涵盖几乎所有节点了。当然，还有一些增加程序稳健性的异常捕捉要有的。   

```python
class TreeNode(object):
    def __init__(self, val=None):
        self.val = val
        self.lchild = None
        self.rchild = None
        self.ancestor = None

def is_lchild(child, ancestor:TreeNode):
    if ancestor.lchild == child:
        return True
    elif ancestor.rchild == child:
        return False
    else:
        return None

def find_next_element(root):
    if not root.lchild and not root.rchild and not root.ancestor:
        return None
    if root.rchild:
        root = root.rchild
        while root.lchild:
            root = root.lchild
        print("next is: ", root.val)
        return root
    elif not root.rchild:
        if not hasattr(root, 'ancestor'):
            return None
        ast = root.ancestor
        if is_lchild(root, ast):
            print("next is: ", ast.val)
            return ast
        else:
            try:
                while not is_lchild(root, ast):
                    root = ast
                    ast = root.ancestor
                print("next is: ", ast.val)
                return ast
            except:
                return None


a = TreeNode("a")
b = TreeNode("b")
c = TreeNode("c")
d = TreeNode("d")
e = TreeNode("e")
f = TreeNode("f")
g = TreeNode("g")
h = TreeNode("h")
i = TreeNode("i")
a.lchild = b
a.rchild = c
b.lchild = d
b.rchild = e
e.lchild = h
e.rchild = i
c.lchild = f
c.rchild = g
d.ancestor = b
e.ancestor = b
h.ancestor = e
i.ancestor = e
f.ancestor = c
g.ancestor = c
c.ancestor = a
b.ancestor = a
```
## NO7 栈和队列的相互实现
用两个栈可以实现队列，由两个队列也可以实现一个栈。想要完成本题的基础是了解栈和队列的基础操作，栈和队列是什么，有什么不同。那么可以想到他们主要的不同就是入栈/队列和出栈/队列的顺序不同了。那么目标就变成很简单的，用两个栈的之间的转换完成队列的操作，反之亦然。
### 两个栈实现队列
做个图的话，可以这么看：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2019-12-20-094541.png)
我们用stack1和stack2来实现queue，stack1是存放队列中数据的地方，stack2辅助完成pop操作即可。在上图中的情况可以看到，现在我们的pop要求输出1，而现在如果直接是stack的话则就是3，我们的做法是先把stack1中的元素依次移动到stack2中，可以看到stack2中的元素则实现了逆序，此时再pop即可。不过要注意的是，每次用完要把元素在放回stack1中。
我们现在的思路是修改了pop，其实如果修改push也是可以的，下面的代码就是修改push版本的：  

```python 
class Stack():
    def __init__(self):
        self.list_ = []

    def is_empty(self):
        return len(self.list_) == 0

    def pop(self):
        return self.list_.pop()

    def push(self, x):
        self.list_.append(x)

class Queue():
    def __init__(self):
        self.stack1 = Stack()
        self.stack2 = Stack()

    def is_empty(self):
        return self.stack1.is_empty()

    def pop(self):
        return self.stack1.pop()

    def push(self, x):
        while not self.stack1.is_empty():
            self.stack2.push(self.stack1.pop())
        self.stack1.push(x)
        while not self.stack2.is_empty():
            self.stack1.push(self.stack2.pop())
```

### 用两个队列实现栈
这个比刚才的要难一些，这是因为你可以看到队列的先进先出准则使它无法像栈那样的轻易形成逆序。在下采用的方法是用计数来做，思路也是修改pop函数，只要我知道这个队列中有几个元素，那么第n个元素则是我要pop的那个。  
举例来说，我现在有一个队列123，现在pop出来的应该是1，而我要让他pop出的是3，那么我先让queue1整体的移动到queue2，在这个过程中我可以计数，知道了有几个元素后，我每次从queue1中拿出一个元素，如果这不是第n次拿出来，我就把它放进queue2中，如果是的话，我就不放，那么最终queue2中的元素就会变成12，这样的话就形成了栈的结构了。  
代码如下：  

```python
class StackByQueue():
    def __init__(self):
        self.queue1 = Queue()
        self.queue2 = Queue()

    def is_empty(self):
        return self.queue1.is_empty()

    def __len__(self):
        res = 0
        while not self.queue1.is_empty():
            res += 1
            self.queue2.push(self.queue1.pop())
        self.queue1, self.queue2 = self.queue2, self.queue1
        return res

    def pop(self):
        length = len(self)
        res = 0
        while not self.queue1.is_empty():
            res += 1
            tmp = self.queue1.pop()
            if res != length:
                self.queue2.push(tmp)
        self.queue1, self.queue2 = self.queue2, self.queue1
        return tmp


    def push(self, x):
        self.queue1.push(x)
```

## NO8 旋转数组的最小值（二分查找）
### 查找
查找和排序是程序设计中非常重要的两个部分，面试中必然会经常考到。手写常见的排序算法并了解它们的复杂度，包括如何去优化他们这是十分重要的。而查找则没有那么复杂，一般不外乎**顺序查找、二分查找、哈希查找和二叉排序树**查找这几种方式。其中哈希查找和二叉排序树查找更多偏向于考察数据结构，例如哈希表的工作原理或者是哈希表冲突的解决方案等，而真正算法相关的就主要是二分查找了。  
如果面试中遇到了在排序后的数组中查找一个数字或者统计某个数字出现的次数，那么我们可以考虑二分查找的方式。  
传统的二分查找原理是不断的缩减规模，从而使查找规模逐次减半，这样我们可以知道其时间复杂度为O(log2n)。  
### 传统二分查找
最传统的二分查找应用在排序数组中查找某一个元素的位置，代码很简单：  

```python
def binary_search(tar, lst):
    cur1 = 0
    cur2 = len(lst)-1
    if lst[cur1] == tar:
        return cur1
    if lst[cur2] == tar:
        return cur2
    while cur1 < cur2:
        mid = int((cur1+cur2)/2)
        if lst[mid] < tar:
            cur1 = mid
        elif lst[mid] > tar:
            cur2 = mid
        else:
            return mid
```
二分查找的套路也很简单，就是先设置两个指针分别指向起点和终点，然后通过引入mid来实现规模的缩减，当然这里的判断是根据问题的不同而不同的，比如在这里我们要看的是tar和lst[mid]之间的关系，而当问题不同的时候，这里的比较可能也不尽相同了。这样不断的进行，就形成了二分查找。  
### 旋转数组的最小值
现在我们来看看这道题，**并不是所有的二分查找都需要在排序好的数组中完成**，本题便是提供了一个很好的例子。
首先，我们说数组{3,4,5,1,2}是数组{1,2,3,4,5}的一个旋转，我们要做的是找到旋转数组中的最小值，注意到旋转之后的数组可以划分成两个排序的子数组，而前面的元素都大于或者等于后面数组的元素。我们还注意到最小的那个数正好是两个数组之间的分界线。  
我们的想法和二分查找一样，不同的是找到mid后并不是直接查看到底是不是tar，而是要跟两个指针做比较，如果判断mid在前一部分，那么就让前一部分的指针指到mid，而如果mid指向了后面的数组，则让后面的指针指向mid，这样不断的往复，可以想到最终的结果是cur1指向了前一部分的末尾而cur2指向了后一部分的头，而我们要的那一个数字不正是cur2所指的部分嘛？  
但是这里有一些问题需要去考虑的，就是{10111}或者{11101}这种情况，这时候两个指针都相等，根本没法判断哪里是前和后，这种情况下，我们只能用顺序查找的方法了。  
上代码如下：  

```python
def find_min(lst: list):
    if len(lst)==1:
        return lst[0]
    cur1 = 0
    cur2 = len(lst)-1
    if lst[cur2] == lst[cur1]:
        return min(lst)
    while cur2-cur1 != 1:
        mid = int((cur2 + cur1)/2)
        if lst[cur2] > lst[mid]:
            cur2 = mid
        elif lst[cur1] < lst[mid]:
            cur1 = mid
    return lst[cur2]
```
通过这种方式，我们可以看到在半排序的数组中二分查找依然有效，但是我觉得其实最值得把握的是二分查找的思想：让问题规模减小，不断减治。    
## NO9 矩阵中的路径（回溯法）
### 回溯法
回溯法是升级版的暴力法，常常和递归用到一起，一般的情况都可以把问题分解成一个树形的结构，进行深度优先搜索，如果这条路不通，就退回到上一步，然后继续进行。  
本人感觉其实回溯法的思想并不难易理解，但是代码写起来总感觉不很顺畅。难点主要集中在这几点，首先如何把握递归的出口，其次是如何回退回去，这并没有一个普世的解决方案，只能是满满积累。
### 题目
现在有一个矩阵，设计算法找出符合某一个特定字符串的这矩阵中的一条路径。规定可以从任何地点开始，可以上下左右各种走，但是不能走重复的路。  
例如下图，bfce则是其中的一条路径，我们的函数实现的功能是输入路径和矩阵，返回一条路径：

想要完成这个任务，我们可以把路看成一个树的结构，每次我们都向下走一步，如果发现不行我们就回到上一个状态，走下一个节点，直到走通。路径的存放我们可以把它设计成一个栈，在找出一个符合条件的位置的时候，我们让这个位置入栈，而若是走不通的话，则让这个存放结果的栈弹出位置，但这个位置从此就加入了“黑名单”，我们以后搜索遍不考虑他了。  
具体实现的话，我把函数分成三个部分，除了主函数之外，我还用了找到头节点和判断能不能走通的两个函数，先上代码看一看：  

```python
def find_path(path, mat):
    res = []
    head = path[0]
    tmp = find_head(head, mat)
    for i in tmp:
        res.append(i)
        excpt = []
        while len(res) < len(path):
            candidate = find_neibour(res[-1], mat, path[len(res)], res+excpt)
            if not candidate:
                tmp_1 = res.pop()
                excpt.append(tmp_1)
            else:
                res.append(candidate)
    return res


def find_head(tar, mat):
    res = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if tar == mat[i][j]:
                res.append((i, j))
    return res


def find_neibour(location, mat, tar, excpt):
    res = []
    if location[1]-1 >= 0:
        if mat[location[0], location[1]-1] == tar:
            res.append((location[0], location[1]-1))
    if location[0]-1 >= 0:
        if mat[location[0]-1, location[1]] == tar:
            res.append((location[0]-1, location[1]))
    if location[1]+1 <= mat.shape[1]-1:
        if mat[location[0], location[1]+1] == tar:
            res.append((location[0], location[1]+1))
    if location[0]+1 <= mat.shape[0]-1:
        if mat[location[0]+1, location[1]] == tar:
            res.append((location[0]+1, location[1]))
    tmp = [x for x in res if x not in excpt]
    if len(tmp) != 0:
        return tmp[0]
    else:
        return None
```
其中find_head不必多说，自然是找到起始位置的函数，而find_neighbour则是在一定的**约束条件**下寻找符合的点坐标的函数，这个约束用excpt参数表示，这让每次的搜索有了“记忆性”，走过的路就不需要继续走了。而find_path是主函数，他每次从res这个栈中拿出一个元素做路径的搜索，但是搜索过的路径以及被“拉黑”的路径又被传到了find_neighbour的参数中去了。这样往复，直到最后栈中的元素与原始的字符串一样长了，这样标志了路径搜索的完成。  
## NO9 剪绳子
有一根绳子，请把绳子剪成m段（m、n都是整数并且大于1），想要直到如何切分才能让每段绳子的长度的乘机达到最大呢？
这个问题是为了知道动态规划和贪心的想法。这道题对应有这两种解法，分别有不同的想法。首先假设在绳子长为n的时候最大的乘积是f(n)，那么怎么考虑呢？首先考虑第一刀可以切1、2……直到(n-1)，这对应的剩下的长度分别为n-1……1，我们可以想到，**剩下的部分要么就切分，要么就不切分**，而切分的最大值已经知道了，就是f(x)，不切分的时候剩下的就是x。那么只需要比较这两种的一个最大值选一下就可以了。  
是不是感觉到了递归的味道？但是如果用递归的话，会频繁的计算大量的资源，显然使用动态规划的算法是更优化的，动态规划的要点是**从下往上，每次都把跟后面计算相关的变量存储下来**，这样的话我们就可以每次无需从头开始。对于本题的话，为了考虑全面，首先想一想特殊的情况，如果n是1或者0或者负数怎么办呢，考虑之后就可以完成我们的想法，代码如下：  

```python
def cut_the_rope(n):
    if n < 1:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        res = [0,1,1]
        for i in range(3, n+1):
            m = 0
            for j in range(1, i):
                tmp = max(j * res[i-j], j * (i-j))
                if tmp > m:
                    m = tmp
            res.append(m)
    return res[-1]
```    
可以看到，这种想法的代价是n方的时间复杂度和n的空间复杂度，这并不优秀。有没有更优秀的解法？有的，贪心策略就可以解决这种问题，我们执行的策略只有一个：**大于等于5的时候尽可能的剪3，等于4的时候不剪，等于3的时候为2，2的时候为1**。为什么可以这样想呢？首先我们可以证明在n>5的时候，3(n-3)>n，且3(n-3)>2(n-2)，这说明剪了3之后会更大，且我们要更多的切长度为3的绳子。  

```python
def greedy_cut_the_rope(n):
    if n >= 5:
        x = math.floor(n/3)
        return math.pow(3, x) * (n-3*x)
    if n <= 4:
        res = [0,1,1,2,4]
        return res[n]
```
可以看到，这更快了。  
## NO10 位运算
位运算是把数字用二进制表示之后，对每一位上0或者1的运算。其中主要的运算有与运算、或运算、异或运算、左移和右移这五种运算。  
位运算个人觉得还是要多掌握技巧，其中比较重要的两条我认为是：  

* 左移和右移代表乘法和除法，但是分别都是2的n次幂的乘除。
* 把一个整数减去1后再和原来的整数做位与运算，得到的结果相当于把整数的二进制表示中最右边的1变成了0。  

那么我们可以看一下这例题：
### 二进制中1的个数
查看一个数的二进制后的1的个数。
同理，还有两道类似的例题，分别是：  

* 查看一个数是不是2的整数次方。
* 输入两个整数m和n，计算需要改变m的二进制表示中的多少位才能得到n。  

这本例题和上述两道题都可以用位运算的技巧二去解决。  
如果一个数是2的整数次方的话，那么他的二进制数表示中只有一位是1。而第二题我们可以用m和n做异或运算，这样不同的数位就是1，就变成了二进制中1的个数的问题了。  
最后，我们简单的解决一下二进制中1的个数的问题好了，贼简单： 
 
```python
def find_one_in_bin(n):
    res = 0
    while n != 0:
        n = n & (n-1)
        res += 1
    return res
```
总结一下，我们可以看到在题目出现二进制或者是2的整数幂这种问题上，我们需要多考虑位运算。    

 
