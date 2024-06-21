# 前言

​        yhf老师班的期末大作业是关于一学期所学的总结，这是我认为有效学习非常重要的一步。虽说我一直推崇学习的有效性，即终身学习——不会遗忘的学习，那么开卷考试是我比较喜欢的结构，因为不会有人永远记住所有case。这种意义上根本就没有必要去对抗遗忘，忘掉的东西，不会的东西，现学就可以了——这不代表考场临时报佛脚，而是将平时学到的思考方式和模版加以应用。废话到此为止，整个总结会分为笔试和机考两个部分。



# 笔试

## 目录

### 1.Sorting Algorithm

### 2.Data Structure





## 1.Sorting Algorithm

排序算法是前人留下的宝藏，完全的成熟意味着完全的学习。衡量一个算法的要素无非是时间复杂度，空间复杂度和稳定性，接下来也会从这几个方面进行学习。

### 1.Bubble Sort

冒泡排序是所有排序中最简单的一类，其算法的核心在于相邻元素的比，每一趟确定右侧一个值。

### 2.Selection Sort

选择排序是冒泡排序的「常数」改进算法，核心在于每一趟进行比较后留下最大或最小值进行交换，对列表的操作数远小于冒泡排序。

### 3.Quick Sort

快速排序则与前二者完全不同，它的思路的重点是将序列分成一系列的子序列，通过自我迭代的算法降低时间复杂度。比较常见的快速排序是使用双指针进行的：

```python
def quicksort(arr, left, right):
  if left < right:
    partition_pos = partition(arr, left, right)
    quicksort(arr, left, partition_pos)
    quickdort(arr, partition_pos, right)

def partition(arr, left, right):
  
    
```

快排通过对原列表的分割实现了算法的进步。

### 4.Merge Sort

归并排序延续了快排关于切割子序列的思路，但是通过强制的左右分划实现了最坏情况算法时间复杂度的提升。

```python
def MergeSort(arr):
  
```

### 5.Insertion Sort

插入排序的核心在于将数列分为已排好和未排好的两类，并将未排好的数挨个插入已排好的数列中。

### 6.Shell Sort

插入排序的升级版



一般排序算法的总结可以通过一张图实现：

|      Name      | Best    | Average |  Worst  | Memory | Stable |
| :------------: | ------- | :------ | :-----: | ------ | ------ |
|  Bubble Sort   | $n$     | $n^2$   |  $n^2$  | 1      | Yes    |
| Selection Sort | $n^2$   | $n^2$   |  $n^2$  | 1      | No     |
|   QuickSort    | $nlogn$ | $nlogn$ |  $n^2$  | $logn$ | No     |
|   Merge Sort   | $nlogn$ | $nlogn$ | $nlogn$ | $n$    | Yes    |
| Insertion Sort | $n$     | $n^2$   |  $n^2$  | 1      | Yes    |
|   Shell Sort   | $nlogn$ |         |         | 1      | No     |











## 2.Data Structure

### 1.线性表

线性表是一种和整数，浮点数，字符串等价的逻辑结构。其一般的储存方式有数组和链表两种。

#### 数组：

在空间中连续储存，访问的时间复杂度为$O(1)$，删除和插入元素的时间复杂度为$O(n)$。

#### 链表：

空间中非连续存储，每个元素包括储存元素和指向下一个元素储存位置的指针，访问的时间复杂度为$O(i)$，删除和插入元素的时间复杂度为$O(1)$。

# 机考

## 1.基础技术

### 1.搜索

dfs和bfs是两种最基础的搜索模式，其大致的算法核心是具有模块化意义的。

dfs大致可以由三个部分构成，一个是终止条件，一个是递归部分，一个是回溯条件。

```python
#马走日
def dfs(a, b, times):
	ans = 0
	if a < 0 or a >= n or b < 0 or b >= m:
		return 0
	if not check[a][b]:
  	return 0
	if times == n*m:
		return 1
	check[a][b] = False
	for dx, dy in ave:
		ans += dfs(a+dx, b+dy, times+1)
	check[a][b] = True
	return ans
```

当已经走过或在棋盘之外时return 0，走满棋盘时return 1，这是终止条件；以dx, dy重复进行行走是递归部分；回溯条件则是列表check标记未走过。从实现方式来看，dfs函数内部的自嵌套是dfs的核心，不过在某些场景，内部过度调动dfs函数会有时间复杂度较高的隐患。

bfs也可以用三个部分解释，一个是出队，一个是判断，一个是入队。

```python
#鸣人与佐助
from collections import deque
bfs = deque([list(origin) + [t] + [0]])
while bfs:
    node = bfs.popleft()
    i, j, k, l = node
    for dx, dy in ave:
        x, y = i+dx, j+dy
        if 0 <= x < m and 0 <= y < n and check[(x, y)] < k:
            if L[x][y] == "+":
                print(l+1)
                judge = False
                break
            elif L[x][y] == "*":
                check[(x, y)] = k
                bfs.append([x, y, k, l+1])
            else:
                if k >= 1:
                    check[(x, y)] = k
                    bfs.append([x, y, k-1, l+1])
        if not judge:
            break
    if not judge:
        break
```

不涉及内嵌套的bfs可以用一个while循环解决，使用deque的双端队列能减少操作时间。popleft是出队；从出队结点的周围寻找新的可能入队结点，判断是否符合入队标准和终止条件，这是判断；最后就是加入队列的操作。与dfs相比bfs的核心算法显得通俗易懂许多。bfs一般用于解决最短路径问题，因此在队列中的元素实际除了坐标之外还包括着距离参数。

### 2.并查集

并查集的主要构成是合并与查找。

```python
#宗教信仰
def find(x):
    if belief[x] != x:
        belief[x] = find(belief[x])
    return belief[x]

def union(x, y):
    x_r, y_r = find(x), find(y)
    if x_r > y_r:
        belief[y_r] = x_r
    elif x_r < y_r:
        belief[x_r] = y_r

j = 0
while True:
    j += 1
    m, n = [int(x) for x in input().split()]
    if m == 0:
        break
    belief, result = [i for i in range(m+1)], 0
    for _ in range(n):
        x, y = [int(x) for x in input().split()]
        union(x, y)
    for i in range(1,m+1):
        if find(i) == i:
            result += 1
    print(f"Case {j}: {result}")
```

find函数和union函数是并查集的核心，find最重要的是一点在于实时更新parent列表，这样在最后判断个数的时候才能使用查找方法。

### 3.二分查找

一种试错算法，确定结果的区间后，通过二分法进行验证查找，最后得到正确结果。

```python
#月度开销
def check(days, max_, m):
    sum, num = 0, 1
    for costs in days:
        if sum+costs > max_:
            num += 1
            sum = 0
        sum += costs
        if num > m:
            return False
    return True

def find(days, m):
    max_ = sum(days)
    min_ = max(days)
    result = max_
    while min_ <= max_:
        mid = (min_+max_)//2
        if check(days, mid, m):
            result = mid
            max_ = mid-1
        else:
            min_ = mid+1
    return result

n, m = [int(x) for x in input().split()]
days = [int(input()) for _ in range(n)]
print(find(days, m))
```

将二分的答案套入题目中给的过程进行模拟排错。

### 4.单调栈

```python
#单调栈
n = int(input())
arr = [int(x) for x in input().split()]
stack, result = [], [0]*n
for i in range(n):
    while stack and arr[i] > arr[stack[-1]]:
        result[stack[-1]] = i+1
        stack.pop()
    stack.append(i)
print(*result)
```

栈内标记下标，通过while循环在不单调发生时pop，得到目标后第一个大于目标元素的索引。

### 5.桶接收

当连通是通过模糊化判断存在时，可以通过边的记录接收数据，用空间换时间。

```python
#词梯
from collections import deque
bucket, dic = {}, {}
for j in range(int(input())):
    word = input()
    dic[word] = []
    for i in range(4):
        val = word[:i] + "_" + word[i+1:]
        if val not in bucket:
            bucket[val] = [word]
        else:
            for value in bucket[val]:
                dic[value].append(word)
                dic[word].append(value)
            bucket[val].append(word)
```



## 2.树

### 1.树类及经典输出

前序，中序，后序，层先四种输出 。

```python
class Treenode:
  def __init__(self, value):
    self.left = None
    self.right = None
    self.val = value

def pre_result(root):
  if not root:
    return []
  output = [root.value]
  output.extend(pre_result(root.left))
  output.extend(preUresult(root.right))
  return output

def in_result(root):
  if not root:
    return []
  output = []
  output.extend(in_result(root.left))
  output.append(root.value)
  output.extend(in_result(root.rihgt))
  return output

def post_result(root):
  if not root:
    return []
  output = []
  output.extend(post_result(root.left))
  output.extend(post_result(root.right))
  output.append(root.value)
  return output

from collections import deque
def level_result(root):
  queue, output = [root], []
  while queue:
    node = queue.leftpop()
    output.append(node.value)
    if node.left:
      queue.append(node.left)
    if node.right:
      queue.append(node.right)
  return output
```

### 2.种树大法

面对没有固定输入顺序数据的接收。

```python
treenode = [Treenode(i) for i in range(n)]
```

直接将所有节点储存在内存中，方便调用。

```python
#遍历树
treenode = {}
def plant_tree(n):
    for _ in range(n):
        order = [int(x) for x in input().split()]
        root, child = order[0], order[1:]
        if root not in treenode:
            treenode[root] = [Treenode(root), False]
        root_ = treenode[root][0]
        for node in child:
            if node not in treenode:
                treenode[node] = [Treenode(node), True]
            treenode[node][1] = True
            node_ = treenode[node][0]
            root_.children.append(node_)
    for node in treenode.values():
        if not node[1]:
            return node[0]
```

将节点和是否有parent的布尔判断式放在一个treenode字典中，用内存换输入。

完全由数字构成的树也可以使用字典建树，肉眼可见的高效，缺点是需要has_parent判断根节点。

```python
#遍历树
treenode = {}
def plant_tree(n):
  for _ in range(n):
    order = int(input())
    root, child = order[0], order[1:]
    treenode[root] = child
```

### 3.数学优化

二叉树本身具有相当强烈的数学结构，尤其是对于满二叉树。

```python
#树的重量
import math
k, n = [int(x) for x in input().split()]
child_value, value = [0] * 2**k, [0] * 2**k
for _ in range(n):
    order = [int(x) for x in input().split()]
    val = 0
    root = num = order[1]
    depth = k-int(math.log2(root))
    if order[0] == 2:
        while num:
            val += value[num]
            num = num//2
        print(val*(2**depth-1)+child_value[root])
    else:
        value[root] += order[-1]
        val = (2**depth-1)*order[-1]
        num = num//2
        while num:
            child_value[num] += val
            num = num//2
```

一个列表将所有孩子的“重量“记录下来，一个列表记录子树的重量和，避免了繁杂的搜索。

## 3.图

### 1.拓扑排序

拓扑排序是判断一个有向图是否有环的基本方式，本质在于使用入度进行判断，当图中有环时，该图不能被拓扑排序，从而走过一遍后入度不全为零。

```python
# 舰队、海域出击！
# 用类似bfs的搜索手段，但核心是入度数的判断，deque很精髓
from collections import deque
def judge():
    stack = deque([])
    for i in range(1, m+1):
        if degree[i] == 0:
            stack.append(i)
            visited[i] = True
    while stack:
        ori = stack.popleft()
        for des in ave[ori]:
            degree[des] -= 1
            if degree[des] == 0:
                stack.append(des)
                visited[des] = True
    return False if all(x == 0 for x in degree[1:]) else True

for _ in range(int(input())):
    m, n = [int(x) for x in input().split()]
    ave = {i: [] for i in range(m+1)}
    degree = [0 for i in range(m+1)]
    check = [False for i in range(m+1)]
    visited = [False for i in range(m+1)]
    for _ in range(n):
        a, b = [int(x) for x in input().split()]
        ave[a].append(b)
        degree[b] += 1
    if judge():
        print("Yes")
    else:
        print("No")
```

### 2.dijkstra

在各边权值均为正的图中寻找两点间最短路径。

```python
#兔子与樱花
#因为要求记录路径，所以以列表方式在每条路上标记。
import heapq
def find(ori, des):
    visit = {go: False for go in ave}
    stack = []
    heapq.heappush(stack, [0, [ori]])
    while stack:
        ori_ = heapq.heappop(stack)
        now = ori_[1][-1]
        visit[now] = True
        if now == des:
            return ori_[1]
        for des_, dis in ave[now].items():
            if not visit[des_]:
                heapq.heappush(stack, [dis+ori_[0], ori_[1]+[des_]])
```

visit在此实现剪枝，import heapq后不再需要记录每条路的当前距离，heapq的写法一定要熟悉，以第一个元素进行排序。



```python
#道路
#记录金币数，替代visit实现剪枝。
import heapq
def find():
    tak = [10001 for i in range(n+1)]
    queue = []
    heapq.heappush(queue, [0, 1, 0])
    while queue:
        d, now, t = heapq.heappop(queue)
        tak[now] = t
        if now == n:
            return d
        for des, val in ave[now].items():
            d_, t_ = val
            if t_+t <= k and t_+t < des[tak]:
                heapq.heappush(queue, [d_+d, des, t_+t])
    return -1
```

接受数据一般采用字典写出的临接表，采用items()进行遍历。

### 3.Prim

图中的最小生成树算法。

```python
#兔子与星空
import heapq
def find():
    stack, result, times = [], 0, 0
    heapq.heappush(stack, [0, "A"])
    while stack:
        now = heapq.heappop(stack)
        if not visit[now[1]]:
            visit[now[1]] = True
            times += 1
            result += now[0]
            if times == n:
                return result
            for des, dis in dic[now[1]].items():
                if not visit[des]:
                    heapq.heappush(stack, [dis, des])
```

本质上是dijkstra的贪心，算法本身是相当通俗的，找到所有连接未访问节点的最短路径即可。
