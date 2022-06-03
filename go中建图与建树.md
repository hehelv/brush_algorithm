# 建图

#### 邻接矩阵

> 一个二维数组：
>
> ```go
> var g [N][N]int
> ```





#### 邻接表

> 用map（参考C++里面的vector用法）：
>
> ```go
> map[int]([]int)
> ```





# 建树

#### 动态树

> 节点：
>
> ```go
> type node struct {
>     val int
>     child []*node
> }
> ```
>
> 



#### 静态树

> 节点：
>
> ```go
> // 邻接表存储
> var h [N]int	//头结点编号
> var ne[N], e[N]	//ne表示每个节点的下一个节点， e存储的是每个节点存的信息
> var idx = 0		//用到的节点编号
> ```
>
> 
