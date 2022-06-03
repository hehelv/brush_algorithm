# DFS

#### 排列问题

> 题目描述：
>
> > 给定一个整数 nn，将数字 1∼n1∼n 排成一排，将会有很多种排列方法。
> >
> > 现在，请你按照字典序将所有的排列方法输出。
>
> ```go
> package main 
> 
> import "fmt"
> 
> const N int = 10
> var n int
> 
> var p [N]int
> var visit [N]bool
> 
> 
> func main() {
>     
>     fmt.Scanf("%d", &n)
>     
>     dfs(0)
>     
> }
> 
> func dfs(u int) {
>     if u == n {
>         for i := 0; i < n; i++ {
>             fmt.Printf("%d ", p[i])
>         }
>         fmt.Println()
>     }
>     
>     for i := 1; i <= n; i++ {
>         if !visit[i] {
>             p[u] = i
>             visit[i] = true
>             dfs(u+1)
>             // 恢复现场
>             visit[i] = false
>         }
>     } 
> }
> ```
>
> 





#### N皇后问题

> 题目描述：
>
> > n−n−皇后问题是指将 nn 个皇后放在 n×nn×n 的国际象棋棋盘上，使得皇后不能相互攻击到，即任意两个皇后都不能处于同一行、同一列或同一斜线上。
>
> ```go
> package main
> 
> import "fmt"
> 
> const N int = 10
> var col [N]bool      // 列是否被占有
> var d, ud [N*2]bool
> var n int
> var g [N][N]byte
> 
> func main() {
>     fmt.Scanf("%d", &n)
>     for i := 0; i < n; i++ {
>         for j := 0; j < n; j++ {
>             g[i][j] = '.'
>         }
>     }
> 
>     dfs(0)
> }
> 
> 
> // 枚举每一行
> func dfs(u int) {
>     if u == n {
>         for i := 0; i < n; i++ {
>             for j := 0; j < n; j++ {
>                 fmt.Printf("%c ", g[i][j])
>             }
>             fmt.Println()
>         }
>         fmt.Println()
>     }
>     
>     
>     // 遍历列， 看那一列可以用
>     for i := 0; i < n; i++ {
>         if col[i] == false && d[i+u] == false && ud[i-u+N] == false {
>             g[u][i] = 'Q'
>             col[i], d[i+u], ud[i-u+N] = true, true, true
>             dfs(u+1)
>             col[i], d[i+u], ud[i-u+N] = false, false, false
>             g[u][i] = '.'
>         }
>     }
> }
> ```
>
> 





> 树的重心
>
> 题目描述：
>
> > 给定一颗树，树中包含 n 个结点（编号 1∼n）和 n−1 条无向边。
> >
> > 请你找到树的重心，并输出将重心删除后，剩余各个连通块中点数的最大值。
> >
> > 重心定义：重心是指树中的一个结点，如果将这个点删除后，剩余各个连通块中点数的最大值最小，那么这个节点被称为树的重心。
> >
> > #### 输入格式
> >
> > 第一行包含整数 n，表示树的结点数。
> >
> > 接下来 n−1 行，每行包含两个整数 a 和 b，表示点 a 和点bb 之间存在一条边。
> >
> > #### 输出格式
> >
> > 输出一个整数 m，表示将重心删除后，剩余各个连通块中点数的最大值。
>
> ```go
> package main
> 
> import "fmt"
> 
> const N int = 1e5 + 10
> var h [N]int
> var e, ne [N*2]int
> var idx = 0
> var n int
> 
> var st [N]bool
> var ans = N
> 
> // 添加a到b的无向边边
> func add (a, b int) {
>     e[idx], ne[idx], h[a] = b, h[a], idx
>     idx++
> }
> 
> // 返回以该节点为根的树，含有的节点个数
> func dfs(u int) int {
>     st[u] = true
>     
>     size, sum := 0, 0
>     
>     for i := h[u]; i != -1; i = ne[i] {
>         j := e[i]
>         
>         if st[j] == true {
>             continue
>         }
>         
>         s := dfs(j)
>         sum += s
>         size = max(size, s)
>     }
>     
>     size = max(size, n- sum-1)
>     ans = min(ans, size)
>     return sum + 1
> }
> 
> func min(a, b int) int {
>     if a < b {
>         return a
>     }
>     return b
> }
> 
> func max(a, b int) int {
>     if a > b {
>         return a
>     }
>     return b
> }
> 
> 
> func main() {
>     fmt.Scanf("%d", &n)
>     var a,b int
>     
>     for i := 0; i <= n; i++ {
>         h[i] = -1
>     }
>     
>     for i := 0; i < n; i++ {
>         fmt.Scanf("%d %d", &a, &b)
>         add(a, b)
>         add(b, a)
>     }
>     
>     dfs(1)
>     
>     fmt.Println(ans)
> }
> 
> 
> ```
>
> 
