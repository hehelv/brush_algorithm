# BFS

#### 问题1：

> 给定一个 n×m 的二维整数数组，用来表示一个迷宫，数组中只包含 00 或 11，其中 00 表示可以走的路，11 表示不可通过的墙壁。
>
> 最初，有一个人位于左上角 (1,1)处，已知该人每次可以向上、下、左、右任意一个方向移动一个位置。
>
> 请问，该人从左上角移动至右下角 (n,m)处，至少需要移动多少次。
>
> 数据保证 (1,1) 处和 (n,m) 处的数字为 0，且一定至少存在一条通路。
>
> #### 输入格式
>
> 第一行包含两个整数 n 和 m。
>
> 接下来 n 行，每行包含 m 个整数（0 或 1），表示完整的二维数组迷宫。
>
> #### 输出格式
>
> 输出一个整数，表示从左上角移动至右下角的最少移动次数。
>
> > ```go
> > package main
> > 
> > import "fmt"
> > 
> > type pair struct {
> >     x,y int
> > }
> > 
> > const N int = 110
> > 
> > var (
> >     g [N][N]int
> >     d [N][N]int
> >     n,m int
> > )
> > 
> > func bfs() int {
> >     q := make([]*pair, 0)
> >     
> >     q = append(q, &pair{0,0})
> >     
> >     
> >     dx := [4]int{-1,1,0,0}
> >     dy := [4]int{0,0,1,-1}
> >     for len(q) != 0 {
> >         head := q[0]
> >         q = q[1:]
> >         
> >         for i := 0; i < 4; i++ {
> >             x, y := head.x + dx[i], head.y + dy[i]
> >             
> >             if x >= 0 && x < n && y >= 0 && y < m && g[x][y] == 0 && d[x][y] == -1 {
> >                 d[x][y] = d[head.x][head.y] + 1
> >                 q = append(q, &pair{x,y})
> >             }
> >         }
> >         
> >     }
> >     
> >     return d[n-1][m-1]
> > }
> > 
> > func main() {
> >     fmt.Scanf("%d%d", &n, &m)
> >     for i := 0; i < n; i++ {
> >         for j := 0; j < m; j++ {
> >             fmt.Scanf("%d", &g[i][j])
> >             d[i][j] = -1
> >         }
> >     }
> >     d[0][0] = 0
> >     
> >     fmt.Println(bfs())
> > }
> > ```
> >



#### 问题二

> 给定一个 nn 个点 mm 条边的有向图，图中可能存在重边和自环。
>
> 所有边的长度都是 11，点的编号为 1∼n1∼n。
>
> 请你求出 11 号点到 nn 号点的最短距离，如果从 11 号点无法走到 nn 号点，输出 −1−1。
>
> #### 输入格式
>
> 第一行包含两个整数 nn 和 mm。
>
> 接下来 mm 行，每行包含两个整数 aa 和 bb，表示存在一条从 aa 走到 bb 的长度为 11 的边。
>
> > ```go
> > package main
> > 
> > import "fmt"
> > 
> > const N int = 1e5 + 10
> > var h [N]int
> > var e, ne [N]int
> > var idx = 0
> > 
> > var n,m int
> > var d [N]int
> > 
> > func add(a, b int) {
> >     e[idx], ne[idx], h[a], idx = b, h[a], idx, idx + 1
> > }
> > 
> > // 宽搜出 u 到 各个点的最短距离
> > func bfs(u int) int {
> >     q := make([]int, 0)
> >     q = append(q, u)
> >     d[u] = 0
> >     
> >     for len(q) != 0 {
> >         head := q[0]
> >         q = q[1:]
> >         
> >         for i := h[head]; i != -1; i = ne[i] {
> >             j := e[i]
> >             if d[j] == -1 {
> >                 d[j] = d[head] + 1
> >                 q = append(q, j)
> >             }
> >         }
> >     }
> >     
> >     return d[n]
> > }
> > 
> > func main() {
> >     fmt.Scanf("%d %d", &n, &m)
> >     
> >     for i := 0; i <= n; i++ {
> >         h[i], d[i] = -1, -1
> >     }
> >     
> >     for i := 0; i < m; i++ {
> >         var a,b int
> >         fmt.Scanf("%d %d", &a, &b)
> >         add(a,b)
> >     }
> >     
> >     fmt.Println(bfs(1))
> > }
> > ```
> >
> > 