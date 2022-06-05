# Flood Fill

#### 解决的问题

> 可以在线性时间复杂度内找到某个点所在的连通块。



#### 池塘计数

> 农夫约翰有一片 N∗MN的矩形土地。
>
> 最近，由于降雨的原因，部分土地被水淹没了。
>
> 现在用一个字符矩阵来表示他的土地。
>
> 每个单元格内，如果包含雨水，则用”W”表示，如果不含雨水，则用”.”表示。
>
> 现在，约翰想知道他的土地中形成了多少片池塘。
>
> 每组相连的积水单元格集合可以看作是一片池塘。
>
> 每个单元格视为与其上、下、左、右、左上、右上、左下、右下八个邻近单元格相连。
>
> 请你输出共有多少片池塘，即矩阵中共有多少片相连的”W”块。
>
> ```go
> package main 
> 
> import "fmt"
> 
> const N int = 1010
> var g [N]string
> var st [N][N]bool       //标记
> var n, m int 
> 
> var ans = 0
> 
> // 通过深度优先遍历给所有的可到达的水打标记
> func dfs(x,y int) {
>     st[x][y] = true
>     dx, dy := [8]int{0,0,1,1,1,-1,-1,-1}, [8]int{-1,1,-1,0,1,-1,0,1}
>     
>     for i := 0; i < 8; i++ {
>         a, b := x+dx[i], y+dy[i]
>         if a >= 0 && a < n && b >= 0 && b < m && st[a][b] == false && g[a][b] == 'W' {
>             dfs(a,b)
>         }
>     }
> }
> 
> 
> func main() {
>     
>     fmt.Scanf("%d %d", &n, &m)
>     
>     for i := 0; i < n; i++ {
>         fmt.Scanf("%s", &g[i])
>     }
>     
>     for i := 0; i < n; i++ {
>         for j := 0; j < m; j++ {
>             if g[i][j] == 'W' && st[i][j] == false {
>                 dfs(i, j)
>                 ans++
>             }
>         }
>     }
>     
>     fmt.Println(ans)
> }
> ```
>
> 



#### 城堡问题

> ```go
> package main
> 
> import "fmt"
> 
> type Pair struct {
>     x, y int
> }
> 
> const N int = 55
> 
> var g [N][N]int
> var st [N][N]bool
> 
> var n,m int
> 
> func bfs(x,y int) int {
>     st[x][y] = true
>     q := make([]Pair, 0)
>     q = append(q, Pair{x, y})
>     
>     dx, dy := [4]int{0,-1,0,1}, [4]int{-1,0,1,0}
>     
>     area := 0
>     
>     for len(q) != 0 {
>         head := q[0]
>         q = q[1:]
>         area++
>         
>         for i := 0; i < 4; i++ {
>             a, b := head.x + dx[i], head.y + dy[i]
>             if a >= 0 && a < n && b >= 0 && b < m && st[a][b] == false && g[head.x][head.y]>>i & 1 == 0 {
>                 q = append(q, Pair{a, b})
>                 st[a][b] = true
>             }
>         }
>     }
>     
>     return area
> }
> 
> func main() {
>     fmt.Scanf("%d %d", &n, &m)
>     
>     cnt, area := 0, 0
>     
>     for i := 0; i < n; i++ {
>         for j := 0; j < m; j++ {
>             fmt.Scanf("%d", &g[i][j])
>         }
>     }
>     
>     for i := 0; i < n; i++ {
>         for j := 0; j < m; j++ {
>             if st[i][j] == false {
>                 area = max(area, bfs(i, j))
>                 cnt++
>             }
>         }
>     }
>     
>     fmt.Println(cnt)
>     fmt.Println(area)
> }
> 
> func max(a, b int) int {
>     if a > b {
>         return a
>     }
>     return b
> }
> ```
>
> 
