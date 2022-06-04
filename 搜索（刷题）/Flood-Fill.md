# Flood Fill

#### 池塘奇数

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

