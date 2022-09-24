```c++
//数据初始化INF和-INF
memset(f,0x3f,sizeof f);//INF
memset(f,0xcf,sizeof f);//-INF
```
### 数字三角形模型
```c++
/*
 * 数字三角形即形如f[i][j]<-(f[i-1][j],f[i][j-1])转移方程的模式
 * */
/*
    1.摘花生：给定一个二维矩阵，从左上角出发，每次只能向右或向下走，问从右下角出来时的经过点权值的最大值
        f[i][j]=max(f[i-1][j],f[i][j-1])+a[i][j]
        cout<<f[n][m]
    2.最低同行费用：题目同上，但求的是最小值
        f[i][j]=min(f[i-1][j],f[i][j-1])+a[i][j]
        cout<<f[n][m];
    3.方格取数：从左上角到右下角按上述方式走两次，经过相同的点只会计算一次权值，求最大值
        定义状态f[k][i][j],表示走k步，A点横坐标为i，B点横坐标为j时的权值
        对于i==j特殊处理
        for(int k=2;k<=n+m;++k)
                for(int i1=1;i1<=n;++i1)
                    for(int i2=1;i2<=n;++i2){
                        int j1 = k-i1,j2=k-i2;
                        if(i1>=1&&i1<=n&&i2>=1&&i2<=n&&j1>=1&&j1<=m&&j2>=1&&j2<=m){
                            int t = a[i1][j1];
                            if(i1!=i2)t+=a[i2][j2];
                            //四个方向
                            int &cur = f[k][i1][i2];
                            cur = max(cur,f[k-1][i1][i2]+t);
                            cur = max(cur,f[k-1][i1-1][i2]+t);
                            cur = max(cur,f[k-1][i1][i2-1]+t);
                            cur = max(cur,f[k-1][i1-1][i2-1]+t);
                        }
                    }
            cout<<f[m+n][n][n];
    4.传纸条：题目约束不能经过相同的点，但是由于所有点都是正数，因此最优解会直接避免这种情况
*/
```

### 最长上升子序列模型
拦截导弹

拦截系统

最长**公共***上升*子序列
```c++
/*
    最长上升子序列模型
        1.f[i]=max(f[k])+1,a[k]<a[i]&&k<i
            初始化f[i]=1;
        2.使用二分查找优化：大则添加，小则替换（上升子序列小于等于替换，不下降子序列大于替换）
            b[1]=a[1];
            for(int i=2;i<=n;++i)
                if(a[i]>b[len])b[++len]=a[i];//大则添加
                else{//小于等于则替换->二分查找后替换
                    int l = 1,r=len;
                    while(l<r){//二分查找模板
                        int mid = (l+r)/2;//mid在左侧不添加
                        if(b[mid]>=a[i])r = mid;
                        else l = mid+1;
                    }
                    b[l]=a[i];
                }
    1.怪盗基德，求从左向右和从右向左的最长上升子序列的最大值，使用二分模板即可。
    2.登山，[1~i]的最长上升子序列长度与[i~n]的最长下降子序列（反向的最长上升子序列）的长度和减去1
        使用dp方法
    3.合唱队形，方法同登山
    4.友好城市，一条河两岸存在港口，一对对友好城市分布在两岸，互为友好城市的港口才能贸易，问在航线不相交的情况下至多有多少条航线。
        f[i]表示前i开通时，前i个港口的最大航线数 初始化f[i]=1
        f[i]=max(f[i],f[k]+1),k<i且k航线与i航线不相交
        max(f[i])为最终答案
    5.最大上升子序列的和：f[i]表示以i结尾的最大上升子和，初始化为a[i]
        f[i]=max(f[i],f[j]+a[i]),a[j]<a[i],j<i;
    6.拦截导弹：存在一个这样的系统，每次只能拦截不高于上一次的导弹，给定一群导弹的高度，问一次最多能拦截多少导弹，要全部拦截需要多少个这样的系统？
        问题一为最大不上升序列，使用模板即可。
        问题二考虑这样的一个贪心策略，对于每个导弹，都用能够拦截的高度最低的系统拦截，对于不能拦截的导弹，则增加一套系统。实际上这种方法就是最大上升子序列的二分查找优化：大则增加（所有系统都无法拦截了，只能增加一套系统拦截），小于等于（该导弹可以拦截）则替换（使用能够拦截的高度最低的系统进行拦截，拦截之后，其值就替换成了当前高度）
    7.拦截系统：该拦截系统只能单调向上或单调向下拦截，问最小需要多少套拦截系统？
        思路：对于每个导弹拥有两种情况，归入单调向下或者单调向上，通过深搜取所有答案的最小值
        爆搜剪枝：仅当当前拦截系统套数小于当前最优情况时，才进行搜索
        dfs通过定义全局变量求最小值，通过与当前最小值比较进行剪枝
        #include "iostream"
        using namespace std;
        int a[55],ans,up[55],down[55],n;
        void dfs(int pos,int len_up,int len_down){
            if(len_up+len_down>=ans)return;//剪枝
            if(pos==n+1){//到达搜索终点 更新最小值
                ans = min(ans,len_down+len_up);
                return;
            }
            //添加到上升子序列中
            int k = 1;
            while(k<=len_up&&a[pos]>up[k])k++;//找到第一个比它大的点 进行替换
            int t = up[k];//保存 用于深搜恢复
            up[k]=a[pos];
            dfs(pos+1,max(len_up,k),len_down);
            up[k]=t;//恢复状态
            //添加到下降子序列中
            k = 1;
            while(k<=len_down&&a[pos]<down[k])k++;
            t = down[k];
            down[k]=a[pos];
            dfs(pos+1,len_up,max(len_down,k));
            down[k]=t;
        }
        int main(){
            while(cin>>n,n){
                for(int i=1;i<=n;++i)cin>>a[i];
                ans = n;
                dfs(1,0,0);
                cout<<ans<<endl;
            }
        }
    8.最长公共上升子序列：最长公共上升子序列前置知识为最长上升子序列和最长公共子序列
        最长上升子序列:f[i]=max(f[j]+1),j<i&&a[j]<a[i]
        最长公共子序列:f[i][j]存储a[1~i],b[1~j]的最长公共子序列
            f[i][j]=f[i-1][j-1],a[i]==b[j]
            f[i][j]=max(f[i][j-1],f[i-1][j]),a[i]!=b[j],即a[i]不在和b[j]不在公共子序列中的最大值
        最长公共上升子序列：
            状态定义：f[i][j]表示a[1~i],b[1~j]中且以b[j]结尾的最长公共子序列的长度的最大值
            转移方程以a[i]是否在公共子序列中进行划分
                a[i]不在公共子序列中:此时a[i]的存在与否与f[i][j]没有影响，即
                    f[i][j]=f[i-1][j]
                a[i]在公共子序列中，此时要求a[i]=b[j]:此时仅考虑a[1~i-1]和b[1~j-1],此时枚举以b[k],1<=k<i的每种情况，即a[1~i-1]与b[1~k]的最长上升公共子序列的最大值+1（1为a[i]=b[j]）
                    f[i][j]=max(f[i-1][k]+1),b[k]<b[j]=a[i],k<j
            朴素方法:由状态转移方程得f[i][j]均由f[i-1][1~j]得到，则有
                for(int i=1;i<=n;++i)
                    for(int j=1;j<=m;++j){
                        f[i][j]=f[i-1][j];
                        if(a[i]==b[j]){
                            int maxv = 1;
                            for(int k=1;k<j;++k)//枚举a[1~i-1]和b[1~k]
                                maxv = max(maxv,f[i-1][k]+1);
                            f[i][j]=max(f[i][j],maxv);
                        }
                    }
                //由于状态表示以b[j]结尾的最大值，因此需要枚举f[n][j]
                int ans = 0;
                for(int i =1;i<=m;++i)ans = max(ans,f[n][i]);
                cout<<ans;
            对上述递推过程进行优化：maxv的含义为b[1~j-1]中小于a[i]的位置的最长公共子序列的最大值，对于每个b[j]=a[i],都会进行一次k循环，而k循环存在大量重复计算，可以在对j进行遍历时保存并更新maxv从而提高效率。
                for(int i=1;i<=n;++i){
                    int maxv = 1;
                    for(int j=1;j<=m;++j){
                        f[i][j]=f[i-1][j];
                        if(a[i]==b[j])f[i][j]=max(f[i][j],maxv);
                        if(b[j]<a[i])maxv=max(maxv,f[i-1][j]+1);//维护[1~j]中满足b[k]<a[i]的f[i-1][k]的最大值，用于j+1的更新
                    }
                }
 * */
```

### 背包问题
```c++
/*
    背包问题有：
        基础背包：01背包，完全背包，多重背包，分组背包
        衍生背包：混合背包，多维背包...
        背包计数：01背包计数，完全背包计数...
    1.采药：01背包问题
    2.装箱问题：每个东西有一定体积，最终能空出的最小空间是多少
        01背包，用总体积减去能装得下的最大体积即可
    3.宠物小精灵之收服：二维01背包，拥有两种代价。
        f[i][j][k]=max(f[i-1][j][k],f[i-1][j-cji][k-cki]+wi)
        使用滚动数组逆序更新
        同时，本题要求球可以为0但是体力不能为0，因此逆序更新时保证体力最小值为1.最终求解时需要求得最大剩余体力，得到最大收服小动物数量后，遍历整个数组，计算体力最大值max(m-j+1) f[i][j]==c
    4.数字组合：给定一个整数序列，问和为M的组合有多少种
        01背包计数问题
        f[i][j]=f[i-1][j-val],初始化f[1][0]=1;
        使用滚动数组优化
            f[0]=1;
            while(n--){
                cin>>k;
                for(int i=m;i>=k;--i)f[i]+=f[i-k];
            }
            cout<<f[m];
    5.买书：书的价格有10元20元50元100元，问小明有多少种方案将将手上的钱全部用掉用来买书
        多重背包计数问题
            f[i][j]=f[i-1][j]+sum(f[i-1][j-k*vi])
        使用滚动数组优化f[i]+=f[i-vi],从前往后遍历时，f[i-vi]就是上式的sum(f[i-1][j-k*vi])部分
            f[0]=1;
                for(int i=1;i<=4;++i)
                    for(int j=b[i];j<=1000;++j)
                        f[j]+=f[j-b[i]];
    6.货币系统：一种货币存在不同的面值，问一个价格有多少不同种面值的组合方式
        同上：多重背包计数问题
    7.多重背包问题Ⅲ,数据很强，二进制优化无法通过
        本题是多重背包单调队列优化，效率比二进制优化要高。
        单调队列需要使用模拟队列
            模拟队列int q[N],hh=0,tt=-1;q[++tt]=val;if(hh<=tt){...};
            单调队列：用于维护滑动窗口最大值，最小值，队列存储下标，便于与窗口边界进行比较
                当前点i，窗口大小k
                while(hh<=tt&&q[hh]<i-k+1)hh++;//超出窗口边界的元素pop
                while(hh<=tt&&a[q[tt]]>=a[i])t--;//删除队列中比当前点大的元素（队列维护最小值）
                q[++tt]=i;//加入队列
                if(条件)cout<<a[q[hh]]<<" ";//队头元素即为区间最值
        多重背包的朴素做法
            for(int i=1;i<=N;++i)
                for(int j=V;j>=v[i];--j)//滚动数组逆序更新
                    for(int k=1;k<=s[i]&&j>=k*v[i];++k)//穷举放入元素个数
                        f[j]=max(f[j],f[j-k*v[i]]+k*w[i]);//更新最大值
        通过对原来数组复制一遍，得以顺序更新
            for(int i=1;i<=N;++i){
                memcpy(g,f,sizeof(f));
                for(int j=v[i];j<=V;++j)
                    for(int k=1;k<=s[i]&&j>=k*v[i];++k)
                        f[j]=max(f[j],g[j-k*v[i]]+k*w[i]);
            }
        从朴素方法的更新过程以及结果可以得知，f[j]是从f[j-v[i]],f[j-s[i]*v[i]]这个大小为s[i]的窗口进行更新的，由此得到单调队列优化,单调队列维护旧数组的下标
            for(int i=1;i<=N;++i){
                memcpy(g,f,sizeof(f));
                for(int j=0;j<v[i];++j){//分组更新
                    hh = 0 ,tt=-1;
                    for(int k=j;k<=V;k+=v[i]){
                        //删除不在[k-s[i]*v[i],k-v[i]]中的元素
                        while(hh<=tt&&q[hh]<k-s[i]*v[i])h++;
                        //更新f[k]
                        if(hh<=tt)f[k]=max(f[k],g[q[hh]]+(k-q[hh])/v[i]*w[i]);
                        //用旧数组更新单调队列
                        while(hh<=tt&&g[k]>=g[q[tt]]+(k-q[tt])/v[i]*w[i])tt--;
                        q[++tt]=j;
                    }
                }
            }
    8.庆功会：多重给背包问题，统一用单调队列优化来写，代码同上
    9.混合背包问题
        混合背包的处理方式同一般背包，对于不同的背包采用不同的策略，01背包逆序更新，完全背包顺序更新，多重背包单调队列优化
        cin>>N>>V;
            for(int i=1;i<=N;++i){
                cin>>v>>w>>s;
                if(s==-1){//01
                    for(int j=V;j>=v;--j)f[j]=max(f[j],f[j-v]+w);
                }
                else if(s==0){//完全
                    for(int j=v;j<=V;++j)f[j]=max(f[j],f[j-v]+w);
                }else{//多重
                    memcpy(g,f,sizeof(f));
                    for(int j=0;j<v;++j){
                        int hh=0,tt=-1;
                        for(int k=j;k<=V;k+=v){
                            while(hh<=tt&&q[hh]<k-v*s)hh++;
                            if(hh<=tt)f[k]=max(f[k],g[q[hh]]+(k-q[hh])/v*w);
                            while(hh<=tt&&g[k]>=g[q[tt]]+(k-q[tt])/v*w)tt--;
                            q[++tt]=k;
                        }
                    }
                }
            }
            cout<<f[V];
    10.二维01背包，逆序更新
        cin>>N>>V>>M;
            for(int i=1;i<=N;++i){
                cin>>v>>m>>w;
                for(int j=V;j>=v;--j)
                   for(int k=M;k>=m;--k)
                    f[j][k]=max(f[j][k],f[j-v][k-m]+w);
            }
            cout<<f[V][M];
    11.潜水员，二维背包问题，问满足条件的最小重量是多少。
        由此引出01背包的三种情况
            1.恰好装v：初始化f[i,i>0]=inf,f[0]=0,f[i]=max(f[i],f[i-v]+w)
            2.至多装v:初始化f[all]=0,f[i]=max(f[i],f[i-v]+w)
            3.至少装v：初始化f[0]=0,f[i,i>0]=inf,f[i]=max(f[i],f[max(0,i-v)]+w)
                至少装v时，f[i]可以由f[j,j>=i-v]转移得到，此时i-j会小于0,取max(0,i-v)
        cin>>n>>m>>k;
            memset(f,0x3f,sizeof(f));
            f[0][0]=0;
            for(int i=1;i<=k;++i){
                cin>>a>>b>>c;
                for(int x=n;x>=0;--x)
                    for(int y=m;y>=0;--y)
                        f[x][y]=min(f[x][y],f[max(0,x-a)][max(0,y-b)]+c);
            }
            cout<<f[n][m];
    12.机器分配：一共M台设备分给N家公司，每家公司获得获得不同数量的设备会产生不同的价值，求价值最大的方案（任意输出一个即可）。
        同背包问题，f[i][j]表示前i家公司分配j台设备的最大值 
            f[i][j]=max(f[i-1][j],f[i-1][j-k]+val[i][k])
        输出方案的方法：在更新时记录前区，迭代输出。
    13.开心的金明：普通01背包
    14.有依赖的背包问题
        核心观点：f[u][j]的第一维代表的是节点，对应树形DP问题；第二维是体积，代表一维分组背包问题
        f[u][j]：在”以uu为根节点的子树”中选，节点uu必选，所选体积不超过jj，的所有方案中，价值最大的方案的价值
        计算f[u][j]时，先通过分组背包的方式计算在所有孩子节点中选的最大价值，最后再考虑节点uu。设节点uu有pp个孩子节点，那么就相当于有pp组物品。
        物品组为1∼p1∼p，总共可用的体积为m−v[u]m−v[u]。现在我们眼中只有pp组物品，先不考虑父亲节点，只考虑计算这pp组物品的最大价值
        根据分组背包的做法，首先枚举物品组，也就是节点uu的某个孩子sonson，对应代码的for (int i = h[u]; i != -1; i = ne[i])
        其次枚举体积jj，也就是考虑在1∼son1∼son的物品组中选，所选体积不超过jj的最大价值。
        最后枚举在物品组sonson中选体积为kk的物品，k∈[0,j]k∈[0,j]，因为1∼son1∼son的物品组一共选了不超过jj的体积
        状态转移是f[u][j] = max(f[u][j], f[u][j - k] + f[son][k])。
        f[u][j-k]: 由于体积jj是从大到小枚举，所以这里f[u][j-k]表示在1∼son−11∼son−1的物品组中选，体积不超过j−kj−k的方案的最大价值，这里省略了表示”在1∼son−11∼son−1的物品组中选”的维度，相当于一维的分组背包。而f[u][j]的第一维代表的不是背包问题，而是树形DP，第二维代表的才是分组背包。所以这道题是树形DP和背包的综合。
        f[son][k]: 由于状态转移之前已经递归调用了dfs(son)，所以以sonson为根的子树已经计算好了。f[son][k]表示在以sonson为根的子树中选，所选体积不超过kk的最大价值
        综上，f[u][j−k]+f[son][k]f[u][j−k]+f[son][k]的前半部分更趋向于分组背包，后半部分趋向于树形DP
        计算完f[u][*]之后，f[u][*]代表的其实是在节点uu的所有子树1∼p1∼p中选的最大价值，没有计算uu的价值，所以需要加上最后的两个for循环
        #include "iostream"
        using namespace std;
        const int SIZE = 110;
        int h[SIZE],idx,N,V,f[SIZE][SIZE],w[SIZE],v[SIZE];
        
        struct Edge{
            int next;
            int ver;
        }edge[SIZE];
        
        void add(int a,int b){
            edge[++idx].ver = b;
            edge[idx].next  = h[a];
            h[a]=idx;
        }
        
        void dfs(int cur){
            int ne = h[cur];
            while(ne){//对于每个分组
                int pos = edge[ne].ver;
                dfs(pos);
                for(int i=V-v[cur];i>=0;--i){//根节点必选，预留出根节点空间 对于每个体积，逆序是为了使用滚动数据节省空间
                    for(int j=0;j<=i;++j)
                        f[cur][i]=max(f[cur][i],f[cur][i-j]+f[pos][j]);//从分组中选择一个
                }
                ne = edge[ne].next;
            }
            for(int i=V;i>=v[cur];--i)f[cur][i]=f[cur][i-v[cur]]+w[cur];//增加根节点
            for(int i=0;i<v[cur];++i)f[cur][i]=0;//根节点不能选的话，价值为0
        }
        int main(){
            cin>>N>>V;
            int root,p;
            for(int i=1;i<=N;++i){
                cin>>v[i]>>w[i]>>p;
                if(p==-1)root =i;
                else add(p,i);
            }
            dfs(root);
            cout<<f[root][V];
        }
    15.01背包计数问题：初始化f[i]=1;
        转移分两种情况：大于f[i]=f[i-v]，等于f[i]+=f[i-v]
        最终结果f[V]
    16.背包问题求具体方案：分为两种情况，任意一种方案和字典序最小方案
        1.任意一种方案，不使用滚动数组优化
            f[i][j]==f[i-1][j]表示不选第i个物品得到最优解
            f[i][j]==f[i-1][j-v]+w表示选第i个物品得到最优解
            这两种情况可以同时实现，递推从f[n][v]开始
        2.字典序最小的方案：题目将物品从1到N进行编号，此时从N到1进行01背包。
            f[i][j]==f[i-1][j]表示不选第i个物品的到最优解
            f[i][j]==f[i-1][j-v]+w表示选第i个物品能得到最优解
            而第i个物品只有选和不选两种选项，仅判断f[i][j]==f[i-1][j-v]+w即可，此时可以保证字典序最小
            贪心策略：尽量选物品编号最小的，于是从1到N开始选（N到1递推），同时在选和不选都成立时，选择。
            cin>>N>>V;
                for(int i=1;i<=N;++i)
                    cin>>v[i]>>w[i];
                for(int i=N;i>=1;--i){
                    for(int j=0;j<=V;++j) {
                        //！！！！所有点都需要进行转移，不能用j=v[i]优化，否则一些状态不会保存
                        f[i][j] = f[i + 1][j];
                        if (j >=v[i])f[i][j] = max(f[i + 1][j], f[i + 1][j - v[i]] + w[i]);
                    }
                }
                int pos=V,step=1;
                while(step<=N){
                    //pos>=v[step]，否则会错
                    if(pos>=v[step]&&f[step][pos]==f[step+1][pos-v[step]]+w[step]){
                        cout<<step<<" ";
                        pos -=v[step];
                    }
                    step++;
                }
    17.能量石：贪心+dp，贪心策略同国王游戏和耍杂技的牛。经过贪心策略分析之后进行01背包。
    18.金明的预算方案：给金明N元购买物品，物品分为主件和附件，附件需要购买了主件之后才能购买。其中每个物品有其价值（价格与重要度的乘积），问能够买得到最大价值。
        最初想到的思路是视作有依赖的背包问题，同14题，此时将根节点设置为0即可，但时间复杂度过大。
        由于此题主件最多拥有两个附件，由此可以将主附件的组合作为一个整体，使用分组背包进行实现。此时只需要解决两个问题：对主附件进行组合后作为一个组 以及 分组背包算法。
 * */
```
### 状态机模型
```c++
/*
    状态机模型就是当前状态由之前得到的额状态转移得到。
    1.大盗阿福：给定一条街，不能够偷相邻的商店，问能够偷到的最大价值
        线性DP方法：考虑状态f[i]为偷前i个商店能够获得的最大值，对于f[i],f[i-1],f[i-2],f[i-3],f[i-4],可知，f[i]不能从f[i-1]得到,可以从f[i-2]，f[i-3]，f[i-4]，f[i-k]得到，但是f[i-2]也能从f[i-4]，f[i-k]得到，因此f[i]从f[i-2]转移要比从f[i-4]转移更优.以此类推,f[i]从f[i-2],f[i-3]转移得到的记过最优。
        状态转移：f[i]=max(f[i-2],f[i-3])+a[i];
        初始条件：f[1]=a[1],f[2]=a[2]
        ans = max(f[N],f[N-1]);也可令a[N+1]=a[N+2]=0,从而ans = f[N+2];
        状态机模型：对于一个店铺，拥有两种状态，偷1或者不偷0，存在如下状态转移关系，不偷到不偷，不偷到偷，偷到不偷。于是f[i][0]=max(f[i-1][0],f[i-1][1]),f[i][1]=f[i-1][0]+a[i];
        最终答案max(f[n][0],f[n][1]);
    2.股票买卖Ⅳ：给定股票价格序列，问最多进行K次买卖能获得的最大价值。
            状态定义
                f[i][j][0]表示进行了j次交易，此时手中不持有股票
                f[i][j][1]表示正在进行第j次交易，此时手中持有股票
            状态转移
                f[i][j][0]=max(f[i-1][j][0],f[i-1][j][1]+w[i]);
                f[i][j][1]=max(f[i-1][j][1],f[i-1][j-1][0]-w[i]);
            初始化时，进行第0次交易拥有0只股票价值为0，其他初始化为-INF
        代码如下：
        int w[SIZE],f[SIZE][110][2],n,k;
        cin>>n>>k;
        for(int i=1;i<=n;++i)cin>>w[i];
        memset(f,0xcf,sizeof f);
        for(int i=0;i<=n;++i)f[i][0][0]=0;
        for(int i=1;i<=n;++i)
            for(int j=1;j<=k;++j){
            f[i][j][0]=max(f[i-1][j][0],f[i-1][j][1]+w[i]);
            f[i][j][1]=max(f[i-1][j][1],f[i-1][j-1][0]-w[i]);
            }
        int ans =0;
        for(int i=0;i<=k;++i)ans =max(ans,f[n][i][0]);
        cout<<ans;
    3.股票买卖Ⅴ：给定股票价格序列，卖股票后需要间隔一天才能买股票，问能获得的最大价值
        状态定义
            f[i][0]表示此时手中不持有股票的价值
            f[i][1]表示此时手中持有股票的价值
        状态转移：
            f[i][0]=max(f[i-1][0],f[i-1][1]+w[i]);
            f[i][1]=max(f[i-1][1],f[i-2][0]-w[i]);//由前一天转移得到
        初始化：f[1][0]=0,f[1][1]=-w[i];
        核心代码：
            f[1][0]=0,f[1][1]=-w[1];
            for(int i=2;i<=n;++i){
                f[i][0]=max(f[i-1][0],f[i-1][1]+w[i]);
                f[i][1]=f[i-1][1];
                if(i>=2)f[i][1]=max(f[i][1],f[i-2][0]-w[i]);
            }
    4.密码设计
        你现在需要设计一个密码 S，S 需要满足：
            S 的长度是 N；
            S 只包含小写英文字母；
            S 不包含子串 T；
            例如：abc 和 abcde 是 abcde 的子串，abd 不是 abcde 的子串。
            请问共有多少种不同的密码满足要求？
            由于答案会非常大，请输出答案模 109+7 的余数。
        该题时一个由kmp的next数组组成的状态机。状态之间进行这样的转移
            设计状态f[i][j]表示前i个字符，匹配j个字符的字符串数量。f[i][j]在其后面增加一个字符得到f[i+1][k]，其中k为s[i~j],c的匹配长度，该长度可以用kmp得到。从f[i][...]到f[i+1][...]转移，最终结果为sum(f[n][...])
        cin>>n;
            cin>>s+1;
            int len = strlen(s+1);
            ne[1]=0;
            for(int i=2,j=0;i<=len;++i){//处理next数组
                while(j&&s[i]!=s[j+1])j=ne[j];
                if(s[i]==s[j+1])j++;
                ne[i]=j;
            }
            f[0][0]=1;//初始化时f[0][0]=1表示长度为0匹配长度为0的种类为1，f[0][k,k>1]表示种类为0，不存在
            for(int i=0;i<n;++i)//从i推i+1，最终只需要计算到n-1(n-1推n)
                for(int j=0;j<len;++j)//遍历可能的长度（[0~len]）
                    for(char c='a';c<='z';++c){
                        int k =j;
                        while(k&&c!=s[k+1])k=ne[k];//匹配
                        if(c==s[k+1])k++;//至多匹配len-1个字符
                        if(k<len)f[i+1][k]=(f[i+1][k]+f[i][j])%mod;//更新f[i+1][...];
                    }
            int ans = 0;
            for(int i=0;i<len;++i)ans = (ans +f[n][i])%mod;//
            cout<<ans;
 * 
```

状态压缩DP
```c++
/*
    状态压缩DP就是通过使用位的0和1来表示状态，通过位运算减少计算量，通过位存储来节省存储空间的一种DP。状态压缩DP的特点：每个点仅有两种状态，通常以行或列的形式考虑。
    1.小国王：给定N×N的棋盘和K个国王，每个国王相邻（8个）位置不能有另一个国王，问一共有多少种存储方式。
        对于棋盘上的每个位置，仅有两种状态：有国王和无国王，将有国王的状态设置成1。同时，对于每一列的国王使用二进制数进行存储。一个列的国王应该满足不相邻，即二进制数的两个位不能相邻。同时对于相邻的列（分别用a和b表示其存储状态），应满足a&b==0，即相同位不能同时为1，同时，a|b应该为合法状态（国王不能相邻）。
        状态表示：f[i][j][k]表示前i列，第i列状态为j，当前有k个国王的种数。
        从i-1列到第i列，状态b转移到状态a：
            if(valid[a]&&valid[b]&&valid[a|b]&&(a&b)==0&&k>=king[b])f[i][a][k]+=f[i-1][b][k-king[b]];
        初始化f[0][0][0]=1,f[..][..][..]=0;表示第0列不放国王共0个国王的状态为一种合法状态，第0列的其他状态均不合法（其他状态都由此状态转移）
        #include "iostream"
        using namespace std;
        const int SIZE = (1<<11)+10;
        long long f[15][SIZE][110];
        int king[SIZE];
        bool valid[SIZE];
        
        int main(){
            int n,k;
            cin>>n>>k;
            f[0][0][0]=1;
            int size = 1<<n;
            for(int i=0;i<size;++i){//
                int cnt = 0;//国王个数
                int pre = -2,cur = 0,val = i;
                bool invalid = false;
                while(val){
                    if(val&1){
                        cnt++;
                        if(cur-pre<2)invalid = true;
                        pre = cur;
                    }
                    cur++;
                    val>>=1;
                }
                if(invalid)valid[i]=false;
                else valid[i]=true,king[i]=cnt;
            }
            for(int i=1;i<=n+1;++i){//每一列
                for(int a=0;a<size;++a){//当前列状态为a
                    if(valid[a])
                        for(int b=0;b<size;++b){//前一列状态为b
                            if(valid[b]&&(a&b)==0&&valid[a|b]){//状态合法
                                for(int num=0;num<=k;++num){//对于每个国王数
                                    if(num>=king[a])f[i][a][num]+=f[i-1][b][num-king[a]];
                                }
                            }
                        }
                }
            }
            cout<<f[n+1][0][k];
        }
    2.玉米田：给定N×N的玉米田，玉米田中有一些地方不能种，同时所种玉米不能相邻，问有多少种种植方式。
        玉米田只有种和不种两种状态。用状态dp实现。f[i][j]表示前i列且第i列的状态为j的方案数。
        对于一列，不能存在相邻的1，预处理合法的状态；对于相邻的两列（状态分别为a和b），不能存在相邻的1，满足(a&b)==0;同时，玉米只能种在肥沃的土地上a|base[i]==base[i]；初始化f[0][0]=1,f[...][...]=0表示0列不种是一种合法状态；ans=f[N+1][0]，第N+1列不种东西的合法种数即为答案。
        #include "iostream"
        using namespace std;
        int f[15][(1<<13)+10];
        bool g[14][14];
        int base[15];//存储每一列的值
        bool valid[(1<<13)+10];
        int mod = 1e8;
        
        int main(){
            int N,M;
            cin>>M>>N;
            for(int i=1;i<=M;++i)
                for(int j=1;j<=N;++j)
                    cin>>g[i][j];
            for(int i=1;i<=N;++i)
                for(int j=1;j<=M;++j){
                    base[i]+=g[j][i]<<(j-1);
                }//ok
            int size = 1<<M;
            for(int i=0;i<size;++i){//保存每列是否为合法状态
                int pre=-2,cur=0,val = i;
                bool invalid = false;
                while(val){
                    if(val&1==1){
                        if(cur-pre<2)invalid = true;
                        pre = cur;
                    }
                    cur++;
                    val>>=1;
                }
                if(invalid)valid[i]=false;
                else valid[i]=true;
            }
            //for(int i=0;i<size;++i)cout<<valid[i]<<" ";//ok
            f[0][0]=1;
            for(int i=1;i<=N+1;++i){//每一列
                for(int a=0;a<size;++a){//当前列的状态
                    if(valid[a]&&(a|base[i])==base[i])//a为合法状态且与base不冲突
                        for(int b=0;b<size;++b)
                            if(valid[b]&&(a&b)==0)
                                f[i][a]=(f[i][a]+f[i-1][b])%mod;
                    
                }
            }
            cout<<f[N+1][0];
        }
    3.炮兵阵地：给定一块地，存在平原和山丘，炮兵只能部署在平原上，同时炮兵可以打到上下两格，问在不误伤的情况下最多能摆放多少炮兵。
        思路：1表示摆放炮兵，0表示不摆放。由于炮兵能影响两列，因此状态表示为f[i][j][k]，j表示i-1列的状态，k表示第i列的状态
        同时需要满足以下条件：
            1.同一列炮兵距离>2
            2.相邻两列炮兵不能相互攻击f[i][a][b]和f[i+1][b][c]应满足(a&b)==0,(a&c)==0,(b&c)==0
            3.炮兵只能摆放在平原上，应满足(a|)base[i]==base[i]
        初始化f[...][...][...]=0，起始列从2开始，ans=f[n+3][0][0]
        同时空间有限，需要使用滚动数组优化，由于递推仅涉及相邻两个状态，采用取模优化
        代码如下：
        #include <ctime>
        #include "iostream"
        #include "cstring"
        using namespace std;
        const int SIZE = (1<<10)+10;
        int f[4][SIZE][SIZE];
        int g[4][SIZE][SIZE];
        int base[110];
        bool valid[SIZE];
        int count[SIZE];
        
        int main(){
            int n,m,size;
            cin>>n>>m;
            size  = 1<<m;
            for(int i=2;i<n+2;++i)
                for(int j=0;j<m;++j){
                    char c;
                    cin>>c;
                    if(c=='P')base[i]+=(1<<(m-j-1));
                }
            for(int i=0;i<size;++i){
                int cur=0,pre=-4,val =i;
                bool invalid = false;
                int cnt=0;
                while(val){
                    if((val&1)==1){//当前位为1
                        if(cur-pre<=2)invalid = true;
                        pre = cur;
                        cnt++;
                    }
                    cur++;
                    val>>=1;
                }
                if(!invalid)valid[i]=true,count[i]=cnt;
            }
            for(int i=2;i<=n+3;++i){
                memset(&f[i%2][0][0],0,sizeof(int)*SIZE*SIZE);
                int x = i%2,y=(i-1)%2;//滚动数组，空间优化
                for(int a=0;a<size;++a)
                    if(valid[a]&&(a|base[i-2])==base[i-2])
                        for(int b=0;b<size;++b)
                            if(valid[b]&&(a&b)==0&&(b|base[i-1])==base[i-1])
                                for(int c=0;c<size;++c)
                                    if(valid[c]&&(a&c)==0&&(b&c)==0&&(c|base[i])==base[i])
                                        f[x][b][c]=max(f[x][b][c],f[y][a][b]+count[c]);
            }
            cout<<f[(n+3)%2][0][0]<<endl;
            return 0;
        }
 * */

```

区间DP
```c++
/*
    区间DP经典题目：石子合并，环形石子合并。
    1.环形石子合并：给定一堆石子环形排列，每次合并两堆石子需要消耗石子体积的能量，问将石子合并成一堆的最大消耗和最小消耗
        思路：合并时考虑合并的每一种可能性，取其最小值或最大值。环形合并可以视为长度为2N的石子中，N堆石子进行合并的结果的最值。
        #include "iostream"
        #include "cstring"
        using namespace std;
        const int SIZE = 410;
        int fl[SIZE][SIZE],fs[SIZE][SIZE],s[SIZE];
        //fl存最大，fs存最小
        int main(){
            int n;
            cin>>n;
            memset(fs,0x3f,sizeof(fs));//初始化为inf
            for(int i=0;i<=2*n;++i)fs[i][i]=0;//初始化
            for(int i=1;i<=n;++i)cin>>s[i],s[i+n]=s[i];
            for(int i=1;i<=2*n;++i)s[i]+=s[i-1];
            //f[i][i+k]=max or min(f[i][i+j]+f[i+j+1][i+k]+s[i+k]-s[i-1])
            for(int k=1;k<=n;++k)//区间长度
                for(int i=1;i+k<=2*n;++i)//左端点
                    for(int j=0;j<k;++j)//左区间长度
                        fl[i][i+k]=max(fl[i][i+k],fl[i][i+j]+fl[i+j+1][i+k]+s[i+k]-s[i-1]),
                        fs[i][i+k]=min(fs[i][i+k],fs[i][i+j]+fs[i+j+1][i+k]+s[i+k]-s[i-1]);
            int ans_max = 0,ans_min=0x3f3f3f3f;
            for(int i=1;i<=n;++i)
                ans_min = min(ans_min,fs[i][i+n-1]),ans_max = max(ans_max,fl[i][i+n-1]);
            cout<<ans_min<<endl<<ans_max<<endl;
        }
    2.能量项链：合并相邻两个元素会获得一定能量，问能够获得能量的最大值。
        #include "iostream"
        #include "algorithm"
        using namespace std;
        const int SIZE = 210;
        typedef pair<int,int> PII;
        PII a[SIZE];
        int f[SIZE][SIZE];
        
        
        int main(){
            int n;
            cin>>n;
            for(int i=1;i<=n;++i){
                cin>>a[i].first;
                a[i+n-1].second = a[1+(i-2+2*n)%(2*n)].second = a[i+n].first=a[i].first;
            }
            for(int k=1;k<n;++k)//f[i][i+k]=max(f[i][i+k],f[i][i+j]+f[i+j+1][i+k]+a[i].first*a[i+j+1].first*a[i+k].second)
                for(int i=1;i+k<=2*n;++i)
                    for(int j=0;j<k;++j)
                        f[i][i+k]=max(f[i][i+k],f[i][i+j]+f[i+j+1][i+k]+a[i].first*a[i+j+1].first*a[i+k].second);
            int ans =0;
            for(int i=1;i<=n;++i)
                ans = max(ans,f[i][i+n-1]);
            cout<<ans;
        }
    3.加分二叉树，给定一个二叉树的中序遍历，问该二叉树什么情况下分值最大。分值的计算方式：叶节点分数为自身权值，非叶节点分数为左子树分数*右子树分数+当前点权值。输出字典序最小的前序遍历。
        最大权值计算方式，采用区间DP，从区间长度从小到大进行计算，得到最终结果。
        最小前序遍历的计算方式：从前往后遍历，计算第一个最大值。使用pre[i][j]存储子树根节点。dfs输出前序遍历
        #include "iostream"
        using namespace std;
        const int SIZE = 40;
        int f[SIZE][SIZE],a[SIZE],pre[SIZE][SIZE];
        void dfs(int l,int r){
            if(l<=r){
                cout<<pre[l][r]<<" ";
                dfs(l,pre[l][r]-1);
                dfs(pre[l][r]+1,r);
            }
        }
        int main(){
            int n;
            cin>>n;
            for(int i=1;i<=n;++i)
                cin>>a[i];
            for(int i=1;i<=n;++i)f[i][i]=a[i],pre[i][i]=i;
            for(int k=1;k<n;++k)//i+k为右端点
                for(int i=1;i+k<=n;++i)
                    for(int j=0;j<=k;++j){
                        int l,r;
                        if(i+j-1<i)l=1;
                        else l = f[i][i+j-1];
                        if(i+j+1>i+k)r=1;
                        else r = f[i+j+1][i+k];
                        if(f[i][i+k]<l*r+a[i+j])f[i][i+k]=l*r+a[i+j],pre[i][i+k]=i+j;
                    }
            cout<<f[1][n]<<endl;
            dfs(1,n);
        }
    4.凸多边形划分：对于一个凸多边形，将其划分成不相交的三角形，每个三角形的权值为三个顶点的乘积。问划分的最小权值是多少？
        由于数据很大，需要用到高精度，可以自己实现高精度，也可以使用__int128
        #include "iostream"
        #include "cstring"
        using namespace std;
        const int SIZE = 55;
        __int128 f[SIZE][SIZE];
        __int128 a[SIZE];
        inline __int128 read(){
            __int128 x = 0, f = 1;
            char ch = getchar();
            while(ch < '0' || ch > '9'){
                if(ch == '-')
                    f = -1;
                ch = getchar();
            }
            while(ch >= '0' && ch <= '9'){
                x = x * 10 + ch - '0';
                ch = getchar();
            }
            return x * f;
        }
        inline void print(__int128 x){
            if(x < 0){
                putchar('-');
                x = -x;
            }
            if(x > 9)
                print(x / 10);
            putchar(x % 10 + '0');
        }
        const __int128 inf = 1e27;
        int main(){
            int n;
            cin>>n;
            for(int i=0;i<SIZE;++i)
                for(int j=0;j<SIZE;++j)f[i][j]=inf;
            for(int i=1;i<=n;++i)a[i]=read(),f[i][i+1]=0;
            for(int k=2;k<n;++k)
                for(int i=1;i+k<=n;++i)
                    for(int j=1;j<k;++j)
                        f[i][i+k]=min(f[i][i+k],f[i][i+j]+f[i+j][i+k]+a[i]*a[i+j]*a[i+k]);
            print(f[1][n]);
        }
 * */
```

树形DP
```c++
/*
    树形DP：树形DP就是在树上进行的动态规划题，比如求树的重心，中心，最长路径等，通常需要通过dfs实现，通过对树与子树之间的关系进行状态转移。
    1.树的最长路径：给定一棵树，求其中两点的最长路径。
        思路：对于一棵树的最长路径，总是会经过一颗子树的根节点，且经过其两个不同的子树（如果存在两颗子树的话），因此对于树的每颗子树，维护一个经过当前根节点的最长路径，并维护一个当前根节点到叶子节点的最大值，用于父节点求其最长路径。答案即为所有子树的路径最大值。
        #include "iostream"
        #include "queue"
        using  namespace std;
        const int SIZE = 20010;
        int h[SIZE],idx;
        int f[SIZE];
        bool v[SIZE];
        
        struct Edge{
            int next;
            int ver;
            int dis;
        }edge[SIZE];
        
        void add(int a,int b,int dis){
            edge[++idx].ver  =b;
            edge[idx].next = h[a];
            h[a]=idx;
            edge[idx].dis = dis;
        }
        
        int dfs(int cur){//返回当前子树根节点到叶子节点的最长路径
            v[cur]=1;
            priority_queue<int> son_dis;
            int ne = h[cur];
            int pos_dis=0;
            int max_son_dis = 0;
            while(ne){
                int pos = edge[ne].ver;
                if(!v[pos]){
                    int cur_dis = dfs(pos)+edge[ne].dis;//当前根节点经过子树的最长路径
                    max_son_dis = max(max_son_dis,cur_dis);
                    son_dis.push(cur_dis);
                }
                ne = edge[ne].next;
            }
            int max_cur = 0;
            if(son_dis.size()){
                max_cur+=son_dis.top();
                son_dis.pop();
            }
            if(son_dis.size()){
                max_cur+=son_dis.top();
                son_dis.pop();
            }//选择路径最长的两个子树，其和为当前子树的最长路径
            f[cur]=max(f[cur],max_cur);
            return max_son_dis;
        }
        
        int main(){
            int n;
            cin>>n;
            for(int i=1;i<=n;++i){
                int a,b,dis;
                cin>>a>>b>>dis;
                add(a,b,dis);
                add(b,a,dis);
            }
            dfs(1);
            int ans = 0;
            for(int i=1;i<=n;++i)ans = max(ans,f[i]);
            cout<<ans;
        }
    2.树的中心：当前点到其他点路径长度的最大值最小，该点就是树的中心。
        思路：对于一个点，计算它到其他点最长距离的最小值分为两个方面，父节点部分和儿子节点部分，儿子节点部分通过一个dfs即可得到。父节点的最大值同样有两个来源，来自当前点兄弟节点，或者来自祖父节点，因此每个点需要维护一个子树最长路径和次长路径，用于计算经过父节点路径的最大值。
        通过两次dfs实现：第一次dfs找到每个子树的最大值和次大值；第二次dfs计算到其他点距离最大值的最小值（深搜时向子节点传入父节点部分的最大值）
        #include "iostream"
        #include "cstring"
        using namespace std;
        const int SIZE = 20010;
        int h[SIZE],idx,max_son[SIZE],second_son[SIZE],f[SIZE];
        bool v[SIZE];
        struct Edge{
            int next;
            int ver;
            int dis;
        }edge[SIZE];
        
        void dfs_1(int cur){//求max_son和second_son
            v[cur]=1;
            int ne = h[cur];
            
            while(ne){
                int pos = edge[ne].ver;
                if(!v[pos]){
                    dfs_1(pos);
                    int cur_path = max_son[pos]+edge[ne].dis;//当前路径长度
                    if(cur_path>=second_son[cur]){//维护最大值和次大值
                        if(cur_path>=max_son[cur])
                            second_son[cur]=max_son[cur],max_son[cur]=cur_path;
                        else
                            second_son[cur]=cur_path;
                    }
                }
                ne = edge[ne].next;
            }
        }
        
        void dfs_2(int cur,int f_max){//最终ans 求所有max的最大值
            v[cur]=1;
            int ne = h[cur];
            f[cur]=f_max;
            while(ne){
                int pos = edge[ne].ver;
                if(!v[pos]){
                    if(max_son[cur]==max_son[pos]+edge[ne].dis){//当前为最长路径，传入次大值
                        dfs_2(pos,edge[ne].dis+max(f_max,max(f_max,second_son[cur])));//传给儿子节点最大值
                    }else//当前不是最长路径，传入最大值
                        dfs_2(pos,max(f_max,edge[ne].dis+max(f_max,max_son[cur])));
                    f[cur]=max(f[cur],max_son[pos]+edge[ne].dis);
                }
                ne=edge[ne].next;
            }
        }
        
        void add(int a,int b,int c){
            edge[++idx].ver = b;
            edge[idx].next  = h[a];
            edge[idx].dis = c;
            h[a]=idx;
        }
        
        int main(){
            int n;
            cin>>n;
            for(int i=1;i<n;++i){
                int a,b,c;
                cin>>a>>b>>c;
                add(a,b,c);
                add(b,a,c);
            }
            memset(v,0,sizeof(v));
            dfs_1(1);
            memset(v,0,sizeof(v));
            dfs_2(1,0);
            int ans = 0x3f3f3f3f;
            for(int i=1;i<=n;++i)ans = min(ans,f[i]);
            cout<<ans;
        }
 * */
```