* quick_sort
```C++
void quick_sort(int *a,int l,int r){
	int i=l,j=r,mid=a[(l+r)>>1];//i j mid三标志
	do{//do while 循环 i<j
		while(a[i]<mid)i++; //while < i++
		while(a[j]>mid)j--; //while > j--
		if(i<=j)swap(a[i++],a[j--]); //<= swap 嵌套i++ j--
	}while(i<j);
	if(l<j)quick_sort(a,l,j);//l j i r 再递归
	if(i<r)quick_sort(a,i,r);
}
```

* merge_sort
```C++
int b[BUF_SIZE];
void merge_sort(int *a,int l, int r){
	if(l==r)return;
	int mid=  (l+r)>>1;
	merge_sort(a,l,mid);
	merge_sort(a,mid+1,r);
	int i=l,j=mid+1,k=l;
	while(i<=mid&&j<=r){
		if(a[i]<a[j]){
			b[k++]=a[i++];
		}else{
			b[k++]=a[j++];
		}
	}
	while(i<=mid)b[k++]=a[i++];
	while(j<=r)b[k++]=a[j++];
	for(k=l;k<=r;++k)a[k]=b[k];
}
```
binary_search_1
```c++
bool check(int){...};
//check将查找范围分成两个区间，while循环查找右区间的左端点

while(l<r){
	mid = l+r>>1;
	if(check(mid))r=mid;
	else l = mid+1;
}
```
binary_serach_2
```c++
bool check(int){...};
//check将查找范围分成两个区间，while循环查找左区间的右端点
while(l<r){
	mid = l+r+1>>1;
	if(check(mid))l=mid;
	else r = mid-1;
}
```
float_binary_search
```c++
bool check(double){...}
//check将查找范围分成两个区间 如> while查找右区间的左端点

double l,r,mid,eps=1e-7;
while(r-l>eps){
	mid = (l+r)/2;
	if(check(mid))r=mid;
	else l =mid;
}
```
高精度加法 正整数+正整数
```c++
vector<int> add(vector<int>&A,vector<int>&B){
	if(A.size()<B.size())return add(B,A);
	vector<int> C;
	int t=0;
	for(int i=0;i<A.size();++i){
		t+=A[i];
		if(i<B.size())t+=B[i];
		C.push_back(t%10);
		t/=10;
	}
	if(t)C.push_back(t);
	return C;
}

//输入预处理方法 倒序存储
char s[BUF_SIZE];
vector<int> A;
cin>>s;
int len = strlen(s);
for(int i =len-1;i>=0;--i)
	A.push_back(s[i]-'0');
```
高精度减法
```c++
//A>=B>=0
vector<int> sub(vector<int>&A,vector<int>&B){
	vector<int>C;
	for(int t=0,i=0;i<A.size();++i){
		t = A[i]-t;
		if(i<B.size())t -= B[i];
		C.push_back((t+10)%10);
		if(t<0)t=1;
		else t = 0;
	}
	while(C.size()>1&&C.back()==0)C.pop_back();
	return C;
}
//使用前对A和B进行符号和顺序处理 以得到正确结果
```
高精度乘法
```c++
//满足A>=0,b>=0
vector<int> mul(vector<int> &A,int b){
	vector<int> C;
	for(int i=0,t=0;i<A.size()||t;++i){
		if(i<A.size())t+=A[i]*b;
		C.push_back(t%10);
		t/=10;
	}
	while(C.size()>1&&C.back()==0)C.pop_back();
	return C;
}
```

高精度除法
```c++
//保证A>=0 b>0
#include "algorithm"
vector<int> div(vector<int>&A,int b,int &r){
	vector<int> C;
	r = 0;
	for(int i=A.size()-1;i>=0;i--){
		r=r*10+A[i];
		C.push_back(r/b);
		r%=b;
	}
	reverse(C.begin(),C.end());
	wihle(C.size()>1&&C.back()==0)C.pop_back();
	return C;
}
```
前缀和：用于无修改的区间查询
```c++
//前缀和均从下标1开始
int a[SIZE],s[SIZE];
for(int i=1;i<l=en;++i)
	cin>>a[i],s[i]=s[i-1]+a[i];
//l-r的和
for(int i=0;i<m;++i)
	cin>>l>>r,cout<<s[r]-s[l-1];
```
二维前缀和：用于无修改的区间查询
```c++
//n乘m的矩阵 求出(x1,y1),(x2,y2)之间矩阵和
int n,m,q,val,x1,y1,x2,y2,s[SIZE][SIZE];
cin>>n>>m>>q;
for(int i=1;i<=n;++i)
	for(int j=1;j<=m;++j)
		cin>>val,s[i][j]=s[i-1][j]+s[i][j-1]+val-s[i-1][j-1];
for(int i=0;i<q;++i)
	cin>>x1>>y1>>x2>>y2,cout<<s[x2][y2]-s[x1-1][y2]-s[x2][y1-1]+s[x1-1][y1-1]<<endl;
```
一维差分：区间修改
```c++
//进行区间修改时（区间同时增加和同时减少一个值）
//在差分情况下，只有左右端点会发生影响，从O(len)降低为O(2)，d[l+=val],d[r+1]-=val
int d[SIZE];
int n,m,val,l,r,pre=0;
cin>>n>>m;
for(int i=1;i<=n;++i)
cin>>val,d[i]=val-pre,pre=val;
for(int i=0;i<m;++i)
cin>>l>>r>>val,d[l]+=val,d[r+1]-=val;
for(int i=1;i<=n;++i)
d[i]+=d[i-1],cout<<d[i]<<" ";

//ticks
//零数组的差分也是零数组，所以差分数组的初始化直接用零数组进行区间修改即可
//二维差分思想相同
cin>>n>>m;
for(int i=1;i<=n;++i)
	cin>>val,d[i]+=val,d[i+1]-=val;
for(int i=0;i<m;++i)
	cin>>l>>r>>val,d[l]+=val,d[r+1]-=val;
for(int i=1;i<=n;++i)
	d[i]+=d[i-1],cout<<d[i]<<" ";
```
二维差分：区间修改
```c++
int d[SIZE][SIZE];
int n,m,q,val,x1,x2,y1,y2;
cin>>n>>m>>q;
for(int i=1;i<=n;++i)
for(int j=1;j<=m;++j)
    cin>>val,d[i][j]+=val,d[i+1][j]-=val,d[i][j+1]-=val,d[i+1][j+1]+=val;

for(int i=0;i<q;++i)
cin>>x1>>y1>>x2>>y2>>val,d[x1][y1]+=val,d[x2+1][y1]-=val,d[x1][y2+1]-=val,d[x2+1][y2+1]+=val;
for(int i=1;i<=n;++i){
for(int j=1;j<=m;++j)
    d[i][j]+=d[i-1][j]+d[i][j-1]-d[i-1][j-1],cout<<d[i][j]<<" ";
cout<<endl;
}
```
得到a的第k位的值
```c++
bool bit = a>>k&1;
//获取n中二进制1的个数
int count = 0;
for(int i=0;i<32;++i)
	if(n>>i&1)count++;
cout<<count;
```
lowbit:n的最后一位1，常用于树状数组
```c++
//lowbit(3)=1
//lowbit(2)=2;
//lowbit(4)=4;
//lowbit(8)=8;
inline int lobit(int n ){return n&-n}
//获取n二进制中1的个数,lowbit方法效率比位右移效率高
int count =0;
while(n){
	int lowbit = n&-n;
	n-=lowbit;
	count++;
}
cout<<count;
```
双指针 [最长连续不重复子序列](https://www.acwing.com/problem/content/801/) [数组元素的目标和](https://www.acwing.com/problem/content/802/)
```c++
//顾名思义，双指针意为使用两个指针进行索引，维护一种性质
//常见问题分类：
//(1) 对于一个序列，用两个指针维护一段区间
//(2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

//最长连续不重复子序列
//基本思路:使用双指针维护一个不重复区间，当指针i移动时判断是否有重复元素（使用数组标记出现过的元素）
//出现重复元素时，记录此时不重复区间长度，并移动j使得区间不重复
//取所有不重复区间的长度的最大值为答案
#include "iostream"
using namespace std;
const int SIZE =100010;
int a[SIZE];
bool v[SIZE];

int main(){
    int n;
    cin>>n;
    for(int i=0;i<n;++i)
        cin>>a[i];
    int max_len =0;
    int i=0,j=0;
    for(;i<n;++i){
        if(v[a[i]]){//a[i]重复 j右移
            max_len = (i-j)>max_len?(i-j):max_len;
            while(a[j]!=a[i])v[a[j++]]=0;
            j++;
            //cout<<j<<" "<<i<<endl;
        }else v[a[i]]=1;
    }
    if(a[i]!=a[j])max_len = (i-j)>max_len?(i-j):max_len;//特判i走到n仍未发生冲突的情况
        cout<<max_len;
}

//数组元素的目标和
//维护两个数的和，i从前往后扫描A序列，j从后往前扫描序列B
//i往后扫描时和变大，j往前扫描时，和变小
//于是：当A[i]+B[j]>val j--; A[i]+B[i]<val i++;A[i]+B[j]==val 跳出循环
#include "iostream"
    using namespace std;
const int SIZE =100010;
int a[SIZE],b[SIZE];

int main(){
    int n,m,val;
    cin>>n>>m>>val;
    for(int i=0;i<n;++i)
        cin>>a[i];
    for(int j=0;j<m;++j)
        cin>>b[j];
    int i=0,j=m-1;
    while(a[i]+b[j]!=val){
        if(a[i]+b[j]>val)j--;
        if(a[i]+b[j]<val)i++;
    }
    cout<<i<<" "<<j;
}
```
（整数）离散化 [区间和](https://www.acwing.com/problem/content/804/)
```c++
//离散化就是将一个很大的区间映射到一个小的区间
//整数离散化的所做法就是排序去重，二分查找得到下标
//
vector<int> alls; // 存储所有待离散化的值
sort(alls.begin(), alls.end()); // 将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end());   // 去掉重复元素
// 二分求出x对应的离散化的值
int find(int x) // 找到第一个大于等于x的位置
{
    int l = 0, r = alls.size() - 1;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1; // 映射到1, 2, ...n
}
//区间和
//采用离散化的方式将值映射到一个较小的区域
//由于可能存在对同一点进行多次修改的情况 采用pair记录操作 并手动去重
//查询时对边界情况进行特判 如查询区间在左端点的左侧或右端点的右侧
#include "iostream"
#include "algorithm"
#include "map"
using namespace std;
const int SIZE = 100010;
pair<int,int> a[SIZE];

struct cmp{
    bool operator()(pair<int,int>A,pair<int,int>B){
        return A.first<=B.first;
    }
};

int main(){
    int n,m,pos,val;
    cin>>n>>m;
    for(int i=1;i<=n;++i)
        cin>>a[i].first>>a[i].second;
    sort(a+1,a+n+1,cmp());
    int i,j;
    for(i=1,j=1;i<=n;++i,++j){
        a[j]=a[i];
        while(i<n&&a[i+1].first==a[j].first)a[j].second+=a[++i].second;
    }
    n = j-1;
    for(int i=1;i<=n;++i)
        a[i].second+=a[i-1].second;
    //for(int i=1;i<=n;++i)
	//	cout<<i<<" "<<a[i].first<<" "<<a[i].second<<endl; 
    int x,y;
    for(int i=0;i<m;++i){
        int l=1,r=n;
        cin>>x>>y;
        while(l<r){
            int mid = l+r>>1;
            if(a[mid].first>=x)r=mid;
            else l=mid+1;
        }
        int posx = l;
        l=1,r=n;
        while(l<r){
            int mid = l+r+1>>1;
            if(a[mid].first<=y)l = mid;
            else r = mid-1;
        }
        int posy = l;
        //cout<<"posx:"<<posx<<" posy:"<<posy<<endl;
        if(y<a[1].first||x>a[n].first)cout<<0<<endl;//左端点的左侧，右端点的右侧
        else if(posx<=posy)cout<<a[posy].second - a[posx-1].second<<endl;//有交集
        else cout<<0<<endl;//无交集
    }
}
```
[区间合并](https://www.acwing.com/problem/content/805/)
```C++
#include "iostream"
#include "map"
#include "algorithm"
using namespace std;

struct cmp{
    bool operator()(pair<int,int> A,pair<int,int>B){
        return A.first<B.first;
    }
};

int main(){
    int n;
    cin>>n;
    vector<pair<int,int> >segs;
    int l,r;
    for(int i=0;i<n;++i)
        cin>>l>>r,segs.push_back({l,r});
    sort(segs.begin(),segs.end(),cmp());
    vector<pair<int,int> >ans;
    int s=segs.begin()->first,e=segs.begin()->second;
    for(auto seg = segs.begin()+1;seg<segs.end();++seg){
        if(seg->first<=e)e = max(seg->second,e);
        else ans.push_back({s,e}),s = seg->first,e = seg->second;
    }
    ans.push_back({s,e});
    cout<<ans.size();
}
```
[单调栈](https://github.com/hehelv/brush_algorithm.git)
```c++
//单调栈的使用场景：求点i左侧离i最近的比它小的值
//考虑a[1~i]中a[j],若a[j]是a[i]左侧第一个比a[i]小的值
//此时，a[j+1~i-1]均>=a[i]
//对于a[i+1]，若a[i+1]>a[i],则a[i]为所求
//若a[i+1]<=a[i],则a[1~j]中为a[i]所求
//由此,a[j+1~i-1]这些在a[i]左侧且>=a[i]的元素均被舍弃
//此时a[i+1]保存的元素单调递增
//使用一个栈维护左侧序列，pop()比a[i+1]大的元素，再将a[i+1]放入栈
#include "iostream"
#include "stack"
using namespace std;
const int SIZE = 100010;
int a[SIZE];

int main(){
    int n,val;
    stack<int> stk;
    cin>>n;
    for(int i=0;i<n;++i){
        cin>>val;
        while(stk.size()&&stk.top()>=val)stk.pop();
        //删除掉栈中比val大的值，将val插入栈
        if(stk.size())cout<<stk.top()<<" ";
        else cout<<"-1 ";
        stk.push(val);
    }
}
```
单调队列 [滑动窗口](https://www.acwing.com/problem/content/156/)
```C++
//单调队列的使用场景：滑动长度为K的窗口，求窗口的最值
//考虑维护区间的最小值，对于a[i-k+1~i]中的元素a[j]，若a[i+1]<a[j]
//此时，只要窗口包含a[i+1]，a[j]都不会成为答案
//因此，在往队列中加入新元素时，可以将大于（也可以是大于等于，处理方式稍有差异）新元素的值pop掉
//此时队列保持单调，最小值位于最左侧

//滑动窗口
//处理思路：入队时将比新元素大的值删除，等于新元素的值保存
//出队时，若队首元素等于区间起始值，出队
//|[a1] a2 a3 a4 [a5]| a6 [a7] [a8]
//由于单调队列队首只会保存窗口中的最小元素，当区间窗口等于队首元素时，进行pop
//同时，最小元素可能存在多个，因此入队时仅删除大于它的元素
//另一种处理方式：队列保存元素的下标，入队时删除大于等于(等于可选)的元素
//出队时判断队首时候不在窗口内，不在则出队
//
#include "iostream"
#include "deque"
using namespace std;
const int SIZE = 1000010;
int a[SIZE];

int main(){
    deque<int> deq;
    int n,k,Min,Max;
    cin>>n>>k;
    for(int i=0;i<n;++i){
        cin>>a[i];
    }
    for(int i=0;i<k;++i){
        while(deq.size()&&deq.back()>a[i])deq.pop_back();
        deq.push_back(a[i]);
    }
    cout<<deq.front()<<" ";
    for(int i=k;i<n;++i){
        if(deq.front()==a[i-k])deq.pop_front();
        while(deq.size()&&deq.back()>a[i])deq.pop_back();
        deq.push_back(a[i]);
        cout<<deq.front()<<" ";
    }
    puts("");
    deq.clear();
    for(int i=0;i<k;++i){
        while(deq.size()&&deq.back()<a[i])deq.pop_back();
        deq.push_back(a[i]);
    }
    cout<<deq.front()<<" ";
    for(int i=k;i<n;++i){
        if(deq.front()==a[i-k])deq.pop_front();
        while(deq.size()&&deq.back()<a[i])deq.pop_back();
        deq.push_back(a[i]);
        cout<<deq.front()<<" ";
    }
    puts("");
}
```

KMP算法
```c++
#include "iostream"
using namespace std;
const int SIZE = 1e6+10;
char t[SIZE],s[SIZE];
int ne[SIZE];//使用next时可能会和某些库冲突
//next[i]=j的含义为以位置i结尾的后缀与前缀的最大匹配长度为j
//t[1,j]=t[i-j+1,i];

int main(){
    cin>>t+1>>s+1;
    int lent = strlen(t+1);
    int lens = strlen(s+1);
    //KMP习惯从1开始
    for(int i=2,j=0;i<=lent;++i){//i=0不需要匹配
        while(j&&t[i]!=t[j+1])j=ne[j];//匹配不成功则退一步，p=0时退无可退
        if(t[i]==t[j+1])j++;//匹配成功，j进一步
        //若匹配不成功，此时p为0
        ne[i]=j;
    }
    for(int i=1,j=0;i<=lens;++i){
        while(j&&s[i]!=t[j+1])j=ne[j];
        if(s[i]==t[j+1])j++;
        if(j==lent){//匹配成功
            //进行匹配成功的操作
            j=ne[j];
        }
    }
}
```
