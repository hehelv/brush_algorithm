算法模板基于[acwing](acwing.com)

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

中缀表达式转后缀表达式
```c++
/*
 * 中缀表达式转后缀表达式的实现思路
 * 对于数字：直接输出
 * 对于左括号：入栈
 * 对于右括号：输出到左括号为止
 * 对于从左往右的运算符：输出栈内优先级大于等于它的操作符
 * 对于从右往左的运算符：输出栈内优先级大于它的操作符
 * pri['(']=0 pri['+']=pri['-']=1 pri['*']+pri['/']=2 pri['^']=3
 * 从左往右+ - * / 从右往左 ^
 * */

// 后缀表达式求值维护一个栈
// 操作数直接入栈
// 遇到操作符从栈中取出两个元素进行运算 再将结果入栈
// 运算顺序为s[top-1] op s[top]

#include "iostream"
#include "stack"
#include "cmath"
using namespace std;
const int SIZE = 1e6+10;
char t[SIZE];//输入序列
stack<char> s;
stack<double> ans;

int pri[200];

void op(char  type){
    double a,b;
    b = ans.top();
    ans.pop();
    a = ans.top();
    ans.pop();
    switch (type) {
        case '+':
            ans.push(a+b);
            break;
        case '-':
            ans.push(a-b);
            break;
        case '*':
            ans.push(a*b);
            break;
        case '/':
            ans.push(a/b);
            break;
        case '^':
            ans.push(pow(a,b));
            break;
    }
}

int main(){
    cin>>t;
    //(1+2)*(3-4)/(5^6)
    pri['+']=pri['-']=1;
    pri['*']=pri['/']=2;
    pri['^']=3;//从右侧开始运算
    for(int i=0;t[i];++i){
        if(t[i]=='(')s.push('(');
        else if(t[i]>='0'&&t[i]<='9')cout<<t[i],ans.push(t[i]-'0');
        else if(t[i]=='+'||t[i]=='-'){//+,-号 不为空或不为( 输出优先级大于等于它的所有元素 入栈
            while(s.size()&&s.top()!='('&&pri[s.top()]>=1){
                op(s.top());
                cout<<s.top();
                s.pop();
            }
            s.push(t[i]);
        }else if(t[i]=='*'||t[i]=='/'){//* /号不为空输出优先级大于等于它的所有元素 入栈
            while(s.size()&&s.top()!='('&&pri[s.top()]>=2){
                op(s.top());
                cout<<s.top();
                s.pop();
            }
            s.push(t[i]);
        }else if(t[i]=='^'){//输出优先级大于^的符号 入栈
            while(s.size()&&s.top()!='('&&pri[s.top()]>3){
                op(s.top());
                cout<<s.top();
                s.pop();
            }
            s.push(t[i]);
        }else if(t[i]==')'){//出栈到(
            char top;
            do{
                top = s.top();
                  if(top!='(')cout<<top,op(top);
                s.pop();
            }while(top!='(');
        }
    }
    while(s.size()){
        op(s.top());
        cout<<s.top();
        s.pop();
    }
    printf("\n%lf",ans.top());
    //(1-2)*(3+4)^2^3/6
    //960,800.16666666666666666666666667
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
        //优化 当t[i+1]==t[j+1]时，若匹配时t[j+1]未匹配成功，则t[i+1]也不会匹配成功
        //此时 可令ne[i]=ne[j];
        //if(t[i+1]==t[j+1])ne[i]=ne[j];
        //else
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

trie字符串计数
```c++
#include "iostream"
using namespace std
const int SIZE = 1e6+10;
int node[SIZE][26],cnt[SIZE],idx;
char s[SIZE];

void insert(char *s){
    int pos =0,cur;
    for(int i=0;s[i];++i){
        cur = s[i]-'a';
        if(!node[pos][cur])node[pos][cur]=++idx;
        pos = node[pos][cur];
    }
    cnt[pos]++;
}

int query(char *s){
    int pos=0,cur;
    for(int i=0;s[i];++i){
        cur = s[i]-'a';
        if(!node[pos][cur])return 0;
        pos = node[pos][cur];
    }
    return cnt[pos];
}

int main(){
    int n;
    cin>>n;
    while(n--){
        char type;
        cin>>type>>s;
        if(type=='Q')
            printf("%d\n",query(s));
        else
            insert(s);
    }
}
```
并查集
```c++
/*
 * 并查集路径压缩
 * 初始化时指向自身 p[i]=i
 * 查询路径上的非根节点指向根节点并返回根节点 if(p[x]!=x)p[x]=find(p[x]);return p[x];
 * 合并时让A的根结点指向B的根节点 p[find(A)]=find(B)
 * 有时需要在AB不属于同一个集合时才能合并 if(find(A)!=find(b))p[find(A)]=find(B);
 * */

const int SIZE = 1e6+10;
int p[SIZE];

void init(int n){for(int i=1;i<=n;++i)p[i]=i};

int find(int x){
    if(p[x]!=x)p[x]=find(p[x]);
    return p[x];
}

void merge(int a,int b){
    if(find(a)!=find(b))
        p[find(a)]=find(b);
}

/*
 * 路径压缩+维护集合大小
 * ss只对根节点有意义，表示根节点对应集合的大小
 * 初始化指向自身 ss为1 p[i]=i,ss[i]=1;
 * 查询时非根节点指向根节点并返回根节点 if(p[x]!=x) p[x]=find(p[x]);return p[x];
 * 合并时B的根加上A集合的大小 再让A的根指向B ss[find(B)]+=ss[find(A)],p[find(A)]=find(B);
 * */

const int SIZE = 1e6+10;
int p[SIZE],ss[SIZE];//不使用size数组是因为size在某些头文件中实现过

void init(int n){for(int i=1;i<=n;++i)p[i]=i,ss[i]=1;}

int find(int x){if(p[x]!=x)p[x]=find(p[x]);return p[x];}

void merge(int a,int b){
    if(find(a)!=find(b)){
        ss[find(b)]+=ss[find(a)];
        p[find(a)]=find(b);
    }
}

/*
 * 路径压缩+维护节点到根节点路径
 * d数组表示节点x到p[x]的距离
 * 初始化时节点指向自身 距离为0 p[i]=i,d[i]=0;
 * 更新查询路径上非根节点的距离并指向、返回根节点 if(p[x]!=x){int u=find(p[x];d[x]+=d[p[x]];p[x]=u)}return p[x];
 * 合并时根据要求初始距离p[find(a)]=find(b),d[find(a)]=distance;
 * */

const int SIZE = 1e6+10;
int p[SIZE],d[SIZE];

void init(int n){for(int i=1;i<=n;++i)p[i]=i,d[i]=0}

int find(int x){
    if(p[x]!=x){
        int u = find(p[x]);
        d[x]+=d[p[x]];
        p[x]=u;
    }
    return p[x];
}

void merge(int a,int b){
    p[find(a)]=find(b);
    d[find(a)]=DISTANCE;
}
```

堆
```c++
//heap priority_queue 可用于输出序列中最小的值 函数有push pop 内部实现有down和up
//heap可以用hp维护堆中i号元素在原序列中的位置 用ph维护原序列i号元素在堆中的位置
const int SIZE = 1e6+10;
int h[SIZE],ph[SIZE],hp[SIZE],idx;

void heap_swap(int a,int b){
    swap(h[a],h[b]);
    swap(hp[a],hp[b]);
    swap(ph[hp[a]],ph[hp[b]]);
}

void down(int u){
    //对于节点u 若存在儿子节点比自己小 与最小的儿子节点交换 并对该儿子节点进行相同的操作
    //pop和序列初始化时使用
    int t=u;
    if(u*2<=idx&&h[u*2]<h[t])t=u*2;
    if(u*2+1<=idx&&h[u*2+1]<h[t])t=u*2+1;
    if(t!=u)heap_swap(t,u),down(t);
}

void　up(int u){
    //若当前节点比父节点小，与父节点交换 push时调用
    while(u/2&&h[u]<h[u/2]){
        heap_swap(u,u/2);
        u/=2;
    }
}

void push(int val){
    //插入元素时，将元素置于堆末并执行up操作
    h[++idx]=val;
    //如果维护hp和ph 需要新参数pos
    //hp[idx]=pos
    //ph[pos]=idx
    up(idx);
}

int pop(){
    //交换第一个元素和最后一个元素 size-- 并对第一个元素进行down操作
    int val = h[1];
    heap_swap(1,idx);
    idx--;
    down(1);
    return val;
}

void D(int k){//删除第K个插入的值
    int heap_pos = ph[k];
    heap_swap(heap_pos,idx);
    idx--;
    if(heap_pos<=idx){
        down(heap_pos);//down和up只会执行其中一个
        up(heap_pos);//不需要动态维护heap_pos
    }
}

void C(int k, int x){//修改第K个插入的值
    h[ph[k]]=x;
    down(ph[k]);//down和up至多执行其中一个
    up(ph[k]);
}

//序列初始化
idx = n;
for(int i=1;i<=n;++i)cin>>h[i];//维护ph和hp时 ph[i]=hp[i]=1;
for(int i=n/2;i;i--)down(i);
```

字符串哈希
```c++
//字符串哈希将字符串视为p进制的数，p通常取131或13331 模取2^64 ，使用unsigned long long存储即可 溢出即取模
//1.不能将字母映射成0，通常a='a',b='b'.....
//2.不考虑冲突，假设不会发生冲突

//前缀字符串hash的性质字串s[l,r]的hash值为h[r]-h[l-1]*p[r-l+1]
//通过前缀字符串哈希可以快速进行字符串比较

typedef unsigned long long ULL;
const int SIZE = 1e6+10;
ULL h[SIZE],p[SIZE];
const ULL P = 131;//底数 取131或13331

//初始化
char s[SIZE];
cin>>s+1;
p[0]=1;//指数初始化
for(int i=1;s[i];++i){
    h[i]=h[i-1]*P+s[i];
    p[i]=p[i-1]*P;
}

ULL get(int l,int r) {//返回序列s[l,r]的hash值
    return h[r]-h[l-1]*p[r-l+1];
}
```

深搜经典问题
```c++
//N皇后问题
//深搜时保证横纵不冲突：每层仅放一个 每列仅放一个 用has维护
//左上方向和右上方向进行判断 保证不冲突
#include "iostream"
using namespace std;
const int SIZE =15;
int n;
bool has[SIZE];
bool v[SIZE][SIZE];
int count;

bool judge(int level,int pos){
    int i,j;
    i = level,j=pos;

    while(--i&&--j){
        if(v[i][j])return false;
    }
    i = level,j=pos;
    while((--i)&&++j<=n){
        if(v[i][j])return false;
    }
    return true;
}

void dfs(int level){
    if(level==n+1){
        count++;
        for(int i=1;i<=n;++i){
            for(int j=1;j<=n;++j)
                if(v[i][j])printf("Q");
                else printf(".");
            puts("");
        }
        puts("");
        return ;
    }
    for(int i=1;i<=n;++i){
        if(!has[i]&&judge(level,i)){
            has[i]=1;
            v[level][i]=1;
            dfs(level+1);
            has[i]=0;
            v[level][i]=0;
        }
    }
}

int main(){
    cin>>n;
    dfs(1);
    cout<<count;
}

//序列字典序问题
//使用algorithm头文件下的 prev_permutation 或 next_permutation(iterator begin,iterator end)
//该函数求序列前一个[后一个]字典序 并返回是否成功[成功则输出 用do while循环]
//如a[1,n] do{print_array}while(next_permutation(a+1,a+n+1));
```

广搜经典问题
```c++
//八数码问题 
/*
 * 3 2 1
 * 6 5 4
 * x 8 7
 * 通过对x与周围元素进行交换 问需要多少次才能得到下图，不能得到输出-1
 * 1 2 3
 * 4 5 6
 * 7 8 x
 * 
 * 八数码需要解决的问题是状态表示的问题 以及保存状态的距离
 * 这里用string保存状态 用unordered_map<string,int>来保存状态的距离
 * */
#include "iostream"
#include "cstring"
#include "algorithm"
#include "queue"
using namespace std;

int bfs(string start){
    int dx[]={-1,0,1,0},dy[]={0,1,0,-1};//二维数组四个方向 恰好倒置
    string end = "12345678x";
    queue<string> q;
    q.push(start);
    unordered_map<string,int> dist;
    dist[start]=0;
    while(q.size()){
        string cur = q.front();
        q.pop();
        int cur_dis = dist[cur];
        if(cur == end)return cur_dis;//string可以进行比较
        int pos = cur.find('x');
        int x = pos/3,y=pos%3;//
        //从0开始一维转二维 x=pos/len y=pos%len
        //从0开始二维转一维 x*len+y
        for(int i=0;i<4;++i){
            int a=x+dx[i],b=y+dy[i];
            if(a>=0&&a<3&&b>=0&&b<3){
                swap(cur[pos],cur[a*3+b]);//
                if(dist.count(cur)==0){//unordered_map 用于查找
                    q.push(cur);
                    dist[cur]=cur_dis+1;
                }
                swap(cur[pos],cur[a*3+b]);//添加后恢复
            }
        }
    }
    return -1;
}

int main(){
    string start;
    char s[2];
    for(int i=0;i<9;++i){
        cin>>s;
        start+=s[0];
    }
    cout<<bfs(start)<<endl;
}
```

树的重心（树上DFS）
```c++
/* 树的重心的定义：重心是指树中的一个节点，如果将这个点删除后，剩余各个连通块中点数的最大值最小，那么这个节点被称为树的重心。
 * 给定一棵树：输出这棵树的重心，并输出删掉重心之后，剩余连通块的最大值
 * 对树进行DFS时，可以将树按照当前点进行划分，分为子树，自身，父节点部分
 * 通过DFS得到每个子树的大小，可以计算得到父节点部分的大小，从而维护各个部分的最大值
 * 在DFS过程中记录最大值最小的点并维护最大快的最小值
 * */

#include "iostream"
using namespace std;
const int SIZE = 1e6+10;
int head[SIZE],idx,n;

struct Edge{
    int next;
    int ver;
}edge[SIZE];

void add(int a,int b){
    edge[++idx].ver = b;
    edge[idx].next = head[a];
    head[a] = idx;
}

bool v[SIZE];

int min_max_son = 0x3f3f3f3f;

int dfs(int pos){
    v[pos]=1;
    int max_son = 0;
    int count_son = 0;//所有子树和
    int ne = head[pos];
    while(ne){
        if(!v[edge[ne].ver]){
            int son_num = dfs(edge[ne].ver);
            count_son += son_num;
            max_son  = max(max_son,son_num);//维护子树的最大值
        }
        ne = edge[ne].next;
    }
    max_son = max(max_son,n-count_son-1);//父节点部分等于总数-子树和-1
    min_max_son = min(max_son,min_max_son);//维护最大块的最小值
    return count_son+1;
}

int main(){
    cin>>n;
    for(int i=1;i<n;++i){
        int a,b;
        cin>>a>>b;
        add(a,b);
        add(b,a);
    }
    dfs(1);
    cout<<min_max_son;
}
```

拓扑排序topology
```c++
/*
 * 拓扑排序维护一个入度数组，从入度为0点的点开始搜索，对每个搜索到的点入度-1
 * 再将入度为0的点加入搜索队列
 * 若搜索结束之后仍有点入度不为0，则说明存在环或自环
 * */

#include "iostream"
#include "queue"
using namespace std;
const int SIZE =1e6+10;
int h[SIZE],idx,indegree[SIZE],n;


struct Edge{
    int next;
    int ver;
}edge[SIZE];

void add(int a,int b){
    indegree[b]++;
    edge[++idx].ver = b;
    edge[idx].next = h[a];
    h[a]=idx;
}

queue<int> top_sort;//记录拓扑排序序列

void get_top_sort(){
    queue<int> q;
    for(int i=1;i<=n;++i){//将入度为0的点入队
        if(indegree[i]==0)q.push(i),top_sort.push(i);
    }
    while(q.size()){
        int cur = q.front();
        q.pop();
        int ne = h[cur];
        while(ne){
            indegree[edge[ne].ver]--;//对搜索到的每个点入度减1
            if(indegree[edge[ne].ver]==0)q.push(edge[ne].ver),top_sort.push(edge[ne].ver);//入度为0，入队
            ne = edge[ne].next;
        }
    }
    int has_loop = 0;
    for(int i=1;i<=n;++i)
        if(indegree[i])has_loop=1;
    if(has_loop)cout<<-1;//存在环
    else while(top_sort.size()){//输出序列
        int cur = top_sort.front();
        top_sort.pop();
        cout<<cur<<" ";
    }
}

int main(){
    int m;
    cin>>n>>m;
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b);
    }
    get_top_sort();
}
```

朴素Dijkstra
```c++
/*
 * O(n^2)
 * 1.初始化距离
 * 2.找到未找到最短路点中距离最小的点
 * 3.该点为下一个最短点
 * 4.用该点更新其他点的距离
 * */

int g[N][N],d[N];
bool  v[N];
void dijkstra(int s,int n){
    memset(d,0x3f,sizeof(d));
    memset(v,0,sizeof(v));
    d[s]=0;
    for(int i=1;i<n;++i){//n-1轮
        int t=-1;//t=-1 与(t==-1||condition)是常用找下标的方式
        for(int j=1;j<=n;++j)//找到未标记的距离最近的点
            if(!v[j]&&(t==-1||d[t]>d[j]))
                t=j;
        for(int j=1;j<=n;++j)//更新其他点的距离
            d[j]=min(d[j],d[t]+g[t][j]);
        v[t]=1;//标记
    }
    for(int i=1;i<=n;++i)//输出距离
        cout<<d[i]<<" ";
}
```

dijkstra堆优化
```c++
/*
 * c++的priority_queue是使用less<T>实现的最大堆
 * 要实现最小堆的效果：在定义时显示指明使用greater<T>
 * priority_queue<PII,vector<PII>,greater<PII> > q;
 * 总之：实现A<B实现最大堆 实现A>B实现最小堆
 * */

#include <cstring>
#include "iostream"
#include "queue"
using namespace std;
const int SIZE = 1e6+10;
int h[SIZE],d[SIZE],idx;
bool v[SIZE];
typedef pair<int,int> PII ;

struct Edge{
    int next;
    int ver;
    int dis;
}edge[SIZE];

void add(int a,int b,int d){
    edge[++idx].ver = b;
    edge[idx].next = h[a];
    h[a]=idx;
    edge[idx].dis = d;
}

void dijkstra(int s,int n){
    priority_queue<PII,vector<PII>,greater<PII>> q;
    //less<T>构建最大堆，使用greater<T>构建最小堆
    memset(d,0x3f,sizeof(d));
    memset(v,0,sizeof(v));
    d[s]=0;
    q.push(PII(0,s));
    while(q.size()){
        PII t = q.top();//优先级队列使用top而非front
        q.pop();
        int cur = t.second;
        int cur_dis = t.first;
        if(v[cur])continue;
        v[cur] = 1;
        int ne = h[cur];
        while(ne){
            int next_ver = edge[ne].ver;
            if(!v[next_ver]&&d[next_ver]>d[cur]+edge[ne].dis)//入队
                d[next_ver] = d[cur]+edge[ne].dis,q.push(PII(d[next_ver],next_ver));
            ne = edge[ne].next;
        }
    }
    if(d[n]==0x3f3f3f3f)cout<<-1;
    else cout<<d[n];
}
```

bellman_ford算法 [有边数限制的最短路](https://www.acwing.com/problem/content/description/855/)
```c++
/*
 * bellman_ford容忍负权边，但是不允许存在负环
 * bellman_ford判断负环的方法：若程序能够进行第N轮松弛，则说明存在负环
 * */
#include "iostream"
#include "cstring"
using namespace std;
const int N = 510;
const int M = 1e4+10;
int d[N],backup[N],idx,n,m;

struct Edge{//bellman_ford算法对边的存储没有要求，只需要能够遍历所有边
    int a,b,dis;
}edge[M];

inline void add(int a,int b,int c){
    edge[++idx]={a,b,c};
}

void bellman_ford(int k){
    memset(d,0x3f,sizeof(d));
    d[1]=0;
    for(int i=1;i<=k;++i){//每循环一次 走一步
        memcpy(backup,d,sizeof(d));//普通的bellman_ford会出现串联的情况
        for(int j=1;j<=m;++j)//更新时只用备份的结果，使得每次循环严格更新一次
            d[edge[j].b]=min(d[edge[j].b],backup[edge[j].a]+edge[j].dis);
    }
    if(d[n]>0x3f3f3f3f/2)cout<<"impossible";
    //这里使用d[n]>0x3f3f3f3f/2是因为存在负权边 如a-/-b a-/-c b-->c=-1
    //a-->c = 0x3f3f3f3f-1 !=0x3f3f3f3f
    else cout<<d[n];
}
/*
 * 串联现象：1 2 1;2 3 1;1 3 3;
 * 更新1->2  2->3 1->3 此时得到了1->3的最短路 但最短路走了两步
 * 每次只使用前一次的结果更新才能保证每次只能走一步
 * */  

int main(){
    int k;
    cin>>n>>m>>k;
    for(int i=1;i<=m;++i){
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    bellman_ford(k);
}
```

SPFA:队列优化的Bellman_ford算法
```c++
/*
 * bellman_ford暴力对所有边进行松弛，SPFA基于这样一个事实进行优化：
 * 对于边AB，只有点A进行松弛了，点B的松弛才具有意义
 * 因此用队列维护一个更新序列，当某一点更新之后，将其邻接点入队（相同的点不需要重复入队，因此需要一个布尔数组对入队的点进行标记）
 * */
#include <cstring>
#include "iostream"
#include "queue"
using namespace std;
const int N =1e5+10;
const int M = 1e5+10;
int h[N],d[N],idx,n,m;
bool v[N];

struct Edge{
    int ver;
    int next;
    int dis;
}edge[M];

void add(int a,int b,int c){
    edge[++idx].ver =b;
    edge[idx].dis = c;
    edge[idx].next = h[a];
    h[a]=idx;
}

void spfa(){
    memset(d,0x3f,sizeof d);
    memset(v,0,sizeof(v));
    d[1]=0;
    queue<int> q;
    q.push(1);
    v[1]=1;
    while(q.size()){
        int u = q.front();
        q.pop();
        v[u]=0;
        int ne = h[u];
        while(ne){
            int cur=  edge[ne].ver;
            if(d[cur]>d[u]+edge[ne].dis){
                d[cur]=d[u]+edge[ne].dis;
                if(!v[cur])q.push(cur),v[cur]=1;
            }
            ne = edge[ne].next;
        }
    }
    if(d[n]>0x3f3f3f3/2)cout<<"impossible";
    else cout<<d[n];
}

int main (){
    cin>>n>>m;
    while (m--){
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    spfa();
}
```

SPFA判断负环
```c++
/*
 * SPFA判断负环的方法：对于图中任意一对点(a,b),如果不存在负环，则a到b的最短路最多经过n-1个点（除a意外的所有点）。当a到b的最短路长度超过n-1时，则说明存在负环。
 * 由此，可以使用cnt标记a到b最短路经过的点数
 * 当d[v]<d[u]+g[u][v] d[v]=d[u]+g[u][v]时，cnt[v]=cnt[u]+1
 * 朴素的方法：对图中每个点进行SPFA，看是否存在负环（仅一个点只能知道该点到其他点的路径是否存在负环，由此需要对每个点进行SFPA）
 * 
 * 巧妙的方法：假设存在一个点0，它到所有点都存在一条长度为0的边，将所有点入队并进行SPFA，只需一次就i能判断负环。
 * */
#include "iostream"
#include "cstring"
#include "queue"
using namespace std;
const int N = 2010;
const int M = 10010;
int h[N],d[N],cnt[N],idx,n,m;
bool v[N];

struct Edge{
    int next;
    int ver;
    int dis;
}edge[M];

void add(int a,int b,int c){
    edge[++idx].ver =b;
    edge[idx].dis = c;
    edge[idx].next = h[a];
    h[a]=idx;
}

void SFPA(){
    queue<int> q;
    for(int i=1;i<=n;++i)
        q.push(i),v[i]=1;
        
    while(q.size()){
        int pos = q.front();
        q.pop();
        v[pos]=0;
        if(cnt[pos]==n){
            cout<<"Yes";
            return;
        }
        int ne = h[pos];
        while(ne){
            int cur = edge[ne].ver;
            if(d[cur]>d[pos]+edge[ne].dis){
                d[cur]=d[pos]+edge[ne].dis;
                cnt[cur]=cnt[pos]+1;
                if(!v[cur])v[cur]=1,q.push(cur);
            }
            ne = edge[ne].next;
        }
    }
    cout<<"No";
}

int main(){
    cin>>n>>m;
    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    SFPA();
    return 0;
}

//bellman_ford也是类似的思路
struct Edge{
    int a,b,dis;
}edge[M];

void bellman_ford(){
    for(int i=1;i<n;++i){//松弛n-1次
        for(int j=1;j<=m;++j){
            if(d[edge[j].b]>d[edge[j].a]+edge[j].dis)
                d[edge[j].b]=d[edge[j].a]+edge[j].dis;
        }
    }
    //进行第n次松弛
    for(int j=1;j<=m;++j){
            if(d[edge[j].b]>d[edge[j].a]+edge[j].dis){//第n次能够松弛，则说明存在负环
                cout<<"Yes";
                return;
            }
    }
    cout<<"No";
}
```

floyd算法
```c++
/*
 * ijk算法，可以用于判断负环：在计算过程中如果出现g[i][i]<0,则说明存在负环
 * */
#include "iostream"
#include "cstring"
using namespace std;
const int N = 201;
int g[N][N],n,m,k;

int main(){
    cin>>n>>m>>k;
    memset(g,0x3f,sizeof(g));
    for(int i=1;i<=n;++i)
        g[i][i]=0;
    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        g[a][b]=min(g[a][b],c);
    }
    
    for(int k=1;k<=n;++k)
        for(int i=1;i<=n;++i)
            for(int j=1;j<=n;++j)
                g[i][j]=min(g[i][j],g[i][k]+g[k][j]);
    while(k--){
        int a,b;
        cin>>a>>b;
        if(g[a][b]>0x3f3f3f3f/2)cout<<"impossible"<<endl;
        else cout<<g[a][b]<<endl;
    }
}
```

prim算法
```c++
/*
 * prim求最小生成树
 * 基本思路：每次将最小生成树集合外最短的边加入集合
 * 实现方法：1.找到集合外的距离最近的点2.将点加入集合3.用新加入的点更新集合外点到集合的距离
 * 使用d数组维护到最小生成树集合的距离，初始化INF，令其中任意一点d=0
 * */
#include "iostream"
#include "cstring"
using namespace std;
const int N = 510;
int g[N][N],d[N],n,m;
bool v[N];

void prim(){
    memset(d,0x3f,sizeof(d));//初始化距离为INF
    d[1]=0;//将任意一点距离置0 作为下一个加入集合的点（此时集合为空集）
    for(int i=1;i<=n;++i){//重复操作n次
        int t=-1;
        for(int j=1;j<=n;++j)
            if(!v[j]&&(t==-1||d[t]>d[j]))t=j;//找到集合外距离最近的点
        v[t]=1;//将该点加入集合
        for(int j=1;j<=n;++j)
            if(!v[j])d[j]=min(d[j],g[t][j]);//更新集合外其它点到集合的距离
    }
    int ans = 0;
    for(int i=1;i<=n;++i)
        if(d[i]==0x3f3f3f3f){//存在边长为INF 则说明不是连通图
            cout<<"impossible";
            return;
        }else ans+=d[i];//得到生成树大小
    
    cout<<ans;
}

int main(){
    memset(g,0x3f,sizeof(g));
    cin>>n>>m;
    while(m--){
        int a,b,c;
        cin>>a>>b>>c;
        g[a][b]=g[b][a]=min(g[a][b],c);//有重边
    }
    prim();
    return 0;
}
```

krustal算法
```c++
/*
 * 每次选择最短的边，如果该边的两个端点属于不同的集合，则说明是生成树的一条边
 * 并查集+优先级队列（或者排序，保证有序即可）
 * */

#include "iostream"
#include "algorithm"
using namespace std;
const int SIZE = 2e5+10;
int p[SIZE],idx,n,m;

struct Edge{
    int a,b,dis;
}edge[SIZE];

struct cmp{
    bool operator()(Edge a,Edge b){
        return a.dis<b.dis;
    }
};

int find(int x){
    if(p[x]!=x)p[x]=find(p[x]);
    return p[x];
}

void merge(int a,int b){
    if(find(a)!=find(b)){
        p[find(a)]=find(b);
    }
}

void krustal(){
    for(int i=1;i<=n;++i)
        p[i]=i;
    sort(edge+1,edge+m+1,cmp());
    int ans =0,cnt = 0;
    for(int i=1;i<=m;++i){
        int a=edge[i].a,b=edge[i].b,dis=edge[i].dis;
        if(find(a)==find(b))continue;
        merge(a,b);
        ans += dis;
        cnt++;
    }
    if(cnt==n-1)cout<<ans;
    else cout<<"impossible";
}

int main(){
    cin>>n>>m;
    for(int i=1;i<=m;++i){
        int a,b,c;
        cin>>a>>b>>c;
        edge[++idx]={a,b,c};
    }
    krustal();
}
```

染色体法判断二分图
```c++
/*
 * 使用深度优先搜索给每个点标记颜色，当两个相邻点颜色相同，则不是二分图
 * */
#include <cstring>
#include "iostream"
using namespace std;
const int SIZE = 2e5+10;
int h[SIZE],color[SIZE],n,m,idx;

struct Edge{
    int next;
    int ver;
}edge[SIZE];

bool dfs(int pos,int c){//将pos点标记为颜色c 发生冲突返回fase 未发生冲突返回true
    color[pos]=c;
    int ne = h[pos];
    while(ne){
        if(color[edge[ne].ver]==-1&&dfs((edge[ne].ver),1-c)==false)return false;//后续节点发生冲突 返回false
        else if(c == color[edge[ne].ver])return false;//当前节点与相邻节点颜色相同 返回false
        ne = edge[ne].next;
    }
    return true;//当前及后续节点均未冲突 返回true
}

bool check(){
    memset(color,-1,sizeof(color));
    bool have_confict = false;
    for(int i=1;i<=n;++i)//可能存在多个连通块
        if(color[i]==-1&&dfs(i,1)==false)
            return false;
    return true;
}

void add(int a,int b){
    edge[++idx].ver = b;
    edge[idx].next = h[a];
    h[a]=idx;
}

int main(){
    cin>>n>>m;
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b);
        add(b,a);
    }
    if(check())cout<<"Yes";
    else cout<<"No";
}
```

匈牙利算法二分图匹配
```c++
/*
 * 左侧n1个顶点 右侧n2个顶点 仅存储n1到n2的单向边
 * match数组存储n2匹配的左侧点 v数组表示是否遍历过（遍历过意味着名花有主）
 * 基本思路：在执行find的时候，如果碰到一个点被其他点匹配了，让匹配的那个点寻找一个新匹配
 * 此时标记C意味着，该点一定会被A或者B匹配，而不能容忍第三个人插足
 * */
#include "iostream"
#include "cstring"
using namespace std;
const int N = 510;
const int M = 1e5+10;
int n1,n2,m,idx,h[N],match[N];
bool v[N];;
struct Edge{
    int ver;
    int next;
}edge[M];

void add(int a,int b){
    edge[++idx].ver =b;
    edge[idx].next = h[a];
    h[a]=idx;
}

bool find(int cur){
    int ne = h[cur];
    while(ne){
        int pos = edge[ne].ver;
        if(!v[pos]){
            v[pos]=1;
            if(match[pos]==0||find(match[pos])){
                match[pos]=cur;
                return true;//匹配成功
            }
        }
        ne = edge[ne].next;
    }
    return false;//匹配失败
}

int count(){
    int ans = 0;
    for(int i=1;i<=n1;++i){
        memset(v,0,sizeof(v));
        if(find(i))ans++;
    }
    return ans;
}

int main(){
    cin>>n1>>n2>>m;
    while(m--){
        int a,b;
        cin>>a>>b;
        add(a,b);
    }
    cout<<count();
}
```