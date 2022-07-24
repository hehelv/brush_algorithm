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
	while(C.size()>1&&C.bakc()==0)C.pop_back();
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
