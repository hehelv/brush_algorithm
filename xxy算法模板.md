* quick_sort
```C++
void quick_sort(int *a,int l,int r){
	int i=l,j=r,mid=a[(l+r)>>1];
	do{
		while(a[i]<mid)i++;
		while(a[j]>mid)j--;
		if(i<=j)swap(a[i++],a[j--]);
	}while(i<j);
	if(l<j)quick_sort(a,l,j);
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
