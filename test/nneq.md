## Linear Algebra

### Forward
$$c={\left\| \mathbf{A}\mathbf{x} -\mathbf{b} \right\|}^2_2$$
$$\mathbf{r}= \mathbf{A}\mathbf{x}$$
$$c={\left\| \mathbf{r} -\mathbf{b} \right\|}^2_2$$
$$r_{j}=\sum_{i=1}^{n}{a_{ji}x_{i}}$$
$$c=\sum_{j=1}^{n}{\left(r_{j}-b_{j}\right)^{2}}$$

### Backward
$$\frac{\partial c}{\partial x_{i}}=\sum_{j=1}^{n}{2\left(r_{j}-b_{j}\right)}\frac{\partial r_j}{\partial x_i}$$
$$\frac{\partial r_{j}}{\partial x_{i}}=a_{ji}$$
$$\frac{\partial c}{\partial x_{i}}=\sum_{j=1}^{n}{2a_{ji}\left(-b_{j}+\sum_{i=1}^{n}{a_{ji}x_{i}}\right)}$$
$$\frac{\partial c}{\partial \mathbf{x}}=2\mathbf{A}^{T}\left(\mathbf{A}\mathbf{x}-\mathbf{b}\right)$$
---
## Neural net
### forward
$$\mathbf{a}^{(0)}=\mathbf{x}$$
$$\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)}=\sigma \left(\mathbf{z}^{(l)} \right)$$
$$l=1,2,3,\dots,L$$
$$\mathbf{c}=\frac{1}{n}{\left\| \mathbf{a}^{(L)} -\mathbf{y} \right\|}^2_2$$
$$a^{(0)}_{i}=x_i$$
$$z^{(l)}_{i}=b^{(l)}_i+\sum_{i=1}^{n}w^{(l)}_{ij}a^{(l-1)}_{j}$$
$$a^{(l)}_{i}=\sigma \left(z^{(l)}_{i}\right)$$
$$l=1,2,3,\dots,L$$
$$c=\frac{1}{n}\sum_{i=1}^{n}{\left(a^{(l)}_{i}-y_i\right)^{2}}$$
### backward
$$\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i}=\sigma'\left(z^{(l)}_{i}\right)$$
$$\frac{\partial z^{(l)}_i}{\partial b^{(l)}_i}=1$$
$$\frac{\partial z^{(l)}_{i}}{\partial w^{(l)}_{ij}}=a^{(l-1)}_{j}$$
$$\frac{\partial z^{(l)}_{i}}{\partial a^{(l-1)}_{j}}=w_{ij}$$
### Conversion
cost against last layer
$$
\frac{\partial c}{\partial a^{(L)}_{i}}=\frac{2}{n}\left(a^{(L)}_{i}-y_{i}\right)
$$
$$
\frac{\partial c}{\partial \mathbf{a}^{(L)}}=\frac{2}{n}\left(\mathbf{a}^{(L)}-\mathbf{y}\right)
$$
cost against untransformed z layer
$$
\begin{align}
	\frac{\partial c}{\partial z^{(l)}_i}&=\sum_{j=1}^{n}{\frac{\partial c}{\partial a^{(l)}_{j}}\frac{\partial a^{(l)}_j}{\partial z^{(l)}_i}} \\
	&=\frac{\partial c}{\partial a^{(l)}_{i}}\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i} \\
	&=\frac{\partial c}{\partial a^{(l)}_{i}}\sigma'\left(z^{(l)}_{i}\right)
\end{align}
$$
$$
\frac{\partial c}{\partial \mathbf{z}^{(l)}}=\frac{\partial c}{\partial \mathbf{a}^{(l)}}\odot \sigma'\left(\mathbf{z}^{(l)}\right)
$$
cost against bias
$$
\frac{\partial c}{\partial b^{(l)}_i}=\frac{\partial c}{\partial z^{(l)}_i}
$$
$$
\frac{\partial c}{\partial \mathbf{b}^{(l)}}=\frac{\partial c}{\partial \mathbf{z}^{(l)}}
$$
cost against weight
$$
\begin{align}
	\frac{\partial c}{\partial w^{(l)}_{ij}}&=\sum_{k=1}^{n}{\frac{\partial c}{\partial z^{(l)}_{k}}\frac{\partial z^{(l)}_k}{\partial w^{(l)}_{ij}}} \\
	&=\frac{\partial c}{\partial z^{(l)}_{i}}\frac{\partial z^{(l)}_i}{\partial w^{(l)}_{ij}} \\
	&=\frac{\partial c}{\partial z^{(l)}_{i}}a^{(l-1)}_{j}
\end{align}
$$
$$
\frac{\partial c}{\partial \mathbf{W}^{(l)}}=\frac{\partial c}{\partial \mathbf{z}^{(l)}}{\mathbf{a}^{(l-1)}}^T
$$
Cost against previous layer
$$
\begin{align}
	\frac{\partial c}{\partial a^{(l-1)}_i}&=\sum_{j=1}^{n}{\frac{\partial c}{\partial z^{(l)}_{j}}\frac{\partial z^{(l)}_j}{\partial a^{(l-1)}_i}} \\
	&=\sum_{j=1}^{n}{\frac{\partial c}{\partial z^{(l)}_{j}}w_{{ji}}}
\end{align}
$$
$$
\frac{\partial c}{\partial \mathbf{a}^{(l-1)}}={\mathbf{W}^{(l)}}^T\frac{\partial c}{\partial \mathbf{z}^{(l)}}
$$
---
## Convolution 1 layer
### Forward
$\mathbf{X}$ is size $(n,m)$
$\mathbf{W}$ is size $(k_{1},k_{2})$
$\mathbf{A}$ is size $(n-k_{1}+1,m-k_{2}+1)$
$$\mathbf{A}=\mathbf{X}*\mathbf{W}$$
$$a_{i,j}=\sum_{p=1}^{k_{1}}\sum_{q=1}^{k_{2}}x_{i+p-1,j+q-1}w_{p,q}$$
### Backward
$$\frac{\partial a_{i,j}}{\partial w_{p,q}}=x_{i+p-1,j+q-1}$$

$$  
\frac{\partial a_{i,j}}{\partial x_{d,e}}  
=  
\begin{cases}  
w_{d-i+1,\;e-j+1},  
& \text{if } 1 \le d-i+1 \le k_1 \text{ and } 1 \le e-j+1 \le k_2,\\  
0,  
& \text{otherwise.}  
\end{cases}  
$$
$$\frac{\partial a_{i,j}}{\partial x_{d,e}}=\sum_{p=1}^{k_{1}}\sum_{q=1}^{k_{2}}w_{p,q}\delta_{i,d-p+1} \delta_{j,e-q+1}​$$

### Conversion
cost against kernel
$$
\begin{align}
	\frac{\partial c}{\partial w_{p,q}}&=\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1} \frac{\partial c}{\partial a_{i,j}} \frac{\partial a_{i,j}}{\partial w_{p,q}} \\
	&=\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1} x_{i+p-1,j+q-1}\frac{\partial c}{\partial a_{i,j}}
\end{align}
$$
$$\frac{\partial c}{\partial \mathbf{W}}=\mathbf{X}* \frac{\partial c}{\partial \mathbf{A}}$$
cost against prev layer
$$
\begin{align}
	\frac{\partial c}{\partial x_{d,e}}&=\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1} \frac{\partial c}{\partial a_{i,j}} \frac{\partial a_{i,j}}{\partial x_{d,e}} \\
	&=\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1} \frac{\partial c}{\partial a_{i,j}} \sum_{p=1}^{k_{1}}\sum_{q=1}^{k_{2}}w_{p,q}\delta_{i,d-p+1} \delta_{j,e-q+1} \\
	&=\sum_{p=1}^{k_{1}}\sum_{q=1}^{k_{2}} w_{p,q}\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1}\frac{\partial c}{\partial a_{i,j}} \delta_{i,d-p+1} \delta_{j,e-q+1}​
\end{align}
$$
Set
$$
\mathbf{G}=\text{pad}_{k_{1}-1,k_{2}-1}\left(\frac{\partial c}{\partial \mathbf{A}} \right)
$$
we see
$$
\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1}\frac{\partial c}{\partial a_{i,j}} \delta_{i,d-p+1} \delta_{j,e-q+1}=g_{d+k_{1}-p,e+k_{2}-q}
$$
Set $r=k_{1}-p+1$ and $s=k_{2}-q+1$
$$
\begin{align}
	\frac{\partial c}{\partial x_{d,e}}
	&=\sum_{p=1}^{k_{1}}\sum_{q=1}^{k_{2}} w_{p,q}\sum_{i=1}^{n-k_{1}+1}\sum_{j=1}^{m-k_{2}+1}\frac{\partial c}{\partial a_{i,j}} \delta_{i,d-p+1} \delta_{j,e-q+1} \\
	&=\sum_{p=1}^{k_{1}}\sum_{q=1}^{k_{2}} w_{p,q}g_{d+k_{1}-p,e+k_{2}-q} \\
	&=\sum_{r=1}^{k_{1}}\sum_{s=1}^{k_{2}} g_{d+r-1,e+s-1}w_{k_{1}-r+1,k_{2}-s+1}​
\end{align}
$$
$$
\begin{align}
	\frac{\partial c}{\partial \mathbf{X}}&=\mathbf{G}*\text{flip}\left(\mathbf{W}\right) \\
	&=\text{pad}_{k_{1}-1,k_{2}-1} \left(\frac{\partial c}{\partial \mathbf{A}} \right)*\text{flip}\left(\mathbf{W}\right)
\end{align}
$$

## Convolution multiple layer
### Forward
$$
O_j = \sum_{i=1}^{n}{I_i*K_{ij}}
$$
### Backward
$$
\begin{align}
\frac{\partial c}{\partial I_i} &= \text{pad}(\frac{\partial c}{\partial O_j}) * \text{flip}(K_{ij})\\
\frac{\partial c}{\partial K_{ij}} &= I_i * \frac{\partial c}{\partial O_j}
\end{align}
$$

## Relu
### Forward



### Backward