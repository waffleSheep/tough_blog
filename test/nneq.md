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
$\mathbf{W}$ is size $(k,k)$
$\mathbf{A}$ is size $(n-k+1,m-k+1)$
$$\mathbf{A}=\mathbf{X}*\mathbf{W}$$
$$a_{i,j}=\sum_{p=1}^{k}\sum_{q=1}^{k}x_{i+p-1,j+q-1}w_{p,q}$$


### Backward

















