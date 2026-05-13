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
$$\frac{\partial c}{\partial a^{(L)}_{i}}=\frac{2}{n}\left(a^{(L)}_{i}-y_{i}\right)$$
$$\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i}=\sigma'\left(z^{(l)}_{i}\right)$$
$$\frac{\partial z^{(l)}_i}{\partial b^{(l)}_i}=1$$
$$\frac{\partial z^{(l)}_{i}}{\partial w^{(l)}_{ij}}=a^{(l-1)}_{j}$$
$$\frac{\partial z^{(l)}_{i}}{\partial a^{(l-1)}_{j}}=w_{ij}$$

























