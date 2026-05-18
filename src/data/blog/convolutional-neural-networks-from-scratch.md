---
author: Karimi Zayan
pubDatetime: 2026-04-17T12:00:00Z
title: Convolutional Neural Networks from scratch
slug: convolutional-neural-networks-from-scratch
featured: true
draft: false
tags:
  - machine-learning
  - math
  - computer-science
  - linear-algebra
  - calculus
description: Building a CNN from the ground up — the math, the code, and the intuition.
---

## Pre-amble (feel free to skip)

I started writing this article more than 4 years ago in 2022 but this article did not take 4 years to make. It was originally a two part article written for NUS High's Computer Science Interest Group, Appventure. However, due to various reasons such as scaling complexity, school work, dwindling motivation and many additional commitments, I never finished the second part.

The original blogpost can be found [here](https://nush.app/blog/2022/05/26/cnn-from-scratch-1/) (Please do not read, it is really bad). Over the last 4 years, I was pestered by my editor Prannaya to finish this but today I finally do it all. This is a full rewrite to make the original more clear and accessible. For the full story, on why it took so long refer to the addendum at the end. 

## Introduction

AI specifically those powered by neural networks have taken over. I setup this entire website with claude code and I almost write all my code for work with claude code. However despite its widespread use, very few people actually know how neural networks work and the math and logic behind them. Well, people actually do kind of have an idea on how it works but their understanding is opaque, they can explain it in general terms but they aren't able to build it themselves. Today, we build it ourselves.

In this article, we build a Convolutional Neural Network from scratch with Numpy. I mean it is not exactly from scratch but it is "from scratch enough". The goal is to do it without for loops excepts for iteration to keep the code clean. We will start with a simple example, then we with move on to an example with a simple Feed Forward Neural Network and finally we will have a full working example with a Convolutional Neural Network on the MNIST data set
## Scope

This article lives in a weird gray area. Neural networks are written in a very optimised way and some amount of agreement on how tensors are implemented is needed for it to actually be implemented well. However, I dont want to touch tensors, because it becomes harder to visualise and I am not super familiar with tensor analysis and differential geometry (I dont know at all). The moment you find a "derivative" of a matrix with respect to another matrix, it is already a mostly empty order-4 tensor (4d matrix). A convolution layer with multiple input channels and output channels has a derivative which is an even emptier order 6 tensor. It is not very useful to think about theis kind of tensors for solving these kinds of problems and most college level courses stick to matrix calculus, which I think is fair. 

Furthermore, tensors really come in when you have to do forward differentiation as opposed to backward differentiation (backprop) which is what we will be doing. 

As for matrix calculus, a full deep dive into matrix calculus is not really needed for this topic, so I will first compute derivatives with summations and then I will convert them into their vector form. Thus, rules like matrix calculus chain rule and product rule need not be covered.

Perhaps one day I will write an article to tackle the various cans of worms. But to keep it scoped well, this article only requires a basic understanding of Linear Algebra, Calculus and Python.

## Gradient Descent Example (Linear System Solution)

First, lets import **Numpy**.

```python
import numpy as np
```

Gradient Descent on a full neural network is a pretty daunting task so let us try Gradient Descent on a simple example. Let's start with simple simultaneous equation systems like the one below
$$
\begin{aligned}
4a+2b&=22\\
3a+8b&=49
\end{aligned}
$$
We will attempt to solve this with Gradient Descent.

If you remember, Gradient Descent is a method used to solve any sort of equation by taking steps towards the real value by using the derivative to predict the direction and size of the step. If you remember in calculus, the minimum of the graph will have a tangent of slope 0 and hence we can understand the direction of these "steps" to solve the problem. We just need a function where the derivative and function result approach 0 as you get closer to the true solution. This function is known as the objective function.

As you probably know, a linear equation is written as such:
$$
A \mathbf{x}-\mathbf{b}=0
$$
where $A$ is a known square matrix, $\mathbf{b}$ is a known vector and $\mathbf{x}$ is an unknown vector.

In this case, for the objective function we will use Linear Least Squares (LLS) function as it is an accurate thing to minimize in this case written below.
$$
F(\mathbf{x}) = {||A\mathbf{x}-\mathbf{b}||}_{2}^{2}
$$
### Multivariable Differentiation

But how exactly do we calculate the derivative of a scalar in terms of a vector? Well we have to learn some kind of multivariable calculus, but don't worry it should be all above board for the most part. Multivarible differentiation is not so bad.

Firstly, let's revise derivatives with this simple example:
$$
\begin{aligned}
y&=\sin{\left(x^2\right)}+5\\
\frac{dy}{dx}&=\frac{d}{dx}\left(\sin{\left(x^2\right)}+5\right)\\
&=2x\cos{\left(x^2\right)}
\end{aligned}
$$
For functions with multiple variables, we can find the partial derivative with respect to each of the variables, as shown below.

$$
\begin{aligned}
f(x,y)&=3xy+x^2\\
\frac{\partial f(x,y)}{\partial x}&=3y+2x\\
\frac{\partial f(x,y)}{\partial y}&=3x
\end{aligned}
$$

Basically, variables which you are not computing the derivative with respect to are treated as constants. A thing to understand is that vectors are just a collection of numbers, so an n-sized vector will have n partial derivatives if the function is $f:\mathbb{R}^{n} \rightarrow \mathbb{R}$ (the derivative is known as the gradient). We can represent the derivative like this

$$
\frac{\partial y}{\partial\mathbf{x}} = 
\begin{bmatrix}
\frac{\partial y}{\partial{\mathbf{x}}_{1}}\\
\frac{\partial y}{\partial{\mathbf{x}}_{2}}\\
\vdots\\
\frac{\partial y}{\partial{\mathbf{x}}_{n}}\\
\end{bmatrix}
$$

With an understanding of the rules above, we can see $g:\mathbb{R} \rightarrow \mathbb{R}^{n}$ and  $f:\mathbb{R}^{n} \rightarrow \mathbb{R}$ with $\mathbf{y}=g(x)$ and $z=\mathbf{f(y)}$ then
$$
\frac{\partial z}{\partial x}=\sum_{i=1}^{n} \frac{\partial z}{\partial y_{i}} \frac{\partial y_{i}}{\partial x}
$$

### Simultaneous equation

Let us start by breaking our equation down, we break it down in these two equations.
$$\mathbf{r}= \mathbf{A}\mathbf{x}$$
$$c={\left\| \mathbf{r} -\mathbf{b} \right\|}^2_2$$
This gives a good break down of the real operation and the cost function. We are trying to optimize $\mathbf{x}$ such that $\mathbf{Ax}$ is as close to $\mathbf{b}$ as possible. Writing the full matrix form of what we have, we get this for the matrix equation
$$
\begin{bmatrix}
r_{1} \\
r_{2} \\
\vdots\\
r_{n}
\end{bmatrix}=
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n}\\
a_{21} & a_{22} & \dots & a_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
a_{n1} & a_{n2} & \dots & a_{nn}
\end{bmatrix}
\begin{bmatrix}
x_{1}\\
x_{2}\\
\vdots\\
x_{n}
\end{bmatrix}
$$
Rewriting this in terms of summations, we get these terms here
$$r_{j}=\sum_{i=1}^{n}{a_{ji}x_{i}}$$
For the cost equation, we get this
$$c=\sum_{j=1}^{n}{\left(r_{j}-b_{j}\right)^{2}}$$
Now, it is easy to calculate the partial derivatives
$$
\frac{\partial c}{\partial r_{j}}=2\left(r_{j}-b_{j}\right)
$$
$$
\frac{\partial r_{j}}{\partial x_{i}}=a_{ji}
$$
It should be easy to see the derivative in terms of summations, but because it will be useful for notation later, we will calculate the derivative first and write in terms of summation and then convert it to the vector form. The vector form also makes computation easier when we are calculating the derivatives with Numpy. The key is to calculate the derivative of $c$ with respect to the vector as opposed to a vector with respect to a vector or a vector with respect to a matrix, as this will save us from messy matrix calculus. For $\frac{\partial c}{\partial r_{j}}=2\left(r_{j}-b_{j}\right)$, we see that
$$
\frac{\partial c}{\partial \mathbf{r}}=2(\mathbf{r}-\mathbf{b})
$$
For $\frac{\partial c}{\partial x_{i}}$, we can compute like this
$$
\begin{align}
	\frac{\partial c}{\partial x_{i}}&=\sum_{j=1}^{n} \frac{\partial c}{\partial r_{j}} \frac{\partial r_{j}}{\partial x_{i}} \\
	&=\sum_{j=1}^{n} \frac{\partial c}{\partial r_{j}} a_{ji}
\end{align}
$$
From this, I think it is quite clear to see that
$$\frac{\partial c}{\partial \mathbf{x}}=\mathbf{A}^{T} \frac{\partial c}{\partial \mathbf{r}}$$
We see that by writing it like this computation can be done very easily as numpy (and every other BLAS library) can support matrix and vector operations clearly.

Finally, combining everything together, we get the three equations we need to solve any linear system with gradient descent.
$$
\begin{aligned}
F(\mathbf{x}) &= {||A\mathbf{x}-\mathbf{b}||}^{2} \\
\nabla F(\mathbf {x} ) &= 2 A^{T}(A\mathbf {x} -\mathbf{b}) \\
\mathbf{x}_{n+1} &= \mathbf{x}_{n}-\gamma \nabla F(\mathbf {x} _{n})
\end{aligned}
$$
where $\gamma$ is the learning rate, we need a small learning rate as it prevents the function from taking large steps and objective functions tend to overblow the "true" error of a function. 
We can now implement this in code form for a very simple linear system written below:

$$
\begin{aligned}
w+3x+2y-z=9\\
5w+2x+y-2z=4\\
x+2y+4z=24\\
w+x-y-3z=-12
\end{aligned}
$$

This can be written as such in matrix form:

$$
\begin{bmatrix}
1 & 3 & 2 & -1\\
5 & 2 & 1 & -2\\
0 & 1 & 2 & 4\\
1 & 1 & -1 & -3
\end{bmatrix}
\begin{bmatrix}
w\\
x\\
y\\
z
\end{bmatrix}
=
\begin{bmatrix}
9\\
4\\
24\\
-12
\end{bmatrix}
$$
### Code Implementation
#### Variables

$$
A=
\begin{bmatrix}
1 & 3 & 2 & -1\\
5 & 2 & 1 & -2\\
0 & 1 & 2 & 4\\
1 & 1 & -1 & -3
\end{bmatrix}
$$

```python
>>> A = np.array([[1,3,2,-1],[5,2,1,-2],[0,1,2,4],[1,1,-1,-3]], dtype=np.float64)
>>> A
array([[ 1.,  3.,  2., -1.],
       [ 5.,  2.,  1., -2.],
       [ 0.,  1.,  2.,  4.],
       [ 1.,  1., -1., -3.]])
```

$$
\mathbf{b}=
\begin{bmatrix}
9\\
4\\
24\\
-12
\end{bmatrix}
$$

```python
>>> b = np.array([[9],[4],[24],[-12]], dtype=np.float64) 
>>> b
array([[  9.],
       [  4.],
       [ 24.],
       [-12.]])
```

$$
\mathbf{x}=
\begin{bmatrix}
w\\
x\\
y\\
z
\end{bmatrix}
$$

```python
>>> x = np.random.rand(4,1)
>>> x
array([[0.09257854],
       [0.16847643],
       [0.39120624],
       [0.78484474]])
```

#### The Objective Function and its Derivative
$$
F(\mathbf{x}) = {||A\mathbf{x}-\mathbf{b}||}^{2}
$$
```python
>>> def objective_function(x):
        return np.linalg.norm(np.matmul(A,x) - b) ** 2
```
$$
\nabla F(\mathbf {x} )=2A^{T}(A\mathbf {x} -\mathbf {b})
$$
```python
>>> def objective_function_derivative(x):
        return 2 * np.matmul(A.T, np.matmul(A,x) - b)
```

In this case, I implemented an arbitrary learning rate and arbitrary step count. In traditional non-machine learning gradient descent, the learning rate changes per step and is determined via a heuristic such as the Barzilai–Borwein method, however this is not necessary as gradient descent is very robust. I used an arbitrary step count for simplicity but you should ideally use some sort of Boolean condition to break the loop such as $F(\mathbf{x})<0.01$.

$$
\mathbf {x}_{n+1}=\mathbf {x}_{n}-\gamma \nabla F(\mathbf {x} _{n})
$$

```python
>>> learning_rate = 0.01
>>> for i in range(5000):
        x -= learning_rate * objective_function_derivative(x)
>>> x
array([[1.],
       [2.],
       [3.],
       [4.]])
```

And to check, we now use a simple matrix multiplication:

```python
>>> np.matmul(A,x)
array([[  9.],
       [  4.],
       [ 24.],
       [-12.]])
```

Voila, we have solved the equation with gradient descent, and the solution is super close. This shows the power of gradient descent.
## Deep Neural Network Layer
Before we jump to full Convolutional Neural Networks, lets start with a simple deep neural network.
### Forward step
The deep neural network case is not so different from our original simple linear system example, we just have to calculate a few for more things. The essence of each neural network is layer is also multiplying the values of  a prior layer with a matrix and getting a new weight.
![multiple perceptron example|308](./images/multiple_perceptron_example.png)
The equation for a single inactivated layer is as follows
$$
\mathbf{z}=\mathbf{W}\mathbf{a}+\mathbf{b}
$$
$$
\begin{bmatrix}
z_{1} \\
z_{2} \\
\vdots\\
z_{m}
\end{bmatrix}=
\begin{bmatrix}
w_{11} & w_{12} & \dots & w_{1n}\\
w_{21} & w_{22} & \dots & w_{2n}\\
\vdots & \vdots & \ddots & \vdots\\
w_{m1} & w_{m2} & \dots & w_{mn}
\end{bmatrix}
\begin{bmatrix}
a_{1}\\
a_{2}\\
\vdots\\
a_{n}
\end{bmatrix}
+
\begin{bmatrix}
b_{1}\\
b_{2}\\
\vdots\\
b_{m}
\end{bmatrix}
$$
where $\mathbf{a}$ is the previous layers values, $\mathbf{W}$ is the weight matrix, $\mathbf{b}$ is the bias vector and $\mathbf{z}$ is the inactivated layer. We see that this part is not much different from the example earlier. There is also an element wise non polynomial activation function added afterwards. By and large, the Relu function is used here, but before that people used the sigmoid activation function. Either way, it is an element activation function and it is computed as follows
$$
\mathbf{a'}=\sigma\left(\mathbf{z}\right)
$$
$$
\begin{bmatrix}
a'_{1} \\
a'_{2} \\
\vdots\\
a'_{m}
\end{bmatrix}=
\begin{bmatrix}
\sigma \left(a_{1}\right)\\
\sigma \left(a_{2}\right)\\
\vdots\\
\sigma \left(a_{n}\right)
\end{bmatrix}
$$
This gives us the full neural network layer. We can also use the same mean square error,as the loss function, thus we can now right the full neural network in terms of vectors and matrices
$$\mathbf{a}^{(0)}=\mathbf{x}$$
$$\mathbf{z}^{(l)}=\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)}=\sigma \left(\mathbf{z}^{(l)} \right)$$
where $l=1,2,3,\dots,L$ for the different layers from the neural network and $\mathbf{x}$ is the input, everything else is layer specific. For the final layer $\mathbf{a}^{(L)}$, we can compare it against the true output $\mathbf{y}$ to get the cost $c$
$$\mathbf{c}=\frac{1}{n}{\left\| \mathbf{a}^{(L)} -\mathbf{y} \right\|}^2_2$$
We can now write out all the equations with indices
$$z^{(l)}_{i}=b^{(l)}_i+\sum_{i=1}^{n}w^{(l)}_{ij}a^{(l-1)}_{j}$$
$$a^{(l)}_{i}=\sigma \left(z^{(l)}_{i}\right)$$
$$c=\frac{1}{n}\sum_{i=1}^{n}{\left(a^{(L)}_{i}-y_i\right)^{2}}$$
This makes it much easier for us to calculate the original derivatives. First we calculate the derivative of the cost with respect to the previous layer, same as earlier this is
$$
\frac{\partial c}{\partial a^{(L)}_{i}}=\frac{2}{n}\left(a^{(L)}_{i}-y_{i}\right)
$$

## Neural Network Implementation (XNOR Gate)

I couldn't find a good, but rather small dataset because most people really do like large datasets and are infuriated when they are not provided that like ~~entitled brats~~ normal people. So, instead, I decided that we will train our neural network to mimic the XNOR gate.

Oh no! Training? Testing? What is that? In all fairness, I am simply trying to show you that the mathematical functions that dictate neural networks as we have found above, fits perfectly with this task of a neural network, and that these neural networks that everyone hears about can really just mimic any function.

![XNOR input output](./images/XNOR_input_output.png)

For those who do not know, the XNOR gates inputs and outputs are written above. It is pretty suitable for this example, because the inputs and outputs are all 0 and 1, hence it is fast to train and there is no bias in the data.

From here, let's try coding out the (x,y) pairs in NumPy:

```python
data = [[np.array([[0],[0]], dtype=np.float64),np.array([[1]], dtype=np.float64)],
        [np.array([[0],[1]], dtype=np.float64),np.array([[0]], dtype=np.float64)],
        [np.array([[1],[0]], dtype=np.float64),np.array([[0]], dtype=np.float64)],
        [np.array([[1],[1]], dtype=np.float64),np.array([[1]], dtype=np.float64)]]
```

We then define a network structure. It doesn't have to be too complex because it is a pretty simple function. I decided on a $2 \rightarrow 3 \rightarrow 1$ multi-layer perceptron (MLP) structure, with the sigmoid activation function.

![multiple perceptron network](./images/multiple_perceptron_network.png)

Next, let's try coding out our mathematical work based off the following class:

```python
class NNdata:
    def __init__(self):
        self.a_0 = None
        self.W_0 = np.random.rand(3,2)
        self.b_0 = np.random.rand(3,1)
        self.z_1 = None
        self.a_1 = None
        self.W_1 = np.random.rand(1,3)
        self.b_1 = np.random.rand(1,1)
        self.z_2 = None
        self.a_2 = None
        self.db_1 = None
        self.dw_1 = None
        self.db_0 = None
        self.dw_0 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def feed_forward(self, x):
        self.a_0 = x

        self.z_1 = np.matmul(self.W_0, self.a_0)+self.b_0
        self.a_1 = self.sigmoid(self.z_1)

        self.z_2 = np.matmul(self.W_1, self.a_1)+self.b_1
        self.a_2 = self.sigmoid(self.z_2)
        return self.a_2

    def loss(self, y):
        return np.linalg.norm(self.a_2-y)**2

    def back_prop(self, y):
        dcdz_2 = 2 * np.matmul((self.a_2-y).T,np.diag(self.sigmoid_derivative(self.z_2).reshape(1)))
        dcdb_1 = dcdz_2
        dcdw_1 = np.matmul(self.a_1, dcdz_2)

        dcda_1 = np.matmul(dcdz_2, self.W_1)
        dcdz_1 = np.matmul(dcda_1, np.diag(self.sigmoid_derivative(self.z_1).reshape(3)))
        dcdb_0 = dcdz_1
        dcdw_0 = np.matmul(self.a_0, dcdz_1)

        self.db_1 = dcdb_1.T
        self.dw_1 = dcdw_1.T
        self.db_0 = dcdb_0.T
        self.dw_0 = dcdw_0.T
```

Next I program gradient descent. There are 3 kinds of gradient descent when there are multiple datapoints, Stochastic, Batch and Mini-Batch. In Stochastic Gradient Descent (SGD), the weights are updated after a single sample is run. This will obviously cause your step towards the ideal value be very chaotic. In Batch Gradient Descent, the weights are updated after every sample is run, and the net step is the sum/average of all the $\nabla F(x)$, which is less chaotic, but steps are less frequent.

Of course, in real life, we can never know which algorithm is better without making an assumption about the data. (No Free Lunch Theorem) A good compromise is Mini-Batch Gradient Descent, which is like Batch Gradient Descent but use smaller chunks of all the datapoints every step. In this case, I use Batch Gradient Descent.

```python
nndata = NNdata()
learning_rate = 0.1
for i in range(10000):
    db_1_batch = []
    dw_1_batch = []
    db_0_batch = []
    dw_0_batch = []
    c = []
    for j in range(4):
        nndata.feed_forward(data[j][0])
        c.append(nndata.loss(data[j][1]))
        nndata.back_prop(data[j][1])
        db_1_batch.append(nndata.db_1)
        dw_1_batch.append(nndata.dw_1)
        db_0_batch.append(nndata.db_0)
        dw_0_batch.append(nndata.dw_0)
    if((i+1) % 1000 == 0):
        print("loss (%d/10000): %.3f" % (i+1, sum(c)/4))
    nndata.b_1 -= learning_rate * sum(db_1_batch)
    nndata.W_1 -= learning_rate * sum(dw_1_batch)
    nndata.b_0 -= learning_rate * sum(db_0_batch)
    nndata.W_0 -= learning_rate * sum(dw_0_batch)
```

Output resource:

```
loss (1000/10000): 0.245
loss (2000/10000): 0.186
loss (3000/10000): 0.029
loss (4000/10000): 0.007
loss (5000/10000): 0.003
loss (6000/10000): 0.002
loss (7000/10000): 0.002
loss (8000/10000): 0.001
loss (9000/10000): 0.001
loss (10000/10000): 0.001
```

Voila! We have officially programmed Neural Networks from scratch. Pat yourself on the back for reading through this. And of course, if you bothered to code this out, try porting it over to different languages like Java, JS or even C (yikes why would [anyone](https://github.com/terminalai/neuralC) subjects themselves to that?).

In the next part, it is time for the actual hard part. Good luck!

## References

A lot of people think I just collated a bunch of sources and rephrased, and honestly I walked into writing this thinking I would be doing just that. The problem is that many sources who have attempted to do this, only cover the single to multi-perceptron layer case and not the multi to multi-perceptron case. Which is pretty sad. The true math is hidden behind mountains of research papers that loosely connect to give the results of this blogpot which I am too incomponent to connect by myself. So, I just did the math myself. (The math may not be presented in this way but it works so it should be correct) Yes, it was a bit crazy, and it destroyed me to my core. This was a great character building moment for me. So these are the actual sources:

- https://numpy.org/
- https://en.wikipedia.org/wiki/Gradient_descent
- https://en.wikipedia.org/wiki/Matrix_calculus
- https://en.wikipedia.org/wiki/Tensor_calculus
- https://en.wikipedia.org/wiki/Ricci_calculus
- https://en.wikipedia.org/wiki/XNOR_gate
- CS5131 Notes (Special thanks to Mr Chua and Mr Ng)

## Addendum

TO BE DONE

<hr>

(Excruciatingly edited by Prannaya)
