---
layout: post
title: "Part 2: Welcome of Multivariate Guassian Distribution"
description: "Going higher in higher dimension- with backgrounds"
subject: "Probabilistic Machine Learning"
---

This is based from the book: Pattern Recognition and Machine Learning and partly from <a href="http://cs229.stanford.edu/section/gaussians.pdf" target="_blank">CS229 notes on Guassian</a>, where we aim to give and expands most proofs and interesting results regarding multivariate Gaussian distributions.

This second part is to introduce some of the mathematical basics such as linear algebra. Further results of linear Gaussian models and others will be presented in the next part. 

## Useful Backgrounds

### Covariance and Covariance Matrix

**Definition (Covariance)**: Given 2 random variables $X$ and $Y$, the covariance is defined as:
<div>
\begin{equation*}
\newcommand{\dby}{\ \mathrm{d}}\newcommand{\argmax}[1]{\underset{#1}{\arg\max \ }}\newcommand{\argmin}[1]{\underset{#1}{\arg\min \ }}\newcommand{\const}{\text{const.}}\newcommand{\bracka}[1]{\left( #1 \right)}\newcommand{\brackb}[1]{\left[ #1 \right]}\newcommand{\brackc}[1]{\left\{ #1 \right\}}\newcommand{\brackd}[1]{\left\langle #1 \right\rangle}\newcommand{\correctquote}[1]{``#1''}\newcommand{\norm}[1]{\left\lVert#1\right\rVert}\newcommand{\abs}[1]{\left|#1\right|}
\operatorname{cov}(X, Y) = \mathbb{E}\Big[ (X - \mathbb{E}[X])(Y - \mathbb{E}[Y]) \Big]
\end{equation*}
</div>
It is clear that $\operatorname{cov}(X, Y) = \operatorname{cov}(Y, X) $

**Definition (Covariance Matrix)**: Now, if we have the collection of random variables in the random vector $\boldsymbol x$ (of size $n$), then the collection of covariance between its elements are collected in covariance matrix
<div>
\begin{equation*}
\operatorname{cov}(\boldsymbol x) = 
\begin{bmatrix}
    \operatorname{cov}(x_1, x_1) & \operatorname{cov}(x_1, x_2) & \cdots & \operatorname{cov}(x_1, x_n)  \\
    \operatorname{cov}(x_2, x_1) & \operatorname{cov}(x_2, x_2) & \cdots & \operatorname{cov}(x_2, x_n)  \\
    \vdots & \vdots & \ddots & \vdots \\
    \operatorname{cov}(x_n, x_1) & \operatorname{cov}(x_n, x_2) & \cdots & \operatorname{cov}(x_n, x_n)  \\
\end{bmatrix} = \mathbb{E}\Big[ (\boldsymbol x - \mathbb{E}[\boldsymbol x])(\boldsymbol x - \mathbb{E}[\boldsymbol x])^T \Big]
\end{equation*}
</div>
The equivalent is clear when we perform the matrix multiplication over this. 

**Remark (Property of Covariance Matrix)**: It is clear that the covariance matrix is (from its defintion):

- _Symmetric_, as $\operatorname{cov}(x_a, x_b) = \operatorname{cov}(x_b, x_a)$
- _Positive semidefinite_, if we consider arbitary constant vector $\boldsymbol a \in \mathbb{R}^n$, then by the linearity of expectation, we have:
<div>
\begin{equation*}
\begin{aligned}
\boldsymbol a^T \operatorname{cov}(\boldsymbol x)\boldsymbol a 
&= \boldsymbol a^T \mathbb{E}\Big[ (\boldsymbol x - \mathbb{E}[\boldsymbol x])(\boldsymbol x - \mathbb{E}[\boldsymbol x])^T \Big]\boldsymbol a  \\
&= \mathbb{E}\Big[ \boldsymbol a^T (\boldsymbol x - \mathbb{E}[\boldsymbol x])(\boldsymbol x - \mathbb{E}[\boldsymbol x])^T\boldsymbol a \Big]  \\
&= \mathbb{E}\Big[ \big(\boldsymbol a^T (\boldsymbol x - \mathbb{E}[\boldsymbol x])\big)^2 \Big] > 0 \\
\end{aligned}
\end{equation*}
</div>

### Linear Algebra

**Definition (Eigenvalues and Eigenvectors)**: Given the matrix $\boldsymbol X \in \mathbb{C}^{n\times n}$, then the pair of vector $\boldsymbol v \in \mathbb{C}^n$ and number $\lambda \in \mathbb{C}$ are called eigenvector and eigenvalue iff
<div>
\begin{equation*}
\begin{aligned}
\boldsymbol A \boldsymbol v = \lambda\boldsymbol v
\end{aligned}
\end{equation*}
</div>

**Proposition (Eigenvalue of Symmetric Matrix)**: One can show that the eigenvalue of symmetric (real) matrix is always real. 

*Proof (From <a href="http://pi.math.cornell.edu/~jerison/math2940/real-eigenvalues.pdf" target="_blank">here</a>)*: Let's consider the pairs of eigenvalues/eigenvectors $\boldsymbol v \in \mathbb{C}^n$ and $\lambda \in \mathbb{C}$ of symmetric (real) matrix $\boldsymbol A$ i.e $\boldsymbol A\boldsymbol v = \lambda\boldsymbol v$. Please note that $(x+yi)(x-yi) = x^2 + y^2 \ge 0$, this means that $\bar{\boldsymbol v}^T\boldsymbol v \ge 0$ ($\bar{\boldsymbol v}$ is vector that contains conjugate element of $\boldsymbol v$). Now, see that:
<div>
\begin{equation*}
\begin{aligned}
\bar{\boldsymbol v}^T\boldsymbol A\boldsymbol v 
&= \bar{\boldsymbol v}^T(\boldsymbol A\boldsymbol v) = \lambda\bar{\boldsymbol v}^T\boldsymbol v  \\
&= (\bar{\boldsymbol v}^T\boldsymbol A^T)\boldsymbol v = \bar{\lambda}\bar{\boldsymbol v}^T\boldsymbol v  \\
\end{aligned}
\end{equation*}
</div>
Note that $\overline{\boldsymbol A\boldsymbol v} = \bar{\lambda}\bar{\boldsymbol v}$
Since $\bar{\boldsymbol v}^T\boldsymbol v > 0$ we see that $\bar{\lambda} = \lambda$, which implies that $\bar{\lambda}$ is real. 
$\tag*{$\Box$}$


**Proposition (Eigenvalue of Positive Definite Matrix)**: One can show that the eigenvalue of positive definite matrix is non-negative. 

*Proof*: We consider the following equations:
<div>
\begin{equation*}
\begin{aligned}
\boldsymbol v^T\boldsymbol A\boldsymbol v = \lambda\boldsymbol v^T\boldsymbol v > 0
\end{aligned}
\end{equation*}
</div>
And so the eigenvector must be $\lambda > 0$.
$\tag*{$\Box$}$

**Definition (Linear Transformation)**: Given the function $A: V \rightarrow W$, where $V$ and $W$ are vector spaces. Then, for $\boldsymbol v \in V, \boldsymbol w \in W$ and $a, b \in \mathbb{R}$
<div>
\begin{equation*}
\begin{aligned}
A(a\boldsymbol v + b\boldsymbol w) = aA(\boldsymbol v) + bA(\boldsymbol w)
\end{aligned}
\end{equation*}
</div>

**Remark (Matrix Multipliacation and Linear Transformation)**: We can represent the linear transformation $L : V \rightarrow W$ in terms of matrix. Let's consider the vector $\boldsymbol v \in V$ (of dimension $n$) together with basis vectors $\brackc{\boldsymbol b_1,\dots,\boldsymbol b_n}$ of $V$ and basis vectors $\brackc{\boldsymbol c_1, \dots, \boldsymbol c_m}$ of $W$. Then we can represen the vector $\boldsymbol v$ as:
<div>
\begin{equation*}
\begin{aligned}
\boldsymbol v = v_1\boldsymbol b_1 + \dots + v_n\boldsymbol b_n
\end{aligned}
\end{equation*}
</div>
This means that we can represent the vector $\boldsymbol v$ as: $(\boldsymbol v_1, \boldsymbol v_2, \dots, \boldsymbol v_n)^T$
Furthermore, we can characterized the transformation of the basis vector:
<div>
\begin{equation*}
\begin{aligned}
    &L(\boldsymbol b_i) = l_{1i} \boldsymbol c_1 + l_{2i}\boldsymbol c_2 + \cdots + l_{mi}\boldsymbol c_m \\
\end{aligned}
\end{equation*}
</div>
for $i=1,\dots,n$. Then we can see that the definition of linear transformation together with the  linear transformation of basis as we have:
<div>
\begin{equation*}
\begin{aligned}
    L(\boldsymbol v) &= v_1L(\boldsymbol b_1) + \dots + v_nL(\boldsymbol v_n) \\
    &= \begin{aligned}[t]
        &v_1\Big( l_{11} \boldsymbol c_1 + l_{21}\boldsymbol c_2 + \cdots + l_{m1}\boldsymbol c_m \Big) \\
        &+v_2\Big( l_{12} \boldsymbol c_1 + l_{22}\boldsymbol c_2 + \cdots + l_{m2}\boldsymbol c_m \Big) \\
        &+v_n\Big( l_{1n} \boldsymbol c_1 + l_{2n}\boldsymbol c_2 + \cdots + l_{mn}\boldsymbol c_m \Big)\\
    \end{aligned} \\
    &= \begin{bmatrix}
        \sum^n_{i=1}v_il_{1i} \\ \sum^n_{i=1}v_il_{2i} \\ \vdots \\ \sum^n_{i=1}v_il_{ni} 
    \end{bmatrix} = \begin{bmatrix}
        l_{11} & l_{12} & \cdots & l_{1n} \\
        l_{21} & l_{22} & \cdots & l_{2n} \\
        \vdots & \vdots & \ddots & \vdots \\
        l_{m1} & l_{m2} & \cdots & l_{mn} \\
    \end{bmatrix}
    \begin{bmatrix}
        v_1 \\ v_2 \\ \vdots \\ v_n
    \end{bmatrix}
\end{aligned}
\end{equation*}
</div>
and so we have show that the linear transformation can be represented as matrix multiplication (in finite space).


**Theorem (Spectral Theorem)** (follows from <a href="https://mast.queensu.ca/~br66/419/spectraltheoremproof.pdf" target="_blank">here</a>): Let $n \le N$ and $W$ be an n-dimensional subspace of $\mathbb{R}^n$. Given a linear transformation $A:W\rightarrow W$ that is symmetric. There are eigenvectors $\boldsymbol v_1,\dots,\boldsymbol v_n\in W$ of $A$ such that $\brackc{\boldsymbol v_1,\dots,\boldsymbol v_n}$ is an orthonormal basis for $W$. For normal matrix, we let $n=N$ and $W=\mathbb{R}^n$. 

**Remark (Eigendecomposition)**: Let's consider the matrix of eigenvectors $\boldsymbol v_1,\boldsymbol v_2\dots,\boldsymbol v_n \in \mathbb{R}^n$ of symmetric matrix $\boldsymbol A \in \mathbb{R}^{n\times n}$ together with eigenvalues $\lambda_1,\lambda_2,\dots,\lambda_n$, then we have :
<div>
\begin{equation*}
\begin{aligned}
    \boldsymbol A
    \begin{bmatrix}
        \kern.6em\vline & \kern.2em\vline\kern.2em & & \vline\kern.6em \\
        \kern.6em\boldsymbol v_1 & \kern.2em\boldsymbol v_2\kern.2em & \kern.2em\cdots\kern.2em &  \boldsymbol v_n\kern.6em  \\
        \kern.6em\vline & \kern.2em\vline\kern.2em & & \vline\kern.6em \\ 
    \end{bmatrix} &= 
    \begin{bmatrix}
        \kern.6em\vline & \kern.2em\vline\kern.2em & & \vline\kern.6em \\
        \kern.6em\boldsymbol v_1 & \kern.2em\boldsymbol v_2\kern.2em & \kern.2em\cdots\kern.2em &  \boldsymbol v_n\kern.6em  \\
        \kern.6em\vline & \kern.2em\vline\kern.2em & & \vline\kern.6em \\
    \end{bmatrix}\begin{bmatrix}
        \lambda_1 & 0 & \cdots & 0 \\
        0 & \lambda_2 & \cdots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \cdots & \lambda_n \\
    \end{bmatrix} \\
\end{aligned}
\end{equation*}
</div>
This is equivalent to $\boldsymbol A\boldsymbol Q = \boldsymbol Q\boldsymbol \Lambda$, which mean that if we right multiply by $\boldsymbol Q^T$ (as we have *orthogonal* eigenvectors), then we have (or in vectorized format):
<div>
\begin{equation*}
\boldsymbol A = \boldsymbol Q\boldsymbol \Lambda\boldsymbol Q^T = \sum^n_{i=1}\lambda_i\boldsymbol v_i\boldsymbol v_i^T
\end{equation*}
</div>
Please note that $\boldsymbol A^{-1}$ can be represented as:
<div>
\begin{equation*}
\boldsymbol A^{-1} = \boldsymbol Q\boldsymbol \Lambda^{-1}\boldsymbol Q^T = \sum^n_{i=1}\frac{1}{\lambda_i}\boldsymbol v_i\boldsymbol v_i^T
\end{equation*}
</div>
as it is clear that $\boldsymbol A\boldsymbol A^{-1} = \boldsymbol Q\boldsymbol \Lambda\boldsymbol Q^T\boldsymbol Q\boldsymbol \Lambda^{-1}\boldsymbol Q^T = \boldsymbol I$. 

**Proposition (Determinant and Eigenvalues)**: Give matrix $\boldsymbol A$ together with eigenvalues of $\lambda_1,\lambda_2,\dots,\lambda_n$, then we can show that
<div>
\begin{equation*}
    \abs{\boldsymbol A} = \prod^n_{i=1}\lambda_i
\end{equation*}
</div>

*Proof (Diagonalizable Matrix)*: We consider the eigendecomposition of $\boldsymbol A$ (if it exists, which is most of the cases here. For more general proof, see linear algebra notes), as we have:
<div>
\begin{equation*}
    \abs{\boldsymbol A} = \abs{\boldsymbol Q\boldsymbol \Lambda\boldsymbol Q^{-1}} = \abs{\boldsymbol Q}\abs{\boldsymbol \Lambda}\abs{\boldsymbol Q^{-1}} = \frac{\abs{\boldsymbol Q}}{\abs{\boldsymbol Q}}\abs{\boldsymbol \Lambda} =  \prod^n_{i=1}\lambda_i
\end{equation*}
</div>
$\tag*{$\Box$}$

**Proposition (Trace and Eigenvalues)**: Given a matrix $\boldsymbol A$ together with eigenvalues of $\lambda_1,\lambda_2,\dots,\lambda_n$, then we can show that:
<div>
\begin{equation*}
    \operatorname{Tr}(\boldsymbol A) = \sum^n_{i=1}\lambda_i
\end{equation*}
</div>
*Proof (Diagonalizable Matrix)*: We consider the eigendecomposition of $\boldsymbol A$. Consider the trace over it, as we have:
<div>
\begin{equation*}
    \operatorname{Tr}(\boldsymbol A) = \operatorname{Tr}(\boldsymbol Q\boldsymbol \Lambda\boldsymbol Q^{-1}) = \operatorname{Tr}(\boldsymbol \Lambda\boldsymbol Q\boldsymbol Q^{-1}) = \operatorname{Tr}(\boldsymbol \Lambda) = \sum^n_{i=1}\lambda_i
\end{equation*}
</div>




### Miscellaneous

**Proposition (Change of Variable)**: If we consider the transformation of the variables where $T : \mathbb{R}^k \supset X \rightarrow \mathbb{R}^k$. Then we can show that:
<div>
\begin{equation*}
\int_{\mathbb{R}^k} f(\boldsymbol y)\dby \boldsymbol y = \int_{\mathbb{R}^k} f(\boldsymbol T(\boldsymbol x))\abs{\boldsymbol J_T(\boldsymbol x)}\dby \boldsymbol x
\end{equation*}
</div>
where $\boldsymbol J_T(\boldsymbol x)$ is the Jacobian of the transformation $T(\cdot)$, which is defined to be:
<div>
\begin{equation*}
\boldsymbol J_T(\boldsymbol x) = \begin{pmatrix}
    \cfrac{\partial T_1(\cdot)}{\partial x_1} & \cfrac{\partial T_1(\cdot)}{\partial x_2}  & \cdots & \cfrac{\partial T_1(\cdot)}{\partial x_n} \\ 
    \cfrac{\partial T_2(\cdot)}{\partial x_1} & \cfrac{\partial T_2(\cdot)}{\partial x_2}  & \cdots & \cfrac{\partial T_2(\cdot)}{\partial x_n} \\ 
    \vdots & \vdots & \ddots & \vdots \\
    \cfrac{\partial T_n(\cdot)}{\partial x_1} & \cfrac{\partial T_n(\cdot)}{\partial x_2}  & \cdots & \cfrac{\partial T_n(\cdot)}{\partial x_n} \\ 
\end{pmatrix}
\end{equation*}
</div>


<!-- *Proof*: Let's consider the induction:

- Starting with $n=1$, the result is obvious.
- For $n>1$, We consider the problem:
<div>
\begin{equation*}
\begin{aligned}
  \max & \quad (\boldsymbol A\boldsymbol w)^T\boldsymbol w \\
  \text{ subject to } & \quad \boldsymbol w^T\boldsymbol w = 1
\end{aligned}
\end{equation*}
</div>
Note that $(\boldsymbol A\boldsymbol w)^T\boldsymbol w = \boldsymbol w^T(\boldsymbol A\boldsymbol w)$
Let's try to solve this constraint problem, as we have the Lagragian (together with its derivative with respected to $\boldsymbol w$) to be:
<div>
\begin{equation*}
\begin{aligned}
&\mathcal{L}(\boldsymbol w, \lambda) = \boldsymbol w^T\boldsymbol A\boldsymbol w - \lambda\Big(  \boldsymbol w^T\boldsymbol w -1 \Big) \\
\iff&\begin{aligned}[t]
    \frac{\partial}{\partial \boldsymbol w}\mathcal{L}(\boldsymbol w, \lambda) &= \frac{\partial}{\partial \boldsymbol w} \boldsymbol w^T\boldsymbol A\boldsymbol w - \lambda\Big(  \boldsymbol w^T\boldsymbol w -1 \Big) \\
    &= 2\boldsymbol A\boldsymbol w - 2\lambda \boldsymbol w
\end{aligned}
\end{aligned}
\end{equation*}
</div>
This means that: $\boldsymbol A\boldsymbol w^* = \lambda\boldsymbol w^*$ meaing that $\boldsymbol w^*$ is the eigenvector of $\boldsymbol A$ with real eigenvalue $\lambda$. If we consider the orthogonal complement of $\boldsymbol w^*$ i.e $W' = \operatorname{span}(\boldsymbol w^*)^{\perp}$. Then, we see that for a vector $\boldsymbol w' \in W'$, we have
<div>
\begin{equation*}
    0 = (\boldsymbol w')^T (\lambda\boldsymbol w^*) = (\boldsymbol w')^T (\boldsymbol A\boldsymbol w^*) = (\boldsymbol A\boldsymbol w')^T\boldsymbol w^*
\end{equation*}
</div>
Note that we use the matrix form here. This means that $\boldsymbol A\boldsymbol w' \in W'$ (for $A : W' \rightarrow W'$). Using induction hypothesis ($n-1$ subspace has the eigenbasis to be $\brackc{\boldsymbol u_1,\dots,\boldsymbol u_{n-1}}$), we have $\brackc{\boldsymbol u_1,\dots,\boldsymbol u_{n-1}, \boldsymbol u_n}$ to be an orthogonal basis of $W$, where we note that $\operatorname{span}(\boldsymbol w^*)^{\perp} \oplus \operatorname{span}(\boldsymbol w^*)= W$ -->


## Multivariate Guassian Distribution: Introduction

**Definition (Multivariate Gaussian)**: It is defined as:
<div>
\begin{equation*}
\mathcal{N}(\boldsymbol x | \boldsymbol \mu, \boldsymbol \Sigma) = \frac{1}{\sqrt{\abs{2\pi\boldsymbol \Sigma}}}\exp\left\{-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu)\right\}
\end{equation*}
</div>
where we call $\boldsymbol \mu \in \mathbb{R}^n$ a mean and $\boldsymbol \Sigma \in \mathbb{R}^{n\times n}$ covariance, which should be symmetric and positive semidefinite (since the covariance is always positive semidefinite and symmetric). 

**Remark (2D Independent Gaussian)**: Now, let's consider, multivariate Gaussian but in the case that both variables are independent to each other with difference variances, as we define the parameters to be:
<div>
\begin{equation*}
\boldsymbol \mu = \begin{bmatrix}
    \mu_1 \\ \mu_2
\end{bmatrix} \qquad \boldsymbol \Sigma = 
\begin{bmatrix}
    \sigma_1^2 & 0 \\
    0 & \sigma_2^2 \\
\end{bmatrix}
\end{equation*}
</div>
Now, let's expand the multivariate Guassian, please note that $\abs{\boldsymbol \Sigma} = \sigma_1^2\sigma_2^2$:
<div>
\begin{equation*}
\begin{aligned}
\mathcal{N}(\boldsymbol x | \boldsymbol \mu, \boldsymbol \Sigma) &= \frac{1}{\sqrt{\abs{2\pi\boldsymbol \Sigma}}}\exp\left\{-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu)\right\} \\
&= \frac{1}{4\pi^2\sigma_1^2\sigma_2^2} \exp\brackc{-\frac{1}{2} \begin{bmatrix} x_1-\mu_1 \\ x_2-\mu_2 \end{bmatrix}^T
\begin{bmatrix}
    \sigma_1^2 & 0 \\
    0 & \sigma_2^2 \\
\end{bmatrix}
\begin{bmatrix} x_1-\mu_1 \\ x_2-\mu_2 \end{bmatrix}} \\
&= \frac{1}{4\pi^2\sigma_1^2\sigma_2^2} \exp\brackc{-\frac{1}{2} \brackb{ \bracka{\frac{x_1-\mu_1}{\sigma_1}}^2 + \bracka{\frac{x_2-\mu_2}{\sigma_2}}^2 }}\\
&= \frac{1}{2\pi\sigma_1^2} \exp\brackc{-\frac{1}{2} \frac{(x_1-\mu_1)^2}{\sigma_1^2}} \frac{1}{2\pi\sigma_2^2} \exp\brackc{-\frac{1}{2} \frac{(x_2-\mu_2)^2}{\sigma_2^2}}\\
&= \mathcal{N}(x_1 | \mu_1, \sigma_1^2)\mathcal{N}(x_2 | \mu_2, \sigma_2^2)
\end{aligned}
\end{equation*}
</div>

**Remark (Shape of Gaussian)**: Let's consider the eigendecomposition of the inverse covariance matrix (which is positive semidefinite and symmetric), as we have
<div>
\begin{equation*}
\begin{aligned}
\boldsymbol \Sigma^{-1} = \sum^n_{i=1}\frac{1}{\lambda_i}\boldsymbol u_i\boldsymbol u_i^T
\end{aligned}
\end{equation*}
</div>
where $\lambda_1,\dots,\lambda_n$ and $\boldsymbol u_1,\dots,\boldsymbol u_n$ are the eigenvalues and eigenvectors, repectively of $\boldsymbol \Sigma$. Let's consider the terms inside the exponential to be:
<div>
\begin{equation*}
\begin{aligned}
(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x - \boldsymbol \mu) 
= \sum^n_{i=1}\frac{(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol u_i\boldsymbol u_i^T(\boldsymbol x - \boldsymbol \mu)}{\lambda_i}
= \sum^n_{i=1}\frac{y_i^2}{\lambda_i}
\end{aligned}
\end{equation*}
</div>
where we have $y_i = (\boldsymbol x - \boldsymbol \mu)^T\boldsymbol u_i$. Consider the vector to be $\boldsymbol y = (y_1,y_2,\dots,y_n)^T = \boldsymbol U(\boldsymbol x - \boldsymbol \mu)$. This gives us the linear transformation over $\boldsymbol x$, which implies the following shape of Gaussian:
- Ellipsoids with the center $\boldsymbol \mu$
- Axis is in the direction of eigenvector $\boldsymbol u_i$
- Scaling of each direction is the eigenvector $\lambda_i$ associated with $\boldsymbol u_i$

**Proposition (Normalization of Gaussian)**: We can show that:
<div>
\begin{equation*}
\begin{aligned}
\int \exp\left\{-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu)\right\} \dby \boldsymbol x = \sqrt{\abs{2\pi\boldsymbol \Sigma}}
\end{aligned}
\end{equation*}
</div>

*Proof*: Let's consider the change of variable, in which we will change the variable from $x_i$ to $y_i$ where $\boldsymbol y = \boldsymbol U(\boldsymbol x - \boldsymbol \mu)$. To do this we have the find the Jacobian of the transformation, which is:
<div>
\begin{equation*}
\begin{aligned}
J_{ij} = \frac{\partial x_i}{\partial y_j} = U_{ji}
\end{aligned}
\end{equation*}
</div>
Consider its determinant, as we have: $\abs{\boldsymbol J}^2 = \abs{\boldsymbol U^T}^2 = \abs{\boldsymbol U^T}\abs{\boldsymbol U} = \abs{\boldsymbol U^T\boldsymbol U} = \abs{I} = 1$. Consider the integration as we have (and use the Gaussian integrations):
<div>
\begin{equation*}
\begin{aligned}
\int \exp\left\{-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu)\right\} \dby \boldsymbol x 
&= \int \exp\brackc{\sum^n_{i=1}-\frac{y_i^2}{2\lambda_i}} |\boldsymbol J| \dby \boldsymbol y \\
&= \int \prod^n_{i=1}\exp\brackc{-\frac{y_i^2}{2\lambda_i}} \dby \boldsymbol y \\
&= \prod^n_{i=1}\int \exp\brackc{-\frac{y_i^2}{2\lambda_i}} \dby y_i \\
&= \prod^n_{i=1}\sqrt{2\pi\lambda_i} = \sqrt{\abs{2\pi\boldsymbol \Sigma}}
\end{aligned} 
\end{equation*}
</div>
Note that we have shown that the determinant is the product of eigenvalues. Thus the prove is completed. 
$\tag*{$\Box$}$
