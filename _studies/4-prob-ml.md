---
layout: post
title: "Part 4: The First model- MLE/MAP x Gaussian x Linear Regression"
description: "One Small Step for Man"
subject: "Probabilistic Machine Learning"
---

**TODO: MAP estimate of Normal Wishart**

This post is based from the book: Pattern Recognition and Machine Learning, where we consider the first probabilisitic model, which is linear regression, we will consider: Maximum Likelihood Estimation (MLE), Baysian/MAP methods, and Type II MLE.

## Background on Matrix Calculus

**Remark (Simply Derivatives)**: We now consider these simple derivatives for vector (the can be proven simply by perform derivative in each dimensions):
<div>
\begin{equation*}
\require{color}
\newcommand{\dby}{\ \mathrm{d}}\newcommand{\argmax}[1]{\underset{#1}{\arg\max \ }}\newcommand{\argmin}[1]{\underset{#1}{\arg\min \ }}\newcommand{\const}{\text{const.}}\newcommand{\bracka}[1]{\left( #1 \right)}\newcommand{\brackb}[1]{\left[ #1 \right]}\newcommand{\brackc}[1]{\left\{ #1 \right\}}\newcommand{\brackd}[1]{\left\langle #1 \right\rangle}\newcommand{\correctquote}[1]{``#1''}\newcommand{\norm}[1]{\left\lVert#1\right\rVert}\newcommand{\abs}[1]{\left|#1\right|}
\definecolor{red}{RGB}{244, 67, 54}
\definecolor{pink}{RGB}{233, 30, 99}
\definecolor{purple}{RGB}{103, 58, 183}
\definecolor{yellow}{RGB}{255, 193, 7}
\definecolor{grey}{RGB}{96, 125, 139}
\definecolor{blue}{RGB}{33, 150, 243}
\definecolor{green}{RGB}{0, 150, 136}
\frac{\partial}{\partial \boldsymbol x}\boldsymbol x^T\boldsymbol a = \frac{\partial}{\partial \boldsymbol x}\boldsymbol a^T\boldsymbol x = \boldsymbol a \qquad \frac{\partial}{\partial x}\boldsymbol A\boldsymbol B = \frac{\partial \boldsymbol A}{\partial x}\boldsymbol B + \frac{\partial \boldsymbol B}{\partial x}\boldsymbol A \qquad \frac{\partial}{\partial \boldsymbol x}\boldsymbol x^T\boldsymbol A\boldsymbol x = (\boldsymbol A + \boldsymbol A^T)\boldsymbol x
\end{equation*}
</div>

**Remark (More Matrix Derivative)**: 
<div>
\begin{equation*}
\frac{\partial}{\partial \boldsymbol A}\operatorname{Tr}(\boldsymbol A\boldsymbol B) = \boldsymbol B^T \qquad \frac{\partial}{\partial \boldsymbol A}\operatorname{Tr}(\boldsymbol A^T\boldsymbol B) = \boldsymbol A \qquad \frac{\partial}{\partial \boldsymbol A}\operatorname{Tr}(\boldsymbol A) = \boldsymbol I \qquad \frac{\partial}{\partial \boldsymbol A}\operatorname{Tr}(\boldsymbol A\boldsymbol B\boldsymbol A^T) = \boldsymbol A(\boldsymbol B + \boldsymbol B^T)
\end{equation*}
</div>

**Proposition (Inverse Derivative)**: We can show that:
<div>
\begin{equation*}
\frac{\partial}{\partial x}\boldsymbol A^{-1} = -\boldsymbol A^{-1}\frac{\partial \boldsymbol A}{\partial x}\boldsymbol A^{-1}
\end{equation*}
</div>

*Proof*: We will differentiate $\boldsymbol A^{-1}\boldsymbol A = \boldsymbol I$ instead, as we have:
<div>
\begin{equation*}
\boldsymbol 0 = \bracka{\frac{\partial}{\partial x} \boldsymbol I}\boldsymbol A^{-1} = \bracka{\frac{\partial}{\partial x}\boldsymbol A^{-1}\boldsymbol A}\boldsymbol A^{-1} = \bracka{\frac{\partial \boldsymbol A^{-1}}{\partial x}\boldsymbol A + \boldsymbol A^{-1}\frac{\partial \boldsymbol A}{\partial x}}\boldsymbol A^{-1}
\end{equation*}
</div>
Re-arrange the equation, and we get what we needed. 
$\tag*{$\Box$}$

**Proposition (Logarithm of Determinant)**: We can show that, assuming that $\boldsymbol A \in \mathbb{R}^n$ is diagonaliable:
<div>
\begin{equation*}
\frac{\partial}{\partial x}\ln\abs{\boldsymbol A} = \operatorname{Tr}\bracka{\boldsymbol A^{-1}\frac{\partial \boldsymbol A}{\partial x}}
\end{equation*}
</div>

*Proof*: Using the connection to eigenvalues. Let's consider the LHS, first, as we have:
<div>
\begin{equation*}
\frac{\partial}{\partial x}\ln\abs{\boldsymbol A} = \frac{\partial}{\partial x} \ln\bracka{\prod^n_{i=1}\lambda_i} = \frac{\partial}{\partial x} \sum^n_{i=1}\ln \lambda_i = \sum^n_{i=1}\frac{1}{\lambda_i}\frac{\partial \lambda_i}{\partial x}
\end{equation*}
</div>
For the RHS, we have:
<div>
\begin{equation*}
\begin{aligned}
    \operatorname{Tr}\bracka{ \brackb{\sum^n_{i=1}\frac{1}{\lambda_i}\boldsymbol u_i\boldsymbol u_i^T }\brackb{\sum^n_{i=1}\frac{\partial \lambda_i}{\partial x}\boldsymbol u_i\boldsymbol u_i^T } } 
    &= \operatorname{Tr}\bracka{\sum^n_{i=1}\sum^n_{j=1}\frac{1}{\lambda_i}\boldsymbol u_i\boldsymbol u_i^T\frac{\partial \lambda_j}{\partial x}\boldsymbol u_j\boldsymbol u_j^T} \\
    &= \sum^n_{i=1}\sum^n_{j=1}\frac{1}{\lambda_i}\frac{\partial \lambda_j}{\partial x}\operatorname{Tr}\bracka{\boldsymbol u_i\boldsymbol u_i^T \boldsymbol u_j\boldsymbol u_j^T} \\
    &= \sum^n_{i=1}\sum^n_{j=1}\frac{1}{\lambda_i}\frac{\partial \lambda_j}{\partial x}\operatorname{Tr}\bracka{\boldsymbol u_j^T\boldsymbol u_i\boldsymbol u_i^T \boldsymbol u_j} = \sum^n_{i=1}\frac{1}{\lambda_i}\frac{\partial \lambda_i}{\partial x}
\end{aligned}
\end{equation*}
</div>
As both sides are equal to each other, we have proven the claim.
$\tag*{$\Box$}$

**Corollary (Derivative of Log-Determinant)**: Using the above result, we can show that
<div>
\begin{equation*}
\frac{\partial}{\partial \boldsymbol A}\ln\abs{\boldsymbol A} = (\boldsymbol A^{-1})^T
\end{equation*}
</div>

*Proof*: Recall the previous results, as we have:
<div>
\begin{equation*}
\frac{\partial}{\partial a_{cd}} \ln\abs{\boldsymbol A} = \operatorname{Tr}\bracka{\boldsymbol A^{-1}\frac{\partial \boldsymbol A}{\partial a_{cd}}} = a^{-1}_{dc}
\end{equation*}
</div>
where we have $a_{ij}^{-1}$ being the $(i, j)$-th element of $\boldsymbol A^{-1}$. Thus, we have proven the equality. 
$\tag*{$\Box$}$

**Proposition (Derivative of Traces)**: We can show that:
<div>
\begin{equation*}
\frac{\partial}{\partial \boldsymbol A}\operatorname{Tr}[\boldsymbol A^T\boldsymbol B\boldsymbol A\boldsymbol C] = \boldsymbol B\boldsymbol A\boldsymbol C + \boldsymbol B^T\boldsymbol A\boldsymbol C^T
\end{equation*}
</div>

*Proof*: We will use the identity mapping $F_1(\cdot)$ and $F_2(\cdot)$ to make the differentiation faster. 
<div>
\begin{equation*}
\begin{aligned}
    \frac{\partial}{\partial\boldsymbol A}\operatorname{Tr}[\boldsymbol A^T\boldsymbol B\boldsymbol A\boldsymbol C] &= \frac{\partial}{\partial\boldsymbol A}\operatorname{Tr}[\boldsymbol F_1(\boldsymbol A)^T\boldsymbol B\boldsymbol F_2(\boldsymbol A)\boldsymbol C] \\
    &= \frac{\partial}{\partial \boldsymbol F_1}\operatorname{Tr}[\boldsymbol F_1^T\boldsymbol B\boldsymbol F_2\boldsymbol C]\frac{\partial \boldsymbol F_1}{\partial \boldsymbol A} + \frac{\partial}{\partial \boldsymbol F_2} \operatorname{Tr}[\boldsymbol F_1^T\boldsymbol B\boldsymbol F_2\boldsymbol C]\frac{\partial \boldsymbol F_2}{\partial \boldsymbol A} \\
    &= \frac{\partial}{\partial \boldsymbol F_1}\operatorname{Tr}[\boldsymbol F_1^T\boldsymbol B\boldsymbol F_2\boldsymbol C]\frac{\partial \boldsymbol F_1}{\partial \boldsymbol A} + \frac{\partial}{\partial \boldsymbol F_2} \operatorname{Tr}[\boldsymbol C\boldsymbol F_1^T\boldsymbol B\boldsymbol F_2]\frac{\partial \boldsymbol F_2}{\partial \boldsymbol A} \\
    &= \frac{\partial}{\partial \boldsymbol F_1}\operatorname{Tr}[\boldsymbol F_1^T\boldsymbol B\boldsymbol F_2\boldsymbol C]\frac{\partial \boldsymbol F_1}{\partial \boldsymbol A} + \frac{\partial}{\partial \boldsymbol F_2} \operatorname{Tr}[\boldsymbol F_2^T\boldsymbol B^T\boldsymbol F_1\boldsymbol C^T]\frac{\partial \boldsymbol F_2}{\partial \boldsymbol A} \\
    &= \boldsymbol B\boldsymbol F_2\boldsymbol C + \boldsymbol B^T\boldsymbol F_1\boldsymbol C^T = \boldsymbol B\boldsymbol A\boldsymbol C + \boldsymbol B^T\boldsymbol A\boldsymbol C^T
\end{aligned}
\end{equation*}
</div>
Note that we set $\boldsymbol F_1(\boldsymbol A) = \boldsymbol F_2(\boldsymbol A) = \boldsymbol A $
$\tag*{$\Box$}$

## Bayes' Theorem and Exponential Family

**Theorem (Bayes)**: We can show that, given $2$ events $X$ and $Y$:
<div>
\begin{equation*}
p(X|Y) = \frac{p(Y | X)p(X)}{p(Y)} \qquad \text{ where } \qquad p(Y) = \int p(Y | X)p(X) \dby X
\end{equation*}
</div>
where we call:
<div class="row">
  <div class="column">
    <ul>
        <li>$p(X \vert Y)$: Posterior</li>
        <li>$p(Y \vert X)$: Likelihood</li>
    </ul>
  </div>
  <div class="column">
    <ul>
        <li>$p(X)$: Prior</li>
        <li>$p(Y)$: Evidence</li>
    </ul>
  </div>
</div> 

*Proof*: This follows from the result of joint probability distribution:
<div>
\begin{equation*}
p(X, Y) = p(X|Y)p(Y) = p(Y | X)p(X)
\end{equation*}
</div>
$\tag*{$\Box$}$

*Remark (Problems of Bayesian Inference)*: It is hard to evaluate $p(Y)$, as we require to perform an integration. Most of the algorithms presented in this series try to mitigate this intractable integral. 

**Definition (Exponential Family)**: Exponential family of probability distribution is the probability distribution that has the form:
<div>
\begin{equation*}
\begin{aligned}
p(\boldsymbol x | \boldsymbol \theta) = & \ f(\boldsymbol x)g(\boldsymbol \theta)\exp\Big( \boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol T(\boldsymbol x) \Big) \\
&\text{ where } g(\boldsymbol \theta) \int f(\boldsymbol x) \exp\Big( \boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol T(\boldsymbol x) \Big) \dby \boldsymbol x = 1
\end{aligned}
\end{equation*}
</div>
where each component are named as:
- Sufficient Statistics $\boldsymbol T(\boldsymbol x) : \mathcal{X}\rightarrow \mathbb{R}^m$
- Natural Parameter $\boldsymbol \phi : \Theta \rightarrow \mathbb{R}^m$
- Auxilliary Functions: $f(\boldsymbol x) : \mathcal{X} \rightarrow \mathbb{R}$ and $g(\boldsymbol \theta) : \Theta \rightarrow \mathbb{R}$

**Definition (Conjugate Prior)**: *(Informally)* When performing Bayesian inference, conjugate prior (of some likelihood distirbution) is a prior such that the posterior belongs to the same family. *(Formally)*. Conjugate prior of the exponential family $p(\boldsymbol x \vert \boldsymbol \theta) = f(\boldsymbol x)g(\boldsymbol \theta)\exp( \boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol T(\boldsymbol x))$ is given as:
<div>
\begin{equation*}
\begin{aligned}
p(\boldsymbol \theta | \boldsymbol \tau, \nu) = f(\boldsymbol \tau, \nu) g(\boldsymbol \theta)^\nu \exp\Big( \boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol \tau \Big)
\end{aligned}
\end{equation*}
</div>

*Remark (Power of Conjugate Prior)*: Let's consider the case, where we have independent and identically distributed dataset $\mathcal{D} = \brackc{\boldsymbol x}^N_{i=1}$, the likelihood, in exponential family form, which is:
<div>
\begin{equation*}
\begin{aligned}
p(\mathcal{D}|\boldsymbol \theta) \propto \prod^N_{i=1} f(\boldsymbol x_i)g(\boldsymbol \theta)\exp\Big( \boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol T(\boldsymbol x_i)\Big) = g(\boldsymbol \theta)^N\bracka{\prod^N_{i=1} f(\boldsymbol x_i)}\exp\bracka{\sum^N_{i=1}\boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol T(\boldsymbol x_i)} 
\end{aligned}
\end{equation*}
</div>
If we consider the conjugate prior, then we have the following posterior:
<div>
\begin{equation*}
\begin{aligned}
p(\boldsymbol \theta | \mathcal{D}) &\propto p(\mathcal{D} | \boldsymbol \theta) p(\boldsymbol \theta | \boldsymbol \tau, \nu) \\
&\propto g(\boldsymbol \theta)^N\bracka{\prod^N_{i=1} f(\boldsymbol x_i)}\exp\bracka{\sum^N_{i=1}\boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol T(\boldsymbol x_i)} \ f(\boldsymbol \tau, \nu) g(\boldsymbol \theta)^\nu \exp\Big( \boldsymbol \phi(\boldsymbol \theta)^T\boldsymbol \tau \Big) \\
&= f\bracka{\boldsymbol \tau + \sum^N_{i=1}\boldsymbol T(\boldsymbol x_i), N + \nu} g(\boldsymbol \theta)^{N+\nu}\exp\bracka{\boldsymbol \phi(\boldsymbol \theta)^T\bracka{\boldsymbol \tau + \sum^N_{i=1}\boldsymbol T(\boldsymbol x_i)}}
\end{aligned}
\end{equation*}
</div>
Note that we don't have to find the normalizing factor directly (as we can use the result from conjugate prior's normalizer), removing the need to perform any integration.

**Definition (Normal-Wishart Distribution)**: Normal-Wishart distribution conjugates unknown mean and covariance normal distribution. It is defined as: $\mathcal{N}(\boldsymbol \mu \vert \boldsymbol \mu_0, (\lambda\boldsymbol \Lambda)^{-1})\mathcal{W}(\boldsymbol \Lambda \vert \boldsymbol W, \nu)$ or:
<div>
\begin{equation*}
\begin{aligned}
\mathcal{N}(&\boldsymbol \mu \vert \boldsymbol \mu_0, (\lambda\boldsymbol \Lambda)^{-1})\mathcal{W}(\boldsymbol \Lambda \vert \boldsymbol W, \nu) \\
&= \frac{1}{\sqrt{\abs{2\pi(\lambda\boldsymbol \Lambda)^{-1}}}}\exp\brackc{-\frac{\lambda}{2}(\boldsymbol \mu - \boldsymbol \mu_0)^T\boldsymbol \Lambda(\boldsymbol \mu-\boldsymbol \mu_0)}B(\boldsymbol W, \nu)\abs{\boldsymbol \Lambda}^{(\nu - D - 1)/2}\exp\bracka{-\frac{1}{2}\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda)} \\
&= \frac{ B(\boldsymbol W, \nu)\abs{\boldsymbol \Lambda}^{(\nu - D - 1)/2} }{\sqrt{\abs{2\pi(\lambda\boldsymbol \Lambda)^{-1}}}}\exp\brackc{-\frac{\lambda}{2}(\boldsymbol \mu - \boldsymbol \mu_0)^T\boldsymbol \Lambda(\boldsymbol \mu-\boldsymbol \mu_0)-\frac{1}{2}\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda)} \\
\end{aligned}
\end{equation*}
</div>

## Preliminaries

**Remark (Setting for Gaussian Parameter Estimation)**: Given the dataset $\brackc{\boldsymbol x}^N_{i=1}$, we are interested in finding the parameters $\boldsymbol \theta = \brackc{\boldsymbol \mu, \boldsymbol \Sigma}$ of the Gaussian that "matches" with the given data.

**Remark (Setting for Linear Regression)**: We are interesting to model the relationship between variable $\boldsymbol x$ and $\boldsymbol y$, where $\boldsymbol y$ is conditionally independent of $\boldsymbol x$, in which, we model the relationship as linear together with additive Gaussian noise:
<div>
\begin{equation*}
p(\boldsymbol y | \boldsymbol x, \boldsymbol W, \boldsymbol \Sigma_y) = \frac{1}{\sqrt{|2\pi\boldsymbol\Sigma_y|}}\exp\brackc{-\frac{1}{2}(\boldsymbol y - \boldsymbol W\boldsymbol x)^T\boldsymbol \Sigma_y^{-1}(\boldsymbol y - \boldsymbol W\boldsymbol x)}
\end{equation*}
</div>
Assuming we are given the dataset $\mathcal{D} = \brackc{(\boldsymbol x_i, \boldsymbol y_i)}^N_{i=1}$

## Maximum Likelihood Estimation (MLE)

**Definition (MLE)**: We would like to find the parameter $\boldsymbol \theta$, by solving the following optimization problem:
<div>
\begin{equation*}
\hat{\boldsymbol \theta}_\text{MLE} = \argmax{\boldsymbol \theta} p(\boldsymbol x | \boldsymbol \theta) \equiv \argmax{\boldsymbol \theta} \log p(\boldsymbol x | \boldsymbol \theta)
\end{equation*}
</div>
Note that the log-likelihood is valid because $\log(\cdot)$ is non-decreasing function

**Proposition (MLE of Gaussian Parameter Estimation- Mean)**: MLE solution of Gaussian Parameter estimation of the mean $\boldsymbol \mu$ is:
<div>
\begin{equation*}
\hat{\boldsymbol \mu} = \frac{1}{N}\sum^N_{i=1}\boldsymbol x_i
\end{equation*}
</div>

*Proof*: The log-likelihood of the problem, first, which is:
<div>
\begin{equation*}
l(\boldsymbol \mu, \boldsymbol \Sigma) = \log \prod^N_{i=1}\mathcal{N}(\boldsymbol x | \boldsymbol \mu, \boldsymbol \Sigma) = -\frac{N}{2}\log\abs{2\pi\boldsymbol \Sigma} - \frac{1}{2}\sum^N_{i=1}(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_i - \boldsymbol \mu)
\end{equation*}
</div>
Now, we can consider the derivative of (negative) log-likelhood with respected to $\boldsymbol \mu$ as:
<div>
\begin{equation*}
\begin{aligned}
\frac{\partial (-l)}{\partial \boldsymbol \mu} 
&= \frac{\partial}{\partial \boldsymbol \mu} \brackb{\frac{N}{2}\log\abs{2\pi\boldsymbol \Sigma} + \frac{1}{2}\sum^N_{i=1}(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_i - \boldsymbol \mu)} \\
&= \frac{1}{2}\frac{\partial}{\partial \boldsymbol \mu} \sum^N_{i=1}(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_i - \boldsymbol \mu) \\
&= \frac{1}{2} \sum^N_{i=1}\bracka{\frac{\partial}{\partial \boldsymbol \mu} \brackb{\boldsymbol \mu^T\boldsymbol \Sigma^{-1}\boldsymbol \mu} - 2 \brackb{\boldsymbol \mu^T\boldsymbol \Sigma^{-1}\boldsymbol x_i}} = N\boldsymbol \Sigma^{-1}\boldsymbol \mu - \boldsymbol \Sigma^{-1}\sum^N_{i=1}\boldsymbol x_i \\
\end{aligned}
\end{equation*}
</div>
If we set the derivative to $0$, we yields the result, like above.
$\tag*{$\Box$}$

**Proposition (MLE of Gaussian Parameter Estimation- Covariance)**: MLE solution of Gaussian Parameter estimation of the covariance $\boldsymbol \Sigma$ is:
<div>
\begin{equation*}
\widehat{\boldsymbol \Sigma} = \frac{1}{N}\sum^N_{i=1} (\boldsymbol x_i-\boldsymbol \mu)(\boldsymbol x_i-\boldsymbol \mu)^T
\end{equation*}
</div>

*Proof*: Consider the derivative of the (negative) log-likelihood with respected to $\boldsymbol \Sigma^{-1}$ (we need the constraint that it is positive semi-definite (psd), however as it turn out that the unconstriant optimization leads to psd. answer):
<div>
\begin{equation*}
\begin{aligned}
\frac{\partial(-l)}{\partial \boldsymbol \Sigma^{-1}} 
&= \frac{\partial}{\partial \boldsymbol \Sigma^{-1}}\brackb{\frac{N}{2}\log\abs{2\pi\boldsymbol \Sigma} + \frac{1}{2}\sum^N_{i=1}(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_i - \boldsymbol \mu)} \\
&= \frac{N}{2}\frac{\partial}{\partial \boldsymbol \Sigma^{-1}}\log\abs{2\pi\boldsymbol \Sigma} + \frac{1}{2}\sum^N_{i=1}\frac{\partial}{\partial \boldsymbol \Sigma^{-1}}\Big[(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_i - \boldsymbol \mu)\Big] \\
&= \frac{N}{2}\frac{\partial}{\partial \boldsymbol \Sigma^{-1}}\log\abs{2\pi\boldsymbol \Sigma} + \frac{1}{2}\sum^N_{i=1}\frac{\partial}{\partial \boldsymbol \Sigma^{-1}}\operatorname{Tr}\Big[(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x_i - \boldsymbol \mu)\Big] \\
&= -\frac{N}{2}\boldsymbol \Sigma^T + \frac{1}{2}\sum^N_{i=1}(\boldsymbol x_i-\boldsymbol \mu)(\boldsymbol x_i-\boldsymbol \mu)^T
\end{aligned}
\end{equation*}
</div>
$\tag*{$\Box$}$


**Proposition (MLE of Linear Regression)**: We can show that MLE solution of the linear regression is:
<div>
\begin{equation*}
\widehat{\boldsymbol W} = \sum^N_{i=1}\boldsymbol y_i\boldsymbol x_i^T\bracka{\sum^N_{i=1}\boldsymbol x_i\boldsymbol x_i^T}^{-1}
\end{equation*}
</div>

*Proof*: Let's start with finding log-likelihood as we have:
<div>
\begin{equation*}
\begin{aligned}
l(\boldsymbol W) = 
-\frac{N}{2}\log\abs{2\pi\boldsymbol \Sigma_y} -\frac{1}{2}\sum^N_{i=1}(\boldsymbol y_i - \boldsymbol W\boldsymbol x_i)^T\boldsymbol \Sigma_y^{-1}(\boldsymbol y_i - \boldsymbol W\boldsymbol x_i)
\end{aligned}
\end{equation*}
</div>
Let's consider the derivative of (negative) log-likelihood with respected to $\boldsymbol W$, which is:
<div>
\begin{equation*}
\begin{aligned}
\frac{\partial (-l)}{\partial \boldsymbol W} &= \frac{\partial}{\partial \boldsymbol W}\brackb{\frac{N}{2}\log\abs{2\pi\boldsymbol \Sigma_y} -\frac{1}{2}\sum^N_{i=1}(\boldsymbol y_i - \boldsymbol W\boldsymbol x_i)^T\boldsymbol \Sigma_y^{-1}(\boldsymbol y_i - \boldsymbol W\boldsymbol x_i)} \\
&= \frac{1}{2}\sum^N_{i=1}\frac{\partial}{\partial \boldsymbol W}(\boldsymbol y_i - \boldsymbol W\boldsymbol x_i)^T\boldsymbol \Sigma^{-1}_y(\boldsymbol y_i - \boldsymbol W\boldsymbol x_i) \\
&= \frac{1}{2}\sum^N_{i=1}\frac{\partial}{\partial \boldsymbol W}\Big[ \boldsymbol y_i^T\boldsymbol \Sigma_y^{-1}\boldsymbol y_i - 2\boldsymbol x_i^T\boldsymbol W^T\boldsymbol \Sigma_y^{-1}\boldsymbol y_i + \boldsymbol x_i^T\boldsymbol W^T\boldsymbol \Sigma^{-1}_y\boldsymbol W\boldsymbol x_i \Big] \\
&= \frac{1}{2}\sum^N_{i=1}\bracka{\frac{\partial}{\partial \boldsymbol W}\operatorname{Tr}\Big[ \boldsymbol x_i^T\boldsymbol W^T\boldsymbol \Sigma^{-1}_y\boldsymbol W\boldsymbol x_i\Big] - \frac{\partial}{\partial \boldsymbol W}\operatorname{Tr}\Big[2\boldsymbol x_i^T\boldsymbol W^T\boldsymbol \Sigma_y^{-1}\boldsymbol y_i \Big]} \\
&= \frac{1}{2}\sum^N_{i=1}\bracka{\frac{\partial}{\partial \boldsymbol W}\operatorname{Tr}\Big[\boldsymbol W^T\boldsymbol \Sigma^{-1}_y\boldsymbol W\boldsymbol x_i\boldsymbol x_i^T\Big] - 2\frac{\partial}{\partial \boldsymbol W}\operatorname{Tr}\Big[\boldsymbol W^T\boldsymbol \Sigma_y^{-1}\boldsymbol y_i\boldsymbol x_i^T \Big]} \\
&= \frac{1}{2}\sum^N_{i=1}\bracka{2\boldsymbol \Sigma^{-1}_y\boldsymbol W\boldsymbol x_i\boldsymbol x_i^T - 2\boldsymbol \Sigma_y^{-1}\boldsymbol y_i\boldsymbol x_i^T} \\
\end{aligned}
\end{equation*}
</div>
If we set the derivative to zero, we get the final result, as required. 
$\tag*{$\Box$}$


## Maximum A Posteriori (MAP) Estimation

**Definition (MAP Estimate)**: To estimate the parameter $\boldsymbol \theta$, we find the mode of the posterior instead of likelihood:
<div>
\begin{equation*}
\begin{aligned}
\argmax{\boldsymbol \theta} p(\boldsymbol \theta | \boldsymbol x) \equiv \argmax{\boldsymbol \theta} \log p(\boldsymbol \theta | \boldsymbol x)
\end{aligned}
\end{equation*}
</div>

**Remark (MAP Estimation)**: If we consider Bayes' theorem:
<div>
\begin{equation*}
\begin{aligned}
p(\boldsymbol \theta | \boldsymbol x) = \frac{p(\boldsymbol x | \boldsymbol \theta)p(\boldsymbol \theta)}{p(\boldsymbol x)}
\end{aligned}
\end{equation*}
</div>
We can see that the denominator doesn't depend on $\boldsymbol \theta$, so we can simply optimize $\boldsymbol \theta$ given $p(\boldsymbol x | \boldsymbol \theta)p(\boldsymbol \theta)$ only. This has an advantage because, we don't have to find the evidence $p(\boldsymbol x)$, which is hard to compute (have to calculate the integral). 

**Proposition (Normal-Wishart Posterior)**: Given the dataset $\brackc{\boldsymbol x_i}^N_{i=1}$, we can show that the posterior of normal distribution with unknown mean and covariance given normal-wishart prior is:
<div>
\begin{equation*}
\begin{aligned}
\abs{\boldsymbol \Lambda}^{(\nu - D + N)/2}\exp\Bigg(-\frac{1}{2}&\operatorname{Tr}\bracka{\brackb{\boldsymbol W^{-1} + \sum^N_{i=1}(\boldsymbol x_i - \bar{\boldsymbol x})(\boldsymbol x_i - \bar{\boldsymbol x})^T + \frac{N\lambda}{N+\lambda}(\bar{\boldsymbol x} - \boldsymbol \mu_0)(\bar{\boldsymbol x} - \boldsymbol \mu_0)^T }\boldsymbol \Lambda } \\
&-\frac{N+\lambda}{2}\bracka{\boldsymbol \mu - \frac{\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x}}{\lambda + N}}^T\boldsymbol \Lambda\bracka{\boldsymbol \mu - \frac{\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x}}{\lambda + N}}\Bigg)
\end{aligned}
\end{equation*}
</div>
where $\bar{\boldsymbol x} = 1/N\sum^N_{i=1}\boldsymbol x_i$

*Proof follows from* <a href="https://stats.stackexchange.com/questions/153241/derivation-of-normal-wishart-posterior" target="_blank">here</a> : Let's consider, the likelihood, first, which we have: 
<div>
\begin{equation*}
\begin{aligned}
\prod^N_{i=1}\frac{1}{\sqrt{\abs{2\pi\boldsymbol \Lambda^{-1}}}} &\exp\brackc{-\frac{1}{2}(x_i-\boldsymbol \mu)^T\boldsymbol \Lambda(\boldsymbol x_i - \boldsymbol mu)} \\
&\propto \abs{\boldsymbol \Lambda}^{N/2}\exp\brackc{-\frac{1}{2}\sum^N_{i=1}(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Lambda(\boldsymbol x_i - \boldsymbol \mu)}
\end{aligned}
\end{equation*}
</div>
For the prior, we have:
<div>
\begin{equation*}
\begin{aligned}
\frac{ B(\boldsymbol W, \nu)\abs{\boldsymbol \Lambda}^{(\nu - D - 1)/2} }{\sqrt{\abs{2\pi(\lambda\boldsymbol \Lambda)^{-1}}}}&\exp\brackc{-\frac{\lambda}{2}(\boldsymbol \mu - \boldsymbol \mu_0)^T\boldsymbol \Lambda(\boldsymbol \mu-\boldsymbol \mu_0)-\frac{1}{2}\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda)} \\
&\propto \abs{\boldsymbol\Lambda}^{(\nu - D)/2}\exp\brackc{-\frac{\lambda}{2}(\boldsymbol \mu - \boldsymbol \mu_0)^T\boldsymbol \Lambda(\boldsymbol \mu-\boldsymbol \mu_0)-\frac{1}{2}\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda)} \\
\end{aligned}
\end{equation*}
</div>
Let's consider the joint distribution (+ some simplification):
<div>
\begin{equation*}
\begin{aligned}
\abs{\boldsymbol\Lambda}^{(\nu - D)/2}&\exp\brackc{-\frac{\lambda}{2}(\boldsymbol \mu - \boldsymbol \mu_0)^T\boldsymbol \Lambda(\boldsymbol \mu-\boldsymbol \mu_0)-\frac{1}{2}\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda)} \abs{\boldsymbol \Lambda}^{N/2}\exp\brackc{-\frac{1}{2}\sum^N_{i=1}(\boldsymbol x_i - \boldsymbol \mu)^T\boldsymbol \Lambda(\boldsymbol x_i - \boldsymbol \mu)} \\
&= \begin{aligned}[t]
    \abs{\boldsymbol\Lambda}^{(\nu - D)/2}& \exp\bracka{-\frac{1}{2}\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda)} \\
    &\exp\bracka{-\frac{1}{2}\brackb{\sum^N_{i=1}\bracka{\boldsymbol x_i^T\boldsymbol \Lambda\boldsymbol x_i} + N\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol \mu - 2N\bar{\boldsymbol x}^T\boldsymbol \Lambda\boldsymbol \mu}} \\
    &\exp\bracka{-\frac{\lambda}{2}\brackb{\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol \mu - 2 \boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu + \boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0}}
\end{aligned} \\
&= \begin{aligned}[t]
    \abs{\boldsymbol\Lambda}^{(\nu - D)/2} \exp\Bigg(-\frac{1}{2}\Bigg[\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda) &+ \sum^N_{i=1}(\boldsymbol x_i^T\boldsymbol \Lambda\boldsymbol x_i) + (N+\lambda)\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol \mu \\
    &- 2(N\bar{\boldsymbol x}^T + \lambda\boldsymbol \mu_0^T) \boldsymbol \Lambda\boldsymbol \mu+ \lambda\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0\Bigg]\Bigg) \\
\end{aligned}
\end{aligned}
\end{equation*}
</div>
Note that, we still uses the completing the square (over $\boldsymbol \mu$) to find the posterior. Let's consider the value within the exponential:
<div>
\begin{equation*}
\begin{aligned}
\operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda) &+ \sum^N_{i=1}(\boldsymbol x_i^T\boldsymbol \Lambda\boldsymbol x_i) + (N+\lambda)\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol \mu - 2(N\bar{\boldsymbol x}^T + \lambda\boldsymbol \mu_0^T) \boldsymbol \Lambda\boldsymbol \mu+ \lambda\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 \\
&= \begin{aligned}[t]
    \operatorname{Tr}(\boldsymbol W^{-1}\boldsymbol \Lambda) 
    &+ {\color{blue} (N+\lambda)\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol \mu - 2(N\bar{\boldsymbol x}^T + \lambda\boldsymbol \mu_0^T) \boldsymbol \Lambda\boldsymbol \mu + \frac{1}{\lambda+N}(\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x})^T\boldsymbol \Lambda(\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x})} \\
    &+ {\color{pink} \sum^N_{i=1}(\boldsymbol x_i^T\boldsymbol \Lambda\boldsymbol x_i) + N\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} - 2N\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} }\\
    &+ {\color{purple} N\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} + \lambda\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 - \frac{1}{\lambda+N}(\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x})^T\boldsymbol \Lambda(\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x})} \\
\end{aligned}
\end{aligned}
\end{equation*}
</div>
For the <font color="#2196f3">blue</font> part, we use the completing the square technique, which gives us the quadratic terms:
<div>
\begin{equation*}
(N+\lambda)\bracka{\boldsymbol \mu - \frac{\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x}}{\lambda + N}}^T\boldsymbol \Lambda\bracka{\boldsymbol \mu - \frac{\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x}}{\lambda + N}}
\end{equation*}
</div>
Similarly, for the <font color="#e91e62">pink</font> term, we also use completing the square to get the quadratic terms:
<div>
\begin{equation*}
\sum^N_{i=1}(\boldsymbol x_i - \bar{\boldsymbol x})^T\boldsymbol \Lambda(\boldsymbol x_i - \bar{\boldsymbol x})
\end{equation*}
</div>
Finally, for <font color="#673ab7">purple</font> term, we can show that it is also, a quadratic terms as:
<div>
\begin{equation*}
\begin{aligned}
N\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} &+ \lambda\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 - \frac{1}{\lambda+N}(\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x})^T\boldsymbol \Lambda(\lambda\boldsymbol \mu_0 + N\bar{\boldsymbol x}) \\
&= \frac{N-\lambda}{N-\lambda}N\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} + \frac{N-\lambda}{N-\lambda}\lambda\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 - \frac{1}{\lambda+N}\Big[\lambda^2\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 + \lambda N\bar{\boldsymbol x}^T\boldsymbol \Lambda\boldsymbol \mu_0 + N^2\bar{\boldsymbol x}^T\boldsymbol \Lambda \bar{\boldsymbol x}\Big] \\
&= \begin{aligned}[t]
    \frac{1}{\lambda +N}\Big[ \lambda^2\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 &+ N\lambda\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 + N\lambda\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} + N^2\bar{\boldsymbol x}^T\boldsymbol \Lambda\bar{\boldsymbol x} \Big] \\
    &- \frac{1}{\lambda+N}\Big[\lambda^2\boldsymbol \mu_0^T\boldsymbol \Lambda\boldsymbol \mu_0 + \lambda N\bar{\boldsymbol x}^T\boldsymbol \Lambda\boldsymbol \mu_0 + N^2\bar{\boldsymbol x}^T\boldsymbol \Lambda \bar{\boldsymbol x}\Big] \\
\end{aligned} \\
&= \frac{N\lambda}{N+\lambda}(\bar{\boldsymbol x} - \boldsymbol \mu_0)^T\boldsymbol \Lambda(\bar{\boldsymbol x} - \boldsymbol \mu_0)
\end{aligned}
\end{equation*}
</div>
We then combine the first part (black, <font color="#2196f3">blue</font> and <font color="#e91e62">pink</font>) together with trace trick:
<div>
\begin{equation*}
\operatorname{Tr}\bracka{\brackb{\boldsymbol W^{-1} + \sum^N_{i=1}(\boldsymbol x_i - \bar{\boldsymbol x})(\boldsymbol x_i - \bar{\boldsymbol x})^T + \frac{N\lambda}{N+\lambda}(\bar{\boldsymbol x} - \boldsymbol \mu_0)(\bar{\boldsymbol x} - \boldsymbol \mu_0)^T }\boldsymbol \Lambda } \\
\end{equation*}
</div>
and we yields the solution, as required. 

**Proposition (MLE Estimate of Linear Regression)**: Given the prior over weight to be $p(\boldsymbol w) = \mathcal{N}(\boldsymbol w | \boldsymbol 0, \boldsymbol A^{-1})$ (we will consider one dimensional output i.e $\boldsymbol w^T\boldsymbol x$ with variance of $\sigma_y^2$), the MAP estimate is:
<div>
\begin{equation*}
\boldsymbol w_\text{MAP} = \bracka{\sigma_y^2\boldsymbol A + \sum^N_{i=1}\boldsymbol x_i\boldsymbol x_i^T}^{-1}\sum^N_{i=1}y_i\boldsymbol x_i
\end{equation*}
</div>

*Proof*: Let's consider the posterior first, as we have note that the likelihood is 
<div>
\begin{equation*}
    p(\brackc{y_i}^N_{i=1} | \brackc{\boldsymbol x_i}^N_{i=1}, \boldsymbol w, \sigma_y^2) = \frac{1}{\sqrt{2\pi\sigma^2_y}}\exp\brackc{-\frac{1}{2\sigma^2_y}\sum^N_{i=1}(y_i - \boldsymbol w^T\boldsymbol x_i)^2}
\end{equation*}
</div>
and, so the posterior over the weight is given as:
<div>
\begin{equation*}
\begin{aligned}
    \log p(\boldsymbol w | \mathcal{D}, \boldsymbol A, \sigma_y^2) &= \log p(\brackc{y_i}^N_{i=1} | \brackc{\boldsymbol x_i}^N_{i=1}, \boldsymbol w, \sigma_y^2)+ \log p(\boldsymbol w | \boldsymbol A) + \text{const} \\
    &= -\frac{1}{2}\boldsymbol w^T\boldsymbol A\boldsymbol w - \frac{1}{2\sigma^2_y}\sum^N_{i=1}(y_i - \boldsymbol w^T\boldsymbol x_i)^2 + \text{const} \\
    &= -\frac{1}{2}\boldsymbol w^T\boldsymbol A\boldsymbol w - \frac{1}{2\sigma^2_y}\sum^N_{i=1}\Big[ y_i^2 - 2y_i\boldsymbol w^T\boldsymbol x_i + (\boldsymbol w^T\boldsymbol x_i)^2 \Big] + \text{const} \\
    &= -\frac{1}{2}\boldsymbol w^T\boldsymbol A\boldsymbol w + \frac{1}{2\sigma^2_y}\sum^N_{i=1}2y_i\boldsymbol w^T\boldsymbol x_i -\frac{1}{2\sigma^2_y}\sum^N_{i=1} \boldsymbol w^T\boldsymbol x_i\boldsymbol x_i^T\boldsymbol w  + \text{const} \\
    &= -\frac{1}{2}\boldsymbol w^T\underbrace{\bracka{\boldsymbol A + \frac{1}{\sigma^2_y}\sum^N_{i=1}\boldsymbol x_i\boldsymbol x_i^T}}_{\boldsymbol \Sigma^{-1}_w}\boldsymbol w + \frac{1}{\sigma^2_y}\sum^N_{i=1}y_i\boldsymbol w^T\boldsymbol x_i + \text{const} \\
    &= -\frac{1}{2}\boldsymbol w^T\boldsymbol \Sigma^{-1}_w\boldsymbol w + \frac{1}{\sigma^2_y}\boldsymbol \Sigma^{-1}_w\boldsymbol \Sigma_w\sum^N_{i=1}y_i\boldsymbol w^T\boldsymbol x_i + \text{const} \\
\end{aligned}
\end{equation*}
</div>
In this form, we have the posterior to be:

<div>
\begin{equation*}
\log p(\boldsymbol w | \mathcal{D}, \boldsymbol A, \sigma_y^2) = \mathcal{N}\bracka{\frac{1}{\sigma^2_y}\boldsymbol \Sigma_w\sum^N_{i=1}y_i\boldsymbol w^T\boldsymbol x_i, \ \boldsymbol \Sigma_w}
\end{equation*}
</div>
Note that the mode of Gaussian is its mean, with some rearrangement we yields the result like ahove, as required.

## Baysian Linear Regression

**Remark (Posterior over matrix of Data)**: We can consider $\boldsymbol X \in \mathbb{R}^{p \times N}$ where $p$ is the dimension of the data. Together with the prior over weight $\boldsymbol \beta \sim \mathcal{N}(\boldsymbol 0, \boldsymbol \Sigma)$ and likelihood $p(\boldsymbol y \vert \boldsymbol X, \boldsymbol \beta) \sim\mathcal{N}(\boldsymbol X^T\boldsymbol \beta, \sigma^2\boldsymbol I)$. Then, we have the posterior $p(\boldsymbol \beta | \boldsymbol X, \boldsymbol y) = \mathcal{N}(\boldsymbol \beta | \boldsymbol \mu_\beta, \boldsymbol \Sigma_\beta)$, where
<div>
\begin{equation*}
\boldsymbol \Sigma_{\boldsymbol \beta} = \bracka{\boldsymbol \Sigma^{-1} + \frac{\boldsymbol X\boldsymbol X^T}{\sigma^2}}^{-1} \qquad \boldsymbol \mu_\beta = \boldsymbol \Sigma_\beta\bracka{\frac{\boldsymbol X\boldsymbol y}{\sigma^2}}
\end{equation*}
</div>

**Corollary (Predictive)**

