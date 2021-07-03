---
layout: post
title: "Part 3: Manipulating Gaussian"
description: "Master the art of Gaussian Bending"
subject: "Probabilistic Machine Learning"
---

This post is based from the book: Pattern Recognition and Machine Learning, where we consider various properties of Gaussian and results, such as conditioning or Gaussian under linear transformation.

## Useful Backgrounds

<span class="anchor" id="prop-1"></span>**Proposition (Inverse of Partition Matrix)**: The block matrix can be inversed as:
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
\begin{bmatrix}
    \boldsymbol A & \boldsymbol B \\ 
    \boldsymbol C & \boldsymbol D \\ 
\end{bmatrix} = 
\begin{bmatrix}
    \boldsymbol M & -\boldsymbol M\boldsymbol B\boldsymbol D^{-1} \\ 
    -\boldsymbol D^{-1}\boldsymbol C\boldsymbol M & \boldsymbol D^{-1}+ \boldsymbol D^{-1}\boldsymbol C\boldsymbol M\boldsymbol B\boldsymbol D^{-1} \\ 
\end{bmatrix}
\end{equation*}
</div>
where we set $\boldsymbol M = (\boldsymbol A-\boldsymbol B\boldsymbol D^{-1}\boldsymbol C)^{-1}$

<span class="anchor" id="prop-4"></span>**Proposition (Inverse Matrix Identity)**: We can show that
<div>
\begin{equation*}
    (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\boldsymbol B\boldsymbol R^{-1} = \boldsymbol P\boldsymbol B^T(\boldsymbol B\boldsymbol P\boldsymbol B^T + \boldsymbol R)^{-1}
\end{equation*}
</div>

*Proof*: The can be proven by right multiply the inverse on the right hand-side:
<div>
\begin{equation*}
\begin{aligned}
    (\boldsymbol P^{-1} + &\boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\boldsymbol B\boldsymbol R^{-1}(\boldsymbol B\boldsymbol P\boldsymbol B^T + \boldsymbol R) \\
    &= (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\boldsymbol B\boldsymbol R^{-1}\boldsymbol B\boldsymbol P\boldsymbol B^T + (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\boldsymbol B\boldsymbol R^{-1}\boldsymbol R \\
    &= (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\boldsymbol B\boldsymbol R^{-1}\boldsymbol B\boldsymbol P\boldsymbol B^T + (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\boldsymbol B \\
    &= (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\Big[\boldsymbol B\boldsymbol R^{-1}\boldsymbol B\boldsymbol P + \boldsymbol P^{-1} \boldsymbol P\Big]\boldsymbol B^T\\
    &= (\boldsymbol P^{-1} + \boldsymbol B^T\boldsymbol R^{-1}\boldsymbol B)^{-1}\Big[\boldsymbol B\boldsymbol R^{-1}\boldsymbol B+ \boldsymbol P^{-1} \Big]\boldsymbol P\boldsymbol B^T \\
    &= \boldsymbol P\boldsymbol B^T \\
\end{aligned}
\end{equation*}
</div>
$\tag*{$\Box$}$

<span class="anchor" id="prop-5"></span>**Proposition (Woodbury Identity)**: We can show that
<div>
\begin{equation*}
    (\boldsymbol A + \boldsymbol B\boldsymbol D^{-1}\boldsymbol C)^{-1} = \boldsymbol A^{-1} - \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1}
\end{equation*}
</div>
*Proof*: The can be proven by right multiply the inverse on the right hand-side:
<div>
\begin{equation*}
\begin{aligned}
    \Big[ \boldsymbol A^{-1} &- \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1} \Big](\boldsymbol A + \boldsymbol B\boldsymbol D^{-1}\boldsymbol C) \\
    &= \boldsymbol A^{-1}(\boldsymbol A + \boldsymbol B\boldsymbol D^{-1}\boldsymbol C) - \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1}(\boldsymbol A + \boldsymbol B\boldsymbol D^{-1}\boldsymbol C) \\
    &= \begin{aligned}[t]
        \boldsymbol A^{-1}\boldsymbol A + \boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C &- \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1}\boldsymbol A \\
        &- \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C \\
    \end{aligned} \\
    &= \begin{aligned}[t]
        \boldsymbol I + \boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C &- \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C \\
        &- \boldsymbol A^{-1}\boldsymbol B(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C \\
    \end{aligned} \\
    &= \begin{aligned}[t]
        \boldsymbol I + \boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C &- \boldsymbol A^{-1}\boldsymbol B\Big[(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol D + (\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}\boldsymbol C\boldsymbol A^{-1}\boldsymbol B\Big]\boldsymbol D^{-1}\boldsymbol C \\
    \end{aligned} \\
    &= \begin{aligned}[t]
        \boldsymbol I + \boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C &- \boldsymbol A^{-1}\boldsymbol B\Big[(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)^{-1}(\boldsymbol D + \boldsymbol C\boldsymbol A^{-1}\boldsymbol B)\Big]\boldsymbol D^{-1}\boldsymbol C \\
    \end{aligned} \\
    &= \begin{aligned}[t]
        \boldsymbol I + \boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C &- \boldsymbol A^{-1}\boldsymbol B\boldsymbol D^{-1}\boldsymbol C \\
    \end{aligned} = \boldsymbol I
\end{aligned}
\end{equation*}
</div>
$\tag*{$\Box$}$

## Conditional & Marginalisation

**Remark (Settings)**: We consider the setting where we consider the partition of random variables:

<div>
\begin{equation*}
\begin{bmatrix}
    \boldsymbol x_a \\ \boldsymbol x_b
\end{bmatrix}
\sim \mathcal{N}\bracka{
\begin{bmatrix}
    \boldsymbol \mu_a \\ \boldsymbol \mu_b
\end{bmatrix}, \begin{bmatrix}
    \boldsymbol \Sigma_{aa} & \boldsymbol \Sigma_{ab} \\ 
    \boldsymbol \Sigma_{ba} & \boldsymbol \Sigma_{bb} \\ 
\end{bmatrix}}
\end{equation*}
</div>
Due to the symmetric of covariance, we have $\boldsymbol \Sigma_{ab} = \boldsymbol \Sigma_{ba}^T$. Furthermore, we denote 
<div>
\begin{equation*}
\boldsymbol \Sigma^{-1} = \begin{bmatrix}
    \boldsymbol \Sigma_{aa} & \boldsymbol \Sigma_{ab} \\ 
    \boldsymbol \Sigma_{ba} & \boldsymbol \Sigma_{bb} \\ 
\end{bmatrix}^{-1} = 
\begin{bmatrix}
    \boldsymbol \Lambda_{aa} & \boldsymbol \Lambda_{ab} \\ 
    \boldsymbol \Lambda_{ba} & \boldsymbol \Lambda_{bb} \\ 
\end{bmatrix} = \boldsymbol \Lambda
\end{equation*}
</div>
where $\boldsymbol \Lambda$ is called precision matrix. One can consider the inverse of partition matrix to find such a value of $\boldsymbol \Lambda$ (will be useful afterward), thus we note that $\boldsymbol \Sigma_{aa} \ne \boldsymbol \Lambda^{-1}_{aa}$, and so on.

**Remark (Complete the Square)**: To find the conditional and marginalision (together with other kinds of Gaussian manipulation), we relies on a method called completing the square. Let's consider the quadratic form expansion:
<div>
\begin{equation*}
\begin{aligned}
    -\frac{1}{2}&(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu) \\
    &= {\color{blue}{-\frac{1}{2}\boldsymbol x^T\boldsymbol \Sigma^{-1}\boldsymbol x + \boldsymbol \mu^T\boldsymbol \Sigma^{-1}\boldsymbol x}} -\frac{1}{2} \boldsymbol \mu^T\boldsymbol \Sigma^{-1}\boldsymbol \mu \\
\end{aligned}
\end{equation*}
</div>
Note that the <font color="#2196f3">blue term</font> are the term that depends on $x$. This means that to find the Gaussian, we will have to find the first and second order of $\boldsymbol x$ only. Or, we can consider each individual elements of partitioned random variable:
<div>
\begin{equation*}
\begin{aligned}
    -\frac{1}{2}&(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Sigma^{-1}(\boldsymbol x-\boldsymbol \mu) \\
    &= \begin{aligned}[t]
        {\color{green}{-\frac{1}{2}(\boldsymbol x_a - \boldsymbol \mu_a)^T\boldsymbol \Lambda_{aa}(\boldsymbol x_a - \boldsymbol \mu_a)}}{\color{yellow}{ - \frac{1}{2}(\boldsymbol x_a - \boldsymbol \mu_a)^T\boldsymbol \Lambda_{ab}(\boldsymbol x_b - \boldsymbol \mu_b)}} \\
        {\color{purple}{-\frac{1}{2}(\boldsymbol x_b - \boldsymbol \mu_b)^T\boldsymbol \Lambda_{ba}(\boldsymbol x_a - \boldsymbol \mu_a)}}{\color{grey}{ - \frac{1}{2}(\boldsymbol x_b - \boldsymbol \mu_b)^T\boldsymbol \Lambda_{bb}(\boldsymbol x_b - \boldsymbol \mu_b)}} \\
    \end{aligned} \\
    &= \begin{aligned}[t]
        &{\color{green} -\frac{1}{2}\boldsymbol x^T_a\boldsymbol \Lambda_{aa}\boldsymbol x_a + \boldsymbol \mu_a^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} -\frac{1}{2} \boldsymbol \mu^T\boldsymbol \Lambda_{aa}\boldsymbol \mu {\color{yellow} -\frac{1}{2}\boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol x_b + \frac{1}{2} \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b} \\ 
        &{\color{yellow}+\frac{1}{2}\boldsymbol \mu_a^T\boldsymbol \Lambda_{ab}\boldsymbol x_b} - \frac{1}{2}\boldsymbol \mu_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b {\color{purple} -\frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a + \frac{1}{2}\boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a + \frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{ba} \boldsymbol \mu_a}  \\
        &- \frac{1}{2}\boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}\boldsymbol \mu_a {\color{grey} -\frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b + \boldsymbol \mu_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b} -\frac{1}{2}\boldsymbol \mu_b^T\boldsymbol \Lambda_{bb}\boldsymbol \mu_b
    \end{aligned} \\
\end{aligned}
\label{eqn:1}\tag{1}
\end{equation*}
</div>
Completing the square is to match this pattern into our formula in order to get new Gaussian distribution. There are $2$ ways to complete the squre that depends on the scenario:
- When we want to find the Gaussian in difference form (but still being Gaussian) i.e conditional
- When we want to marginalise some variables out *or* when we have to find the true for of the distribution without relying on knowing the final form (or when we are not really sure about the final form) i.e marginalisation, posterior

Let's just show how it works with examples. 

<span class="anchor" id="prop-3"></span>**Proposition (Conditional)**: Consider the following Gaussian 
<div>
\begin{equation*}
\begin{bmatrix}
    \boldsymbol x_a \\ \boldsymbol x_b
\end{bmatrix}
\sim p(\boldsymbol x_a, \boldsymbol x_b) = \mathcal{N}\bracka{
\begin{bmatrix}
    \boldsymbol \mu_a \\ \boldsymbol \mu_b
\end{bmatrix}, \begin{bmatrix}
    \boldsymbol \Sigma_{aa} & \boldsymbol \Sigma_{ab} \\ 
    \boldsymbol \Sigma_{ba} & \boldsymbol \Sigma_{bb} \\ 
\end{bmatrix}}
\end{equation*}
</div>
We can show that: $p(\boldsymbol x_a | \boldsymbol x_b) = \mathcal{N}(\boldsymbol x | \boldsymbol \mu_{a|b}, \boldsymbol \Lambda^{-1}_{aa})$, where we have
- $\boldsymbol \mu_{a\lvert b} = \boldsymbol \mu_a - \boldsymbol \Lambda_{aa}^{-1}\boldsymbol \Lambda_{ab}(\boldsymbol x_b - \boldsymbol \mu_b)$
- Or, we can set $\boldsymbol K = \boldsymbol \Sigma_{ab}\boldsymbol \Sigma_{bb}^{-1}$, where we have
<div>
\begin{equation*}
\begin{aligned}
    &\boldsymbol \mu_{a|b} = \boldsymbol \mu_a + \boldsymbol K(\boldsymbol x_b - \boldsymbol \mu_b)  \qquad \begin{aligned}[t]
        \boldsymbol \Sigma_{a|b} &= \boldsymbol \Sigma_{aa} - \boldsymbol K\boldsymbol \Sigma_{bb}\boldsymbol K^T \\
        &= \boldsymbol \Sigma_{aa} - \boldsymbol \Sigma_{ab}\boldsymbol \Sigma_{bb}^{-1}\boldsymbol \Sigma_{ba} \\
    \end{aligned}
\end{aligned}
\end{equation*}
</div>
Please note that this follows from <a href="#prop-1">block-matrix inverse result</a> (you can try plugging the results in).

*Proof*: We consider the expansion of the conditional distribution, as we have:
<div>
\begin{equation*}
\begin{aligned}
    -\frac{1}{2}(\boldsymbol x_a - \boldsymbol \mu_{a|b})^T\boldsymbol \Sigma_{a|b}^{-1}(\boldsymbol x_a - \boldsymbol \mu_{a|b}) = {\color{red}-\frac{1}{2}\boldsymbol x_a^T\boldsymbol \Sigma_{a|b}^{-1}\boldsymbol x_a} + {\color{blue}\boldsymbol x_a^T\boldsymbol \Sigma_{a|b}^{-1}\boldsymbol \mu_{a|b}} + \text{const}
\end{aligned}
\end{equation*}
</div>
Let's consider the values, which should be equal to equation $\eqref{eqn:1}$, as we can see that:
- The <font color="#f44336">red term</font>: we set $\boldsymbol \Sigma_{a\lvert b}^{-1} = \boldsymbol \Lambda_{aa}$ (consider the first <font color="#009688">green term</font> of the equation $\eqref{eqn:1}$)
- The <font color="#2196f3">blue term</font>., we will have to consider $(\dots)^T\boldsymbol x_a$. We have:
<div>
\begin{equation*}
\begin{aligned}
{\color{green} \boldsymbol \mu_{a}^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} &{\color{yellow} - \frac{1}{2}\boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol x_b + \frac{1}{2} \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b} {\color{purple} - \frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a + \frac{1}{2}\boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a} \\
&= \boldsymbol x_a^T\Big[ \boldsymbol \Lambda_{aa}\boldsymbol \mu_a - \boldsymbol \Lambda_{ab}\boldsymbol x_b + \boldsymbol \Lambda_{ab}\boldsymbol \mu_b \Big] = \boldsymbol x_a^T\Big[ \boldsymbol \Lambda_{aa}\boldsymbol \mu_a - \boldsymbol \Lambda_{ab}(\boldsymbol x_b - \boldsymbol \mu_b) \Big] \\
\end{aligned}
\end{equation*}
</div>
Please note that $\boldsymbol \Lambda_{ab}^T = \boldsymbol \Lambda_{ba}$. Now, let's do "pattern" matching, as we have:
<div>
\begin{equation*}
\begin{aligned}
\boldsymbol x_a^T&\boldsymbol \Lambda_{aa}\boldsymbol \mu_{a|b} = \boldsymbol x_a^T\Big[ \boldsymbol \Lambda_{aa}\boldsymbol \mu_a - \boldsymbol \Lambda_{ab}(\boldsymbol x_b - \boldsymbol \mu_b) \Big] \\
\implies&\boldsymbol \mu_{a|b} \begin{aligned}[t]
    &= \boldsymbol \Lambda_{aa}^{-1}\boldsymbol \Lambda_{aa}\boldsymbol \mu_a - \boldsymbol \Lambda_{aa}^{-1}\boldsymbol \Lambda_{ab}(\boldsymbol x_b - \boldsymbol \mu_b) \\
    &= \boldsymbol \mu_a - \boldsymbol \Lambda_{aa}^{-1}\boldsymbol \Lambda_{ab}(\boldsymbol x_b - \boldsymbol \mu_b) \\
\end{aligned} 
\end{aligned}
\end{equation*}
</div>
Thus the proof is complete. 
$\tag*{$\Box$}$


<span class="anchor" id="prop-2"></span>**Proposition (Marginalisation)**: Consider the following Gaussian 
<div>
\begin{equation*}
\begin{bmatrix}
    \boldsymbol x_a \\ \boldsymbol x_b
\end{bmatrix}
\sim p(\boldsymbol x_a, \boldsymbol x_b) = \mathcal{N}\bracka{
\begin{bmatrix}
    \boldsymbol \mu_a \\ \boldsymbol \mu_b
\end{bmatrix}, \begin{bmatrix}
    \boldsymbol \Sigma_{aa} & \boldsymbol \Sigma_{ab} \\ 
    \boldsymbol \Sigma_{ba} & \boldsymbol \Sigma_{bb} \\ 
\end{bmatrix}}
\end{equation*}
</div>
We can show that: $p(\boldsymbol x_a) = \mathcal{N}(\boldsymbol x | \boldsymbol \mu_{a}, \boldsymbol \Sigma_{aa})$


*Proof*: We collect the terms that contains $\boldsymbol x_b$ so that we can integrate it out, as we have:
<div>
\begin{equation*}
\begin{aligned}
{\color{grey} -\frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b} &{\color{grey} +} {\color{grey}\boldsymbol \mu_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b}
{\color{purple} -\frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a + \frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{ba} \boldsymbol \mu_a} 
{\color{yellow} -\frac{1}{2}\boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol x_b+\frac{1}{2}\boldsymbol \mu_a^T\boldsymbol \Lambda_{ab}\boldsymbol x_b} \\
&=  -\frac{1}{2}\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b + \boldsymbol \mu_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b -\boldsymbol x_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a + \boldsymbol x_b^T\boldsymbol \Lambda_{ba} \boldsymbol \mu_a \\
&=  -\frac{1}{2}\Big[\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b - 2\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol \Lambda_{bb}^{-1}\underbrace{\Big(\boldsymbol \Lambda_{bb}\boldsymbol \mu_b -\boldsymbol \Lambda_{ba}(\boldsymbol x_a - \boldsymbol \mu_a)\Big)}_{\boldsymbol m}\Big] \\
&=  -\frac{1}{2}\Big[\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol x_b - 2\boldsymbol x_b^T\boldsymbol \Lambda_{bb}\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m + (\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)^T\boldsymbol \Lambda_{bb}(\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m) \Big] + \frac{1}{2}(\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)^T\boldsymbol \Lambda_{bb}(\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)  \\
&=  -\frac{1}{2}(\boldsymbol x_b - \boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)^T\boldsymbol \Lambda_{bb}(\boldsymbol x_b - \boldsymbol \Lambda_{bb}^{-1}\boldsymbol m) + {\color{blue}\frac{1}{2}\boldsymbol m^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m}  \\
\end{aligned}
\end{equation*}
</div>
If we integrate our the quantity, to be:
<div>
\begin{equation*}
\int \exp\brackc{-\frac{1}{2}(\boldsymbol x_b - \boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)^T\boldsymbol \Lambda_{bb}(\boldsymbol x_b - \boldsymbol \Lambda_{bb}^{-1}\boldsymbol m)}\dby \boldsymbol x_b
\end{equation*}
</div>
We can use the Gaussian integration, like in part 1 and part 2. Now, we consider the other terms that doesn't depends on $\boldsymbol x_b$ i.e all terms that depends on $\boldsymbol x_a$ together with the <font color="#2196f3">blue term</font> that been left out. 
<div>
\begin{equation*}
\begin{aligned}
{\color{blue}\frac{1}{2}\boldsymbol m^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol m} &{\color{green} -}{\color{green} \frac{1}{2}\boldsymbol x^T_a\boldsymbol \Lambda_{aa}\boldsymbol x_a + \boldsymbol \mu_a^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} {\color{yellow} + \frac{1}{2} \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b}{\color{purple}+ \frac{1}{2}\boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a} \\
&= \begin{aligned}[t]
    \frac{1}{2}\Big(&\boldsymbol \Lambda_{bb}\boldsymbol \mu_b -\boldsymbol \Lambda_{ba}(\boldsymbol x_a - \boldsymbol \mu_a)\Big)^T\boldsymbol \Lambda_{bb}^{-1}\Big(\boldsymbol \Lambda_{bb}\boldsymbol \mu_b -\boldsymbol \Lambda_{ba}(\boldsymbol x_a - \boldsymbol \mu_a)\Big) \\
    &{\color{green} - \frac{1}{2}\boldsymbol x^T_a\boldsymbol \Lambda_{aa}\boldsymbol x_a + \boldsymbol \mu_a^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} + \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b
\end{aligned} \\
&= \begin{aligned}[t]
    \frac{1}{2}\Big[ \boldsymbol \mu_b^T\boldsymbol \Lambda_{bb}\boldsymbol \mu_b &- (\boldsymbol x_a - \boldsymbol \mu_a)^T\boldsymbol \Lambda_{ba}^T\boldsymbol \mu_b \\
    &- \boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}(\boldsymbol x_a - \boldsymbol \mu_a) + (\boldsymbol x_a - \boldsymbol \mu_a)^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}(\boldsymbol x_a - \boldsymbol \mu_a) \Big] \\
    &{\color{green} - \frac{1}{2}\boldsymbol x^T_a\boldsymbol \Lambda_{aa}\boldsymbol x_a + \boldsymbol \mu_a^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} + \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b
\end{aligned} \\
&= \begin{aligned}[t]
    \frac{1}{2}\Big[ \boldsymbol \mu_b^T\boldsymbol \Lambda_{bb}\boldsymbol \mu_b &- \boldsymbol x_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \mu_b + \boldsymbol \mu_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \mu_b - \boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}\boldsymbol x_a + \boldsymbol \mu_b^T\boldsymbol \Lambda_{ba}\boldsymbol \mu_a \\
    &+ \boldsymbol x_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\boldsymbol x_a - \boldsymbol x_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\boldsymbol\mu_a - \boldsymbol \mu_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\boldsymbol x_a \\
    &+ \boldsymbol \mu_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\boldsymbol \mu_a \Big] {\color{green} - \frac{1}{2}\boldsymbol x^T_a\boldsymbol \Lambda_{aa}\boldsymbol x_a + \boldsymbol \mu_a^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} + \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b
\end{aligned} \\
&= \begin{aligned}[t]
    \frac{1}{2}\Big[-2\boldsymbol x_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \mu_b &+ \boldsymbol x_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\boldsymbol x_a - 2\boldsymbol x_a^T\boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\boldsymbol\mu_a\Big] \\
    &{\color{green} - \frac{1}{2}\boldsymbol x^T_a\boldsymbol \Lambda_{aa}\boldsymbol x_a + \boldsymbol \mu_a^T\boldsymbol \Lambda_{aa}\boldsymbol x_a} + \boldsymbol x_a^T\boldsymbol \Lambda_{ab}\boldsymbol \mu_b + \text{const}
\end{aligned} \\
&= \begin{aligned}[t]
    -\frac{1}{2}\boldsymbol x_a^T\Big[\boldsymbol \Lambda_{aa} - \boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\Big]\boldsymbol x_a &+ \boldsymbol x_a^T\Big[\boldsymbol \Lambda_{aa} - \boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba}\Big]\boldsymbol\mu_a + \text{const}
\end{aligned} \\
\end{aligned}
\end{equation*}
</div>
If we comparing this form to the normal quadratic expanision of Gaussian, we can set the $\boldsymbol \mu$ of marginalised Gaussian is $\boldsymbol \mu_a$, while the covariance is $(\boldsymbol \Lambda_{aa} - \boldsymbol \Lambda_{ba}^T\boldsymbol \Lambda_{bb}^{-1}\boldsymbol \Lambda_{ba})^{-1}$. If we compare this to the inverse matrix partition, we can see that this is equal to $\boldsymbol \Sigma_{aa}$. Thus complete the proof. 
$\tag*{$\Box$}$

**Proposition (Linear Gaussian Model)**: Consider the distribution to be: $p(\boldsymbol x) = \mathcal{N}(\boldsymbol x \vline {\color{yellow} \boldsymbol \mu} , {\color{blue} \boldsymbol \Lambda^{-1}})$ and $p(\boldsymbol y \vline \boldsymbol x) = \mathcal{N}(\boldsymbol y \vline {\color{purple}\boldsymbol A}\boldsymbol x + {\color{green} \boldsymbol b}, {\color{red} \boldsymbol L^{-1}})$. We can show that the following holds:
<div>
\begin{equation*}
p(\boldsymbol y) = \mathcal{N}(\boldsymbol y \vline {\color{purple}\boldsymbol A}{\color{yellow} \boldsymbol \mu} + {\color{green} \boldsymbol b}, {\color{red} \boldsymbol L^{-1}} + {\color{purple}\boldsymbol A}{\color{blue} \boldsymbol \Lambda^{-1}}{\color{purple}\boldsymbol A^T}) \qquad p(\boldsymbol x \vline \boldsymbol y) = \mathcal{N}\bracka{ \boldsymbol x \vline {\color{grey} \boldsymbol \Sigma}\brackc{ {\color{purple}\boldsymbol A^T} {\color{red} \boldsymbol L}(\boldsymbol y-{\color{green} \boldsymbol b}) + {\color{blue} \boldsymbol \Lambda}{\color{yellow} \boldsymbol \mu}}, {\color{grey} \boldsymbol \Sigma} }
\end{equation*}
</div>
where we have ${\color{grey} \boldsymbol \Sigma} = ({\color{blue} \boldsymbol \Lambda} + {\color{purple}\boldsymbol A^T}{\color{red} \boldsymbol L}{\color{purple}\boldsymbol A})^{-1}$

*Proof*: We will consider the joint random variable $\boldsymbol z = (\boldsymbol x, \boldsymbol y)^T$. Let's consider the joint distribution and the inside of exponential: 
<div>
\begin{equation*}
\begin{aligned}
-\frac{1}{2}&(\boldsymbol x - \boldsymbol \mu)^T\boldsymbol \Lambda(\boldsymbol x - \boldsymbol \mu) - \frac{1}{2}(\boldsymbol y - \boldsymbol A\boldsymbol x - \boldsymbol b)^T\boldsymbol L(\boldsymbol y - \boldsymbol A\boldsymbol x - \boldsymbol b) + \text{const} \\
&= \begin{aligned}[t]
    -\frac{1}{2}\Big[ \boldsymbol x^T\boldsymbol \Lambda\boldsymbol x &- 2\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol x + \boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol \mu + \boldsymbol y^T\boldsymbol L\boldsymbol y - \boldsymbol y^T\boldsymbol L\boldsymbol A\boldsymbol x - \boldsymbol y^T\boldsymbol L\boldsymbol b\\
    &-\boldsymbol x^T\boldsymbol A^T\boldsymbol L\boldsymbol y + \boldsymbol x^T\boldsymbol A^T \boldsymbol L \boldsymbol A\boldsymbol x + \boldsymbol x^T\boldsymbol A^T \boldsymbol L \boldsymbol b - \boldsymbol b^T\boldsymbol L\boldsymbol y +\boldsymbol b^T\boldsymbol L\boldsymbol A\boldsymbol x + \boldsymbol b^T\boldsymbol L\boldsymbol b \Big] + \text{const}
\end{aligned} \\
&= \begin{aligned}[t]
    -\frac{1}{2}\Big[ \boldsymbol x^T\boldsymbol \Lambda\boldsymbol x &- 2\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol x + \boldsymbol y^T\boldsymbol L\boldsymbol y - \boldsymbol y^T\boldsymbol L\boldsymbol A\boldsymbol x - \boldsymbol y^T\boldsymbol L\boldsymbol b\\
    &-\boldsymbol x^T\boldsymbol A^T\boldsymbol L\boldsymbol y + \boldsymbol x^T\boldsymbol A^T \boldsymbol L \boldsymbol A\boldsymbol x + \boldsymbol x^T\boldsymbol A^T \boldsymbol L \boldsymbol b - \boldsymbol b^T\boldsymbol L\boldsymbol y +\boldsymbol b^T\boldsymbol L\boldsymbol A\boldsymbol x  \Big] + \text{const}
\end{aligned} \\
&= \begin{aligned}[t]
    -\frac{1}{2}\Big[ \boldsymbol x^T\Big(\boldsymbol \Lambda + \boldsymbol A^T \boldsymbol L \boldsymbol A\Big)\boldsymbol x &+ \boldsymbol y^T\boldsymbol L\boldsymbol y - \boldsymbol y^T\boldsymbol L\boldsymbol A\boldsymbol x - \boldsymbol x^T\boldsymbol A^T\boldsymbol L\boldsymbol y\\
    &+ 2\boldsymbol x^T\boldsymbol A^T \boldsymbol L \boldsymbol b  - 2\boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol x - 2\boldsymbol y^T\boldsymbol L\boldsymbol b \Big] + \text{const}
\end{aligned} \\
&= \begin{aligned}[t]
    -\frac{1}{2}\Big[ \boldsymbol x^T\Big(\boldsymbol \Lambda + \boldsymbol A^T \boldsymbol L \boldsymbol A\Big)\boldsymbol x &+ \boldsymbol y^T\boldsymbol L\boldsymbol y - \boldsymbol y^T\boldsymbol L\boldsymbol A\boldsymbol x - \boldsymbol x^T\boldsymbol A^T\boldsymbol L\boldsymbol y\Big]\\
    &- \boldsymbol x^T\boldsymbol A^T \boldsymbol L \boldsymbol b + \boldsymbol \mu^T\boldsymbol \Lambda\boldsymbol x + \boldsymbol y^T\boldsymbol L\boldsymbol b  + \text{const}
\end{aligned} \\
&= \begin{aligned}[t]
    -\frac{1}{2}
    \begin{pmatrix}
        \boldsymbol x \\ \boldsymbol y
    \end{pmatrix}^T
    \begin{pmatrix}
        \boldsymbol \Lambda + \boldsymbol A^T\boldsymbol L\boldsymbol A & -\boldsymbol A^T\boldsymbol L \\
        -\boldsymbol L\boldsymbol A & \boldsymbol L
    \end{pmatrix}
    \begin{pmatrix}
        \boldsymbol x \\ \boldsymbol y
    \end{pmatrix}  + 
    \begin{pmatrix}
        \boldsymbol x \\ \boldsymbol y
    \end{pmatrix}^T
    \begin{pmatrix}
        \boldsymbol \Lambda\boldsymbol \mu - \boldsymbol A^T\boldsymbol L\boldsymbol b \\
        \boldsymbol L\boldsymbol b
    \end{pmatrix} + \text{const}
\end{aligned} \\
\end{aligned}
\end{equation*}
</div>
We can use the <a href="#prop-1">block-matrix inverse result</a>, to show that:
<div>
\begin{equation*}
\begin{aligned}
\begin{pmatrix}
    \boldsymbol \Lambda + \boldsymbol A^T\boldsymbol L\boldsymbol A & -\boldsymbol A^T\boldsymbol L \\
    -\boldsymbol L\boldsymbol A & \boldsymbol L
\end{pmatrix}^{-1} = 
\begin{pmatrix}
    \boldsymbol \Lambda^{-1} & \boldsymbol \Lambda^{-1}\boldsymbol A^T \\
    \boldsymbol A\boldsymbol \Lambda^{-1} & \boldsymbol L^{-1} + \boldsymbol A\boldsymbol \Lambda^{-1}\boldsymbol A^T
\end{pmatrix}
\end{aligned}
\end{equation*}
</div>
Recall the Gaussian pattern matching, we can see that the mean is equal to:
<div>
\begin{equation*}
\begin{aligned}
\begin{pmatrix}
    \boldsymbol \Lambda^{-1} & \boldsymbol \Lambda^{-1}\boldsymbol A^T \\
    \boldsymbol A\boldsymbol \Lambda^{-1} & \boldsymbol L^{-1} + \boldsymbol A\boldsymbol \Lambda^{-1}\boldsymbol A^T
\end{pmatrix}
\begin{pmatrix}
    \boldsymbol \Lambda\boldsymbol \mu - \boldsymbol A^T\boldsymbol L\boldsymbol b \\
    \boldsymbol L\boldsymbol b
\end{pmatrix} = \begin{pmatrix}
    \boldsymbol \mu \\ \boldsymbol A\boldsymbol \mu + \boldsymbol b
\end{pmatrix}
\end{aligned}
\end{equation*}
</div>
Please note that with <a href="#prop-2">marginalisation result</a>, it is obvious to see how this leads to the final result. Now, for the <a href="#prop-3">conditional result</a>, we have the usual result that $\boldsymbol \Sigma= \boldsymbol \Lambda_{xx} = (\boldsymbol \Lambda + \boldsymbol A^T\boldsymbol L\boldsymbol A)^{-1}$, for the mean:
<div>
\begin{equation*}
\begin{aligned}
    \boldsymbol \mu - &(\boldsymbol \Lambda + \boldsymbol A^T\boldsymbol L\boldsymbol A)^{-1}(\boldsymbol A^T\boldsymbol L)(-\boldsymbol y + A\boldsymbol \mu + \boldsymbol b) \\
    &= \boldsymbol \Sigma\boldsymbol A^T\boldsymbol L(\boldsymbol y - \boldsymbol b) + \boldsymbol \mu - \boldsymbol \Sigma\boldsymbol A^T\boldsymbol L\boldsymbol A\boldsymbol \mu
\end{aligned}
\end{equation*}
</div>
We want to show that $\boldsymbol \mu - \boldsymbol \Sigma\boldsymbol A^T\boldsymbol L\boldsymbol A\boldsymbol \mu = \boldsymbol \Sigma\boldsymbol \Lambda\boldsymbol \mu$. 
<div class="row">
  <div class="column">
    <i>LHS</i>: Apply <a href="#prop-4">inverse indentity</a>, where we consider the section highlight in <font color="#e91e62">pink</font>:
    <div>
    \begin{equation*}
    \begin{aligned}
        \boldsymbol \mu &- \boldsymbol \Sigma\boldsymbol A^T\boldsymbol L\boldsymbol A\boldsymbol \mu \\
        &= \boldsymbol \mu - {\color{pink}(\boldsymbol \Lambda + \boldsymbol A^T\boldsymbol L\boldsymbol A)^{-1}\boldsymbol A^T\boldsymbol L}\boldsymbol A\boldsymbol \mu \\
        &= \boldsymbol \mu - \boldsymbol \Lambda^{-1}\boldsymbol A^T(\boldsymbol A\boldsymbol \Lambda^{-1}\boldsymbol A^T + \boldsymbol L^{-1})^{-1}\boldsymbol A\boldsymbol \mu
    \end{aligned}
    \end{equation*}
    </div>
  </div>
  <div class="column">
    <i>RHS</i>: We consider the section highlight in <font color="#2196f3">blue</font> and use <a href="#prop-5">Woodbury Identity</a>:
    <div>
    \begin{equation*}
    \begin{aligned}
        \boldsymbol \Sigma\boldsymbol \Lambda\boldsymbol \mu &= {\color{blue}(\boldsymbol \Lambda + \boldsymbol A^T\boldsymbol L\boldsymbol A)^{-1}}\boldsymbol \Lambda\boldsymbol \mu \\
        &= \boldsymbol \mu - \boldsymbol \Lambda^{-1}\boldsymbol A^T(\boldsymbol A\boldsymbol \Lambda^{-1}\boldsymbol A^T + \boldsymbol L^{-1})^{-1}\boldsymbol A\boldsymbol \mu
    \end{aligned}
    \end{equation*}
    </div>
  </div>
</div> 
Now we have show that both are equal, thus we conclude the proof. 
$\tag*{$\Box$}$