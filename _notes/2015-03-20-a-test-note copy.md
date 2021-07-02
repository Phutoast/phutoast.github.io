---
layout: post
title: "A Paper Note Title"
description: "This is the test note"
comments: true
keywords: "dummy content, lorem ipsum"
topic: "Learning Theory"
conf: "NeurIPS"
---

This post is based from the book: Pattern Recognition and Machine Learning, where we aim to give and expands most proofs and interesting results regarding Gaussian distributions, together with some results that makes the complete picture. 

## Gaussian Definition and Fundamental Integral

**Definition (Single Variable Gaussian Distribution)**: The Gaussian distribution is defined as 

<div>
\begin{equation*}
\newcommand{\dby}{\ \mathrm{d}}\newcommand{\argmax}[1]{\underset{#1}{\arg\max \ }}\newcommand{\argmin}[1]{\underset{#1}{\arg\min \ }}\newcommand{\const}{\text{const.}}\newcommand{\bracka}[1]{\left( #1 \right)}\newcommand{\brackb}[1]{\left[ #1 \right]}\newcommand{\brackc}[1]{\left\{ #1 \right\}}\newcommand{\brackd}[1]{\left\langle #1 \right\rangle}\newcommand{\correctquote}[1]{``#1''}\newcommand{\norm}[1]{\left\lVert#1\right\rVert}\newcommand{\abs}[1]{\left|#1\right|}
    p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left\{ -\frac{1}{2\sigma^2}(x-\mu)^2\right\}
\end{equation*}
</div>

where $\mu$ is a parameter called mean, while $\sigma$ is a parameter called standard derivation. However, we also define $\sigma^2$ as variance. Finally, the standard normal distribution is Gaussian distribution with mean $0$ and variance $1$. 

**Proposition (Gaussian integral)**: The integration of $\exp(-x^2)$ is equal to

<div>
\begin{equation*}
    \int^\infty_{-\infty}\exp(-x^2)\dby x = \sqrt{\pi}
\end{equation*}
</div>

We can use this identity to find the normalizing factor of Gaussian distribution.

_Proof_: We will transform the problem into $2$D polar coordinate system, which make the integration easier. 

<div>
\begin{equation*}
\begin{aligned}
    \bracka{\int^\infty_{-\infty}\exp(-x^2)\dby x}^2 &= \bracka{\int^\infty_{-\infty}\exp(-x^2)\dby x}\bracka{\int^\infty_{-\infty}\exp(-y^2)\dby y} \\
&= \int^\infty_{-\infty}\int^\infty_{-\infty}\exp\bracka{-(x^2+y^2)}\dby x\dby y \\
&= \int^{2\pi}_{0}\int^\infty_{0}\exp\bracka{-r^2}r\dby r\dby\theta \\
&= 2\pi\int^\infty_{0}\exp\bracka{-r^2}r\dby r = \pi\int^{0}_{-\infty} \exp\bracka{u}\dby u = \pi
\end{aligned}
\end{equation*}
</div>

Please note that we use $u$-substution with $-r^2$, in the last step. $\tag*{$\Box$}$

<span class="anchor" id="corollary-1"></span>**Corollary (Normalization of Gaussian Distribution)**: We consider the integration
<div>
\begin{equation*}
    \int^\infty_{-\infty}\exp\left\{ -\frac{(x-\mu)^2}{2\sigma^2}\right\} \dby x = \sqrt{2\pi\sigma^2}
\end{equation*}
</div>

*Proof*: Starting by setting $z=x-\mu$, which we have:
<div>
\begin{equation*}
\begin{aligned}
    \int^\infty_{-\infty}\exp\left\{ -\frac{z^2}{2\sigma^2}\right\} \dby z &= \sigma\sqrt{2} \int^\infty_{-\infty}\frac{1}{\sqrt{2}\sigma}\exp\left\{ -\frac{z^2}{2\sigma^2}\right\} \dby z \\
&= \sigma\sqrt{2} \int^\infty_{-\infty} \exp(-y^2)\dby = \sigma\sqrt{2\pi}
\end{aligned}
\end{equation*}
</div>
where we have $y = z/(\sqrt{2}\sigma)$, and we finishes the proof. $\tag*{$\Box$}$

## Statistics of Single Variable Gaussian Distribution

**Definition (Odd Function)**: The function $f(x)$ is an odd function iff $f(-x) = -x$

<span class="anchor" id="lemma-1"></span>**Lemma (Odd Function Integration):** The integral over $[-a, a]$, where $a\in\mathbb{R}^+$ of an odd function $f(x)$ is $0$ i.e
<div>
\begin{equation*}\int^a_{-a}f(x)\dby x = 0\end{equation*}
</div>

_Proof_: We can see that:
<div>
\begin{equation*}
\begin{aligned}
    \int^a_{-a}f(x)\dby x &= \int^0_{-a}f(x)\dby x + \int^a_{0}f(x)\dby x \\
    &= \int^a_{0}f(-x)\dby x + \int^a_{0}f(x)\dby x \\
    &= -\int^a_{0}f(x)\dby x + \int^a_{0}f(x)\dby x = 0
\end{aligned}
\end{equation*}
</div>

Thus complete the proof $\tag*{$\Box$}$

**Proposition (Mean of Gaussian)**: The expectation $\mathbb{E}_{x}[x]$ of nornally distributed Gaussian is $\mu$

_Proof_: Let's consider the following integral 
<div>
\begin{equation*}
\begin{aligned}
    \frac{1}{\sqrt{2\pi\sigma^2}}\int^\infty_{-\infty} \exp\bracka{-\frac{(x-\mu)^2}{2\sigma^2}}x\dby x
\end{aligned}
\end{equation*}
</div>
We will set $z = x - \mu$, where we have:
<div>
\begin{equation*}
\begin{aligned}
    \frac{1}{\sqrt{2\pi\sigma^2}}&\int^\infty_{-\infty} \exp\bracka{-\frac{z^2}{2\sigma^2}}(z+\mu)\dby x \\
    &= \frac{1}{\sqrt{2\pi\sigma^2}}\Bigg[ \underbrace{\int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}z\dby z}_{I_1} + \underbrace{\int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}\mu\dby z}_{I_2} \Bigg]
\end{aligned}
\end{equation*}
</div>
Let's consider $I_1$, where it is clear that 
<div>
\begin{equation*}
    g(x) = \exp\bracka{-\frac{z^2}{2\sigma^2}}z = -\bracka{-\exp\bracka{-\frac{(-z)^2}{2\sigma^2}}z} = -g(-x)
\end{equation*}
</div>
Thus the function $g(x)$ is and odd function. Therefore, making the integration $I_1$ vanishes to $0$. Please see the <a href="#lemma-1">Lemma</a> above for the proof. Now, for the second integration, we can simply recall the <a href="#corollary-1">normalization result</a> of the Gaussian, where
<div>
\begin{equation*}
    \int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}\mu\dby z = \mu\int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}\dby z = \mu\sqrt{2\pi\sigma^2}
\end{equation*}
</div>
Finally, we have
<div>
\begin{equation*}
\begin{aligned}
    \frac{1}{\sqrt{2\pi\sigma^2}}\int^\infty_{-\infty} \exp\bracka{-\frac{z^2}{2\sigma^2}}(z+\mu)\dby x = \frac{1}{\sqrt{2\pi\sigma^2}}\Bigg[ 0 + \mu\sqrt{2\pi\sigma^2} \Bigg] = \mu
\end{aligned}
\end{equation*}
</div>
Thus complete the proof. $\tag*{$\Box$}$

**Lemma**: The variance $\operatorname{var}(x) = \mathbb{E}[(x - \mu)^2]$ is equal to $\mathbb{E}[x^2] - \mathbb{E}[x]^2$

_Proof_: This is an application of expanding the definition
<div>
\begin{equation*}
\begin{aligned}
    \mathbb{E}[(x - \mu)^2] &= \mathbb{E}[x^2 -2x\mu + \mu^2] \\
    &= \mathbb{E}[x^2] - 2\mathbb{E}[x]\mu + \mathbb{E}[\mu^2] \\
    &= \mathbb{E}[x^2] - 2\mathbb{E}[x]^2 + \mathbb{E}[x]^2 \\
    &= \mathbb{E}[x^2] - \mathbb{E}[x]^2 \\
\end{aligned}
\end{equation*}
</div>
$\tag*{$\Box$}$

**Proposition**: The variance of normal distribution is $\sigma^2$

_Proof_: Let's consider the following equation, where we set $z = x-\mu$:
<div>
\begin{equation*}
\begin{aligned}
    \frac{1}{\sqrt{2\pi\sigma^2}}& \int^\infty_{-\infty} \exp\bracka{-\frac{(x-\mu)^2}{2\sigma^2}}x^2\dby x  = \frac{1}{\sqrt{2\pi\sigma^2}} \int^\infty_{-\infty} \exp\bracka{-\frac{z^2}{2\sigma^2}}(z+\mu)^2\dby z \\
    &=\frac{1}{\sqrt{2\pi\sigma^2}}\brackb{\int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}z^2\dby z + \int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}2\mu z\dby z + \int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}\mu^2\dby z } \\
    &=\frac{1}{\sqrt{2\pi\sigma^2}}\brackb{\int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}z^2\dby z + 0 + \mu^2\sqrt{2\pi\sigma^2} \dby z }
\end{aligned}
\end{equation*}
</div>
Now let's consider the first integral, please note that
<div>
\begin{equation*}
    \frac{d}{dz} \exp\bracka{-\frac{z^2}{2\sigma^2}} = \exp\bracka{-\frac{z^2}{2\sigma^2}}\bracka{-\frac{z}{\sigma^2}}
\end{equation*}
</div>
So we can perform an integration by-part considering $u=z$ and $dv = \exp\bracka{-\frac{z^2}{2\sigma^2}}\bracka{-\frac{z}{\sigma^2}}$:
<div>
\begin{equation*}
\begin{aligned}
    \int^\infty_{-\infty}&\exp\bracka{-\frac{z^2}{2\sigma^2}}z^2\dby z = -\sigma^2\int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}\bracka{-\frac{z}{\sigma^2}}z\dby z \\
    &= -\sigma^2\brackb{\left.z\exp\bracka{-\frac{z^2}{2\sigma^2}}\right|^\infty_{-\infty} - \int^\infty_{-\infty}\exp\bracka{-\frac{z^2}{2\sigma^2}}\dby z} \\
    &= -\sigma^2[0 - \sqrt{2\pi\sigma^2}] = \sigma^2\sqrt{2\pi\sigma^2}
\end{aligned}
\end{equation*}
</div>
To show that the evaluation on the left-hand side is zero, we have
<div>
\begin{equation*}
\begin{aligned}
    \lim_{z\rightarrow\infty} z\exp\bracka{-\frac{z^2}{2\sigma^2}} &- \lim_{z\rightarrow-\infty} z\exp\bracka{-\frac{z^2}{2\sigma^2}} \\
    &= 0 - \lim_{z\rightarrow-\infty}1\cdot\exp\bracka{-\frac{x^2}{2\sigma^2}}\bracka{-\frac{\sigma^2}{z}} \\
    &= 0-0 = 0
\end{aligned}
\end{equation*}
</div>
The first equality comes from L'Hospital's rule. Combinding the results:
<div>
\begin{equation*}
\begin{aligned}
    \mathbb{E}[x^2] &- \mathbb{E}[x]^2 = \sigma^2 + \mu^2 - \mu^2 = \sigma^2
\end{aligned}
\end{equation*}
</div>
Thus complete the proof. $\tag*{$\Box$}$


