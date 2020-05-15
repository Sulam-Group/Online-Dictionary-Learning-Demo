# Online Dictionary Learning demo

This noteboook is intended to provide a simple pytorch implementation for dictionary learning (DL) based on stochastic gradient descent. This is in essence a multi-purpose DL method that I put together in the midst of my own work, and I frequently use to carry out simple experiments and tests.
This implementation does not follow any particular publication, though it is perhaps closest to the OSDL work on sparse dictionaries (without the double-sparisy component), or to the Online Dictionary Learning algorithm by Mairal (though without their convex surrogate function approach).

Look at Description.ipynb for the general description of the algorithm.

------

### Basic algorithm description
The Dictionary Learning problem is concerned with

![\min_{\gamma_i,D} \sum_{i=1}^N \|y_i - D\gamma_i\|_2^2 \+  \lambda\ g(\gamma_i), \quad s.t. \quad \ \|d_j\|_2 = 1, \forall j](https://render.githubusercontent.com/render/math?math=%5Cmin_%7B%5Cgamma_i%2CD%7D%20%5Csum_%7Bi%3D1%7D%5EN%20%5C%7Cy_i%20-%20D%5Cgamma_i%5C%7C_2%5E2%20%5C%2B%20%20%5Clambda%5C%20g(%5Cgamma_i)%2C%20%5Cquad%20s.t.%20%5Cquad%20%5C%20%5C%7Cd_j%5C%7C_2%20%3D%201%2C%20%5Cforall%20j)

where <img src="https://render.githubusercontent.com/render/math?math=g(\gamma_i)"> is a spase-enforcing penalty term such as the <img src="https://render.githubusercontent.com/render/math?math=\ell_1"> norm or the <img src="https://render.githubusercontent.com/render/math?math=\ell_0"> pseudo-norm. Both will be considered in this implementation. 

Note that the sum of the reconstruction error per sample can be denoted in matrix form as <img src="https://render.githubusercontent.com/render/math?math=\|Y - D\Gamma\|^2_F">, where matrices <img src="https://render.githubusercontent.com/render/math?math=Y"> and <img src="https://render.githubusercontent.com/render/math?math=Gamma"> have the vectors <img src="https://render.githubusercontent.com/render/math?math=y_i"> and <img src="https://render.githubusercontent.com/render/math?math=\gamma_i"> in their columns, respectively.

Most dictionary learning methods employ an alternating minimization approach to address the above non-convex problem, by alternating between:
* Minimizing the objective w.r.t. ![\gamma_i](https://render.githubusercontent.com/render/math?math=%5Cgamma_i) while keeping the dictionary fixed, termed **Sparse Coding**, and
* Minimizing the objective w.r.t. *D*, while keeping the representations ![\gamma_i](https://render.githubusercontent.com/render/math?math=%5Cgamma_i) fixed, termed **Dictionary Update**.

We will employ such alternating approach as well. 

More broadly, one could apply a **batch** scheme: perform sparse coding _on all_ training examples, and then update $D$ accordingly (and iterate these steps). Alternatively, one might employ a **stochastic optimization** approach, and minimize the loss above one sample (or one mini-batch) at a time. We will employ this latter online implementation.

#### Sparse Coding
When minimizing for every representation $\gamma_i$ (with the dictionary $D$ being fixed), this implementation allows for two sparse-enforcing penalties:
* When the sparsity penalty function is the $\ell_1$ norm, the problem to be minimized is

![\min_{\gamma_i} \|y_i - D\gamma_i\|_2^2 + \lambda \ \|\gamma_i\|_1 \ \forall i](https://render.githubusercontent.com/render/math?math=%5Cmin_%7B%5Cgamma_i%7D%20%5C%7Cy_i%20-%20D%5Cgamma_i%5C%7C_2%5E2%20%2B%20%5Clambda%20%5C%20%5C%7C%5Cgamma_i%5C%7C_1%20%5C%20%5Cforall%20i)

and we employ the Fast Iterative Threhsolding Algorithm from Beck and Teboulle, or FISTA for short.

* When the sparsity penalty funcition is the non-convex and non-smooth L0 pseudo-norm, we opt for a constraint formulation and we minimize:

![\min_{\gamma_i} \|y_i - D\gamma_i\|_2^2 \quad s.t. \quad \|\gamma_i\|_0 \leq k \ \forall i,](https://render.githubusercontent.com/render/math?math=%5Cmin_%7B%5Cgamma_i%7D%20%5C%7Cy_i%20-%20D%5Cgamma_i%5C%7C_2%5E2%20%5Cquad%20s.t.%20%5Cquad%20%5C%7C%5Cgamma_i%5C%7C_0%20%5Cleq%20k%20%5C%20%5Cforall%20i%2C)

and we employ the Iterative Hard Tresholding method. Generally speaking, this L0 approach may lead to higher number of "dead" filters (atoms that are not used nor trained), which is typically solved by introducing other simple exteneral regularization techniques (replacing unused and repeated atoms, progressively reducing the target cardinality through training, etc).

#### Dictionary Update
After having found all $\gamma_i$ for each $y_i$ in the mini-batch, the dictionary update problem is concerned with

![\min_{D} \|Y_i - D\Gamma_i\|_2^2 \quad s.t. \quad \ \|d_j\|_2 = 1, \forall j.](https://render.githubusercontent.com/render/math?math=%5Cmin_%7BD%7D%20%5C%7CY_i%20-%20D%5CGamma_i%5C%7C_2%5E2%20%5Cquad%20s.t.%20%5Cquad%20%5C%20%5C%7Cd_j%5C%7C_2%20%3D%201%2C%20%5Cforall%20j.)

Ignoring for a moment the L2 constraint on the dictionary atoms, one could minimize this L2 loss with the Least Squares solution. In favor or a less severe minimization (note, this is still an alternating minimization approach for a non-convex problem), we simply perform a grandient step so as to minimize this norm, followed by a renormalization of the atoms to unit norm.


This Sparse Coding and Dictionary Update steps are iterated, every time for a different mini-batch, with stochastic gradient descent (with momentum).
