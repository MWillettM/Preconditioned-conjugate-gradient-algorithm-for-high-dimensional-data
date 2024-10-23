# Preconditioned-conjugate-gradient-algorithm-for-high-dimensional-data

The linear system of equations 

$A\mathbf{x} = \mathbf{b}$

naturally arises from the typical interpolation problem for scattered data with $n$ function values $f_{i}$ at the distinct points $\mathbf{x}_i \in \mathbb{R}^d$ where we wish to find an interpolant, $s(\mathbf{x})$ that satisfies the equations $s(\mathbf{x}_i) = f_i, \hspace{5mm} i = 1, ..., n.$

Specifically, we look to use interpolants of the form

$$ s(\mathbf{x}) = \sum^n_{i=1}\lambda_i{\phi({\|\mathbf{x}-\mathbf{x}_i\|})}, \hspace{5mm} \mathbf{x} \in \mathbb{R}^d $$

We generate the coefficients $\lambda_i$ from the data centers and knowledge of the RBF we wish to use. 

A common method for doing so is the Conjugate Gradient method (note that this only works for RBFs that generate a positive definite matrix e.g. the inverse multiquadric, the gaussian, the quadratic inverse, etc). This method benefits from 'preconditioning' (basically pre-multiplying our intial equation to 'flatten' the spectrum of the interpolation matrix which should massively speed out the convergence rate). 

We use a novel preconditioner that is highly effective for data drawn from high dimensional space - you can judge for yourself in the demo. 

We use an efficient implementation of the 'untransformed preconditioned conjugate gradient algorithm' (see section 12 of An Introduction to the Conjugate Gradient Method Without the Agonizing Pain) along with an unpreconditioned classic implementation of the conjugate gradient method. We also have a short bit of code to evaluate your interpolant as well.

Typical workflow:
1. Generate data centers and function values yourself, or use the code in Other_functions. It is generally suggested that you normalise your data centers so that the RBFs are not too 'flat'.

2. Generate the interpolation matrix - you can use the interpolation_matrix_generator in Other_functions

3. Generate preconditioner - use precon_generator in Other_functions

4. Use untransformed_Pc_GC to find our vector of coefficients

5. You can then use interp in Other_functions to evaluate your interpolant anywhere, using the data centers and coefficients from earlier. 


