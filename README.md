# Preconditioned-conjugate-gradient-algorithm-for-high-dimensional-data

The linear system of equations 
\begin{equation}
A\bx = \mathbf{b}
\label{eq: Classic Linear System}
\end{equation}

naturally arises from the typical interpolation problem for scattered data with $n$ function values $f_{i}$ at the distinct points $\mathbf{x}_i \in \mathbb{R}^d$ where we wish to find an interpolant, $s(\mathbf{x})$ that satisfies the equations
\[    s(\mathbf{x}_i) = f_i, \hspace{5mm} i = 1, ..., n.
\]

Specifically, we look to use interpolants of the form
\[    s(\mathbf{x}) = \sum^n_{i=1}\lambda_i{\phi({\|\mathbf{x}-\mathbf{x}_i\|})}, \hspace{5mm} \mathbf{x} \in \mathbb{R}^d
    \label{interp definition}
\]
