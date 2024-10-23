import Precon_Conjugate_Gradient as PCG
import Other_functions as wt


#this takes data centers, values at those points, the kind of RBF you wish to use and a value, x, at which you wish your interpolant to be evaluated.
#this returns - iteration count for convergence of the CG algorithm, iterpolant coefficients and the interpolant evaluated at x.
#it is suggested that you scale your data so that |x_i|<1 to prevent the RBFs from being too flat.
#where appropriate, we use a shape parameter of 1 in the RBFs.

def interpolant_generator_evaluator(data_centers, data_values, RBF,x):
    x_i = [i for i in data_centers]
    
    n = len(x_i)

    f_i = [i for i in data_values]

    if n != len(f_i):
        print('data_centers and data_values different length')
        return
    
    else:

        #Generating the interpolation matrix, phi.
        phi = wt.interpolation_matrix_generator(RBF,x_i)

        #generating our preconditioner
        precon_inv, const_1, const_2 = wt.precon_generator(n,1,'Q','')

        #generating our interpolation coefficients
        solution, count = PCG.untransformed_Pc_GC(phi,f_i,precon_inv,1e-10,const_1,const_2)

        #evaluating our interpolant at x
        interpolant_eval = wt.interp(solution,x_i,x,RBF)

        return count, solution, interpolant_eval
    
