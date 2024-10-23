import numpy as np
import Precon_Conjugate_Gradient as PCG
import Other_functions as wt
import random


#this demo version illustrates the increasing effectiveness of our preconditioner as the dimension of the underlying data grows. 
#for our demo, we use the Quadratic Inverse RBF and draw from the multidimensional normal distribution, scaled appropriately. 

n=1000

for d in[2,3,6,10,25,100,158,1000,3981,10000,25118,63095]:
    x_i = wt.points_normal(n,d)/(np.sqrt(2*d-1))
    
    f_i = [random.random() for _ in range(n)]

    #Generating the interpolation matrix, phi.
    phi = wt.interpolation_matrix_generator('Quadratic_inv',x_i)

    #generating our preconditioner
    precon_inv, const_1, const_2 = wt.precon_generator(n,1,'Quadratic_inv','')

    #preconditioned solution
    solution, count = PCG.untransformed_Pc_GC(phi,f_i,precon_inv,1e-10,const_1,const_2)

    #unpreconditioned solution
    solution2,count2 = PCG.CG_no_precon(phi,f_i,1e-10)

    print('dimension:',d,'Preconditioned Iteration count:',count, 'unpreconditioned iteration count:',count2)