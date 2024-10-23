import numpy as np

#Setup for 'Untransformed Preconditioned Conjugate Gradient Method' to solve
#Ax=b with a preconditioner M, which gives us
#inv(M)Ax = inv(M)b


#A is the interpolation matrix
#x is the solution
#M is the preconditioner
#precon_inv can be generated from Other_functions.precon_generator, or feel free to use your own - you will need to adjust the code for the action of your preconditioner on a vector.

#See README for full description of the algorithm.

#Returns the coefficients of the interpolant and the iteration count.

def untransformed_Pc_GC(A,b,precon_inv,tolerance,c1,c2):
    
    #Setup
    n=np.size(b)
    initial_guess = np.zeros(n)
    residual = b - A@initial_guess

    #efficient calculation of our preconditioner's action on a vector
    def precon_action_on_vector(const_1,const_2,vector):
        vector_of_ones = np.ones(len(vector))
        return const_1*np.sum(vector) * vector_of_ones - const_2 * vector

    search_direction = precon_action_on_vector(c1,c2,residual)

    #x will be our solution
    x = initial_guess

    dummy_variable = residual.T @ search_direction

    #iteration count
    k=0
    #Implementation
    while np.linalg.norm(residual) > tolerance:
        k+=1
        #dummy variable is used twice to avoid recalculation
        
        step_size = dummy_variable / (search_direction.T @ A @ search_direction)
        x += step_size * search_direction

        residual -= step_size * A@search_direction

        new_dummy = residual.T @ precon_action_on_vector(c1,c2,residual)

        scale_factor = new_dummy / dummy_variable

        dummy_variable = new_dummy

        search_direction = precon_action_on_vector(c1,c2,residual)+ scale_factor*search_direction

        if k> 1000:
            break

    return x,k

#Unpreconditioned algorithm for comparison

def CG_no_precon(A,b,tolerance):

    #Setup
    n=np.size(b)
    initial_guess = np.zeros(n)
    residual = b - A@initial_guess
    search_direction = residual

    #x will be our solution
    x = initial_guess

    dummy_variable = residual.T @ residual

    #iteration coun
    k=0
    #Implementation
    while np.linalg.norm(residual) > tolerance:
        k+=1
        #dummy variable is used twice to avoid recalculation

        step_size = dummy_variable / (search_direction.T @ A @ search_direction)
        x += step_size * search_direction

        residual -= step_size * A@search_direction

        new_dummy = residual.T @ residual

        scale_factor = new_dummy / dummy_variable

        dummy_variable = new_dummy

        search_direction = residual + scale_factor*search_direction

        if k>1000:
            break

    return x,k
