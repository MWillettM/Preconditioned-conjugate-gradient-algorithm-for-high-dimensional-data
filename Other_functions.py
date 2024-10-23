import numpy as np

#Preconditioner generator - returns our preconditioner and its coefficients (the preconditioner is of the form aI + b e^Te)
#n = number of data points
#c = shape parameter for the RBFs - we are safe to use 1 here, but feel free to adjust to your liking.
#distribution - if your data is uniformly distributed in the unit hypercube/unit sphere, select accordingly. Otherwise we take k=1.

def precon_generator(n,c,RBF,distribution):
    if distribution == 'cube':
        k=1
    if distribution =='ball':
        k=np.sqrt(2)
    else:
        k=1

    if RBF == 'Gaussian':
        phi_rt2 = Gaussian(k,c)
    if RBF == 'MQ':
        phi_rt2 = MQ(k,c)
    if RBF == 'MQ_inv':
        phi_rt2 = MQ_inv(k,c)
    if RBF == 'Quadratic_inv':
        phi_rt2 = Quadratic_inv(k,c)
    const_1 = phi_rt2 / ((phi_rt2-c)*(c-phi_rt2+n*phi_rt2))
    const_2 = 1 / (phi_rt2-c)

    preconditioner = const_1 * np.ones((n,n)) - const_2 * np.eye(n)
    
    return preconditioner, const_1, const_2

#A function that evaluates our interpolant using 
#coefficients - generated from the PCG algorithm output
#centers - the same centers we used to generate the coefficients
#x_value - the point we want to evaluate our interpolant at
#RBF_type - make sure to use the same RBF type as the one used to generate the interpolant.

#returns the value of the interpolant.

def interp(coefficients, centers, x_value, RBF_type):
    if RBF_type == 'Gaussian':
        center_vector = [Gaussian(x_value - center,1) for center in centers]
    if RBF_type == 'MQ':
        center_vector = [MQ(x_value - center,1) for center in centers]
    if RBF_type == 'MQ_inv':
        center_vector = [MQ_inv(x_value - center,1) for center in centers]
    if RBF_type == 'Quadratic_inv':
        center_vector = [Quadratic_inv(x_value - center,1) for center in centers]
    
    return coefficients.T @ center_vector


#RBF functions - note that our algorithm is not certain to converge for the multiquadric (MQ) as the interpolation matrix generated is not positive definite. 
def MQ(point,c):
    return np.sqrt(c**2+np.linalg.norm(point))

def Gaussian(point, c):
    return np.exp(-c * np.linalg.norm(point))

def MQ_inv(point,c):
    return 1/np.sqrt(c**2+np.linalg.norm(point))

def Quadratic_inv(point,c):
    return 1/(c**2+np.linalg.norm(point))

#interpolation matrix generator
def interpolation_matrix_generator(RBF_type, data):
    phi = np.ones((len(data),len(data)))
    for i, point in enumerate(data):
        for j, center in enumerate(data):
            phi[i,j] = np.linalg.norm(point-center)

    if RBF_type == 'Gaussian':
        vectorized_function = np.vectorize(Gaussian)
    if RBF_type == 'MQ':
        vectorized_function = np.vectorize(MQ)
    if RBF_type == 'MQ_inv':
        vectorized_function = np.vectorize(MQ_inv)
    if RBF_type == 'Quadratic_inv':
        vectorized_function = np.vectorize(Quadratic_inv)

    return vectorized_function(phi,1)


#data generator for a range of underlying distributions.
def points_in_unit_ball(n,d):
    cube = np.random.standard_normal(size=(n, d))
    norms = np.linalg.norm(cube,axis=1)
    surface_sphere = cube/norms[:,np.newaxis]
    scales = np.random.uniform(0,1, size= n)
    points = surface_sphere * (scales[:, np.newaxis])**(1/d)
    return points

def points_integer_grid(n,d, s_min, s_max):
    points = np.random.randint(s_min,s_max, size = (n,d))
    return points

def points_in_cube(n,d,normalised):
    points = np.random.uniform(0,1,size=(n,d))
    if normalised == 'no':
        return points
    if normalised == 'yes':
        return points/np.sqrt(d/6-17/120)

def points_normal(n,d):
    points = np.random.normal(0,1,size=(n,d))
    return points

