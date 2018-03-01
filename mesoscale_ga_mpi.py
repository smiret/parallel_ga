################################################################################################################################################
#Sorting Function

def sort1(Pivec):

    # a3 = 0.0
    # a4 = 0.0

    for ww in range(0, len(Pivec)):

        for tt in range(ww, 0, -1):
            if (Pivec[tt,0] < Pivec[tt-1,0]):
                a3 = np.copy(Pivec[tt,:])
                a4 = np.copy(Pivec[tt-1,:])

                # print a3
                # print a4

                Pivec[tt-1,:] = a3
                Pivec[tt, :] = a4

            #     print a3
            #     print a4
            #
            # print tt
            # print Pivec


    return Pivec



################################################################################################################################################
#sphere generator

def sphere_gen(volume_fraction, sphere_number):

    import numpy as np


    #sphere_number_float = np.float_(sphere_number)
    sphere_table = np.zeros((sphere_number,4))

    sphere_radius = ((volume_fraction/sphere_number)*(3.0/(4.0*np.pi)))**(1.0/3.0)

    sphere_table[:,0] = np.random.uniform(low=0.0, high=1.0, size=(sphere_number))
    sphere_table[:,1] = np.random.uniform(low=0.0, high=1.0, size=(sphere_number))
    sphere_table[:,2] = np.random.uniform(low=0.0, high=1.0, size=(sphere_number))
    sphere_table[:,3] = sphere_radius

    count = 0

    for i in range(0,sphere_number-1):

        sphere_i = sphere_table[i]

        for j in range(i+1, sphere_number):

            sphere_j = sphere_table[j]

            #Check if sphere_j is inside sphere_i

            if ((sphere_i[0]-sphere_j[0])**2.0+ (sphere_i[1]-sphere_j[1])**2.0+
                        (sphere_i[2]-sphere_j[2])**2.0)**0.5 < 2.0*sphere_radius:

                sphere_table[j,0:3] =np.random.rand(3)

                count += 1

            if count > 2000:

                break

            i = 0 #reset the loop by setting the looping variable back to 0

    return sphere_table




################################################################################################################################################




import numpy as np
from effective_prop_fem_dec1_2017 import effective_prop_fem
#import sphere_generator
from datetime import datetime
from mpi4py import MPI
startTime = datetime.now()


#Step 1: Initiate a set of sphere table for different microstructures



string_size = 100
volume_fraction = 0.05
number_spheres = 100
fortran_table_size = 150

sphere_radius = ((volume_fraction / number_spheres) * (3.0 / (4.0 * np.pi))) ** (1.0 / 3.0)

sphere_table_matrix = np. zeros((string_size,fortran_table_size,4))

#Initiate MPI environment
comm = MPI.COMM_WORLD
rank = int(comm.Get_rank())
size = int(comm.Get_size())

process_size = int(string_size/size)
sendbuf_scatter = None



#sphere_table_matrix[0] = np.ones((150,4))

if rank == 0:

    for l in range(0, string_size):

        sphere_table = sphere_gen(volume_fraction, number_spheres)

        sphere_table_matrix[l, 0:number_spheres ,0] = sphere_table[:,0]
        sphere_table_matrix[l, 0:number_spheres, 1] = sphere_table[:,1]
        sphere_table_matrix[l, 0:number_spheres, 2] = sphere_table[:,2]
        sphere_table_matrix[l, 0:number_spheres, 3] = sphere_radius

#sphere_table_matrix[:, 55:150,:] = 0.0


#Put in the Material Properties (from previous Genetic Algorithm assuming isotropic bounds)

#Fiber and Matrix
k11_sphere = 100.11529332108428 #W/mk
k22_sphere = 100.11529332108428 #W/mk
k33_sphere = 100.11529332108428 #W/mk

k_sphere_mat = np.array([[k11_sphere, 0.0, 0.0],
                        [0.0, k22_sphere, 0.0],
                        [0.0, 0.0, k33_sphere]])

k11_matrix = 94.380986720059013
k22_matrix = 94.380986720059013
k33_matrix = 94.380986720059013

k_matrix_mat = np.array([[k11_matrix, 0.0, 0.0],
                        [0.0, k22_matrix, 0.0],
                        [0.0, 0.0, k33_matrix]])

#Elasticity forces and parameters

c11 = 7.0/3.0 #107.3e9 #Gpa
c12 = 1.0/3.0 #28.3e9 #Gpa
c44 = 1.0 #60.9e9 #Gpa

#Fiber
k_sphere = 1028892679637.1346 #Pa
u_sphere = 720224875745.99426 #Pa
c11_sphere = (k_sphere + (4.0/3.0)*u_sphere)
c12_sphere = (k_sphere - (2.0/3.0)*u_sphere)
c44_sphere = u_sphere

#Matrix
k_matrix = 234038326664.34158 #Pa
u_matrix = 163826828665.03912 #Pa
c11_matrix = (k_matrix + (4.0/3.0)*u_matrix)
c12_matrix = (k_matrix - (2.0/3.0)*u_matrix)
c44_matrix = u_matrix



Etensor_sphere = np.array([[c11_sphere, c12_sphere, c12_sphere, 0.0, 0.0, 0.0],
                          [c12_sphere, c11_sphere, c12_sphere, 0.0, 0.0, 0.0],
                          [c12_sphere, c12_sphere, c11_sphere, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, c44_sphere, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, c44_sphere, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, c44_sphere]])

Etensor_matrix = np.array([[c11_matrix, c12_matrix, c12_matrix, 0.0, 0.0, 0.0],
                          [c12_matrix, c11_matrix, c12_matrix, 0.0, 0.0, 0.0],
                          [c12_matrix, c12_matrix, c11_matrix, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, c44_matrix, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, c44_matrix, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, c44_matrix]])


#Densities
density_sphere = 9003.1525065990718

density_matrix = 11855.070522954735

#################Desired Properties

#Thermal
k11_desired = 60.500000000000000 #W/mk
k22_desired = 60.500000000000000 #W/mk
k33_desired = 60.500000000000000 #W/mk

k_mat_desired = np.array([[k11_desired, 0.0, 0.0],
                        [0.0, k22_desired, 0.0],
                        [0.0, 0.0, k33_desired]])


#Mechanical

k_desired = 166670000000.00000 #Pa
u_desired = 76923000000.000000 #Pa
c11_desired = (k_desired + (4.0/3.0)*u_desired)
c12_desired = (k_desired - (2.0/3.0)*u_desired)
c44_desired = u_desired


Etensor_desired = np.array([[c11_desired, c12_desired, c12_desired, 0.0, 0.0, 0.0],
                          [c12_desired, c11_desired, c12_desired, 0.0, 0.0, 0.0],
                          [c12_desired, c12_desired, c11_desired, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, c44_desired, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, c44_desired, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, c44_desired]])

#Density

density_desired = 7850.0000000000000

#############################################################################
#Computation Parameters

weight_mech = 10.0
weight_therm = 1.0
weight_density = 1.0

beta = 1.01

Nx = 9

Etensor_eff_mat = np.zeros((string_size, np.shape(Etensor_matrix)[0], np.shape(Etensor_matrix)[1]))
Kmat_eff_mat = np.zeros((string_size, np.shape(k_matrix_mat)[0], np.shape(k_matrix_mat)[1]))
# density_eff_mat = np.zeros((string_size))

# Etensor_eff_mat_serial = np.zeros((string_size, np.shape(Etensor_matrix)[0], np.shape(Etensor_matrix)[1]))
# Kmat_eff_mat_serial = np.zeros((string_size, np.shape(k_matrix_mat)[0], np.shape(k_matrix_mat)[1]))

cost_func_vec = np.zeros((string_size,2))





if rank == 0:
    sendbuf_scatter = np.empty([string_size,fortran_table_size,4], dtype='float')
    sendbuf_scatter = sphere_table_matrix
    # local_result_0 = 0.0

recvbuf_scatter = np.empty([process_size,fortran_table_size,4], dtype='float')

comm.Scatter(sendbuf_scatter, recvbuf_scatter, root = 0)

#send_buf = np.array([[local_1], [local_2], [local_3]])
local_result_kmat = np.zeros((process_size,3,3))
local_result_etensor = np.zeros((process_size,6,6))
# local_result_density = np.zeros(process_size)
# inclusion_volume_mat = np.zeros(process_size)


for i in range(0, process_size):
    [local_result_kmat[i], local_result_etensor[i]] = effective_prop_fem(Nx, k_matrix_mat, k_sphere_mat, Etensor_matrix, Etensor_sphere,
                                                    recvbuf_scatter[i], beta)


    # local_result_density[i] = density_matrix + (volume_fraction)*(density_sphere - density_matrix)



# print 'rank', rank
# print 'local densities', local_result_density

# print 'Local Kmat', local_result_kmat
# print 'Local Etensor', local_result_etensor

#Gather the results from the processors
recvbuf_kmat = None
recvbuf_etensor = None
# recvbuf_density = None
if rank == 0:
    recvbuf_kmat = np.empty([string_size,3,3], dtype='float')
    recvbuf_etensor = np.empty([string_size, 6, 6], dtype='float')
    # recvbuf_density = np.empty([string_size], dtype='float')

comm.Gather(local_result_kmat, recvbuf_kmat, root = 0)
comm.Gather(local_result_etensor, recvbuf_etensor, root=0)
# comm.Gather(local_result_density, recvbuf_density, root = 0)

#recv_buffer = np.zeros(5)

#Communication
#comm.Scatter(send_buf, local_result, root = 0)
if rank == 0:
    Kmat_eff_mat += recvbuf_kmat
    Etensor_eff_mat += recvbuf_etensor
    # density_eff_mat += recvbuf_density

    # print 'density eff mat', density_eff_mat



#Terminate MPI environment
# comm.Abort()

if rank ==0:

    for qq in range(0, string_size):

        cost_func_vec[qq,0] = weight_mech*(np.linalg.norm(Etensor_desired-Etensor_eff_mat[qq]) \
                                           /np.linalg.norm(Etensor_desired)) \
                              + weight_therm*np.linalg.norm(k_mat_desired - Kmat_eff_mat[qq]) \
                                /np.linalg.norm(k_mat_desired)
                              # + weight_density*np.linalg.norm(density_desired - density_eff_mat[qq])

        cost_func_vec[qq, 1] = qq






    #Sort the cost function
    cost_func_sorted = sort1(cost_func_vec)



#Rerun the process inside a loop
    #To a certain number of iteration
    #To a certain cost function value


##########################Loops starts here


iterations = 0
max_iter = 500

if rank == 0:

    cost_func_min_vec = np.zeros(max_iter+1)
    cost_func_min_vec[0] = np.min(cost_func_sorted[:,0])


# #The Big While Loop

#

# while np.min(cost_func_min_vec) < 0.1 and iterations < max_iter:
#Only do iterations for now - bug with costfuncminvec broadcasting
while iterations < max_iter:

    if rank == 0:

        # print 'Iteration', iterations
        # print 'Cost Minimum', cost_func_min_vec[iterations]
        # print 'calculation time', datetime.now() - startTime
        # print 'Sorted Cost 1', cost_func_sorted

        # Keep the Top Performing Parents for the sphere table
        keep_parents = 3
        make_children = 3

        parents_vec = cost_func_sorted[0:keep_parents]
        parents_index_vec = cost_func_sorted[0:keep_parents, 1]

        sphere_table_matrix_parents = np.zeros((keep_parents, fortran_table_size, 4))
        sphere_table_matrix_children = np.zeros((make_children, fortran_table_size, 4))

        for qq in range(0, keep_parents):
            sphere_table_matrix_parents[qq] = sphere_table_matrix[int(parents_index_vec[qq])]

        # Generate children from parents
        for ww in range(0, make_children):
            sphere_table_matrix_children[ww] = (sphere_table_matrix_parents[ww] + sphere_table_matrix_parents[ww - 1]) / 2.0

        # Create new sphere table based on parents and children

        sphere_table_matrix_new = np.zeros((string_size, fortran_table_size, 4))
        sphere_table_matrix_new[0:keep_parents] = sphere_table_matrix_parents
        sphere_table_matrix_new[keep_parents:keep_parents + make_children] = sphere_table_matrix_children

        #Reinitiate the explorers here

        for l in range(keep_parents + make_children, string_size):
            # sphere_table = np.zeros((number_spheres, 4))

            sphere_table = sphere_gen(volume_fraction, number_spheres)

            sphere_table_matrix_new[l, 0:number_spheres, 0] = sphere_table[:,0]
            sphere_table_matrix_new[l, 0:number_spheres, 1] = sphere_table[:,1]
            sphere_table_matrix_new[l, 0:number_spheres, 2] = sphere_table[:,2]
            sphere_table_matrix_new[l, 0:number_spheres, 3] = sphere_radius

        # sphere_table_matrix_new[:, 45:150, :] = 0.0

        #Zero out the vectors and matrices
        Etensor_eff_mat = np.zeros((string_size, np.shape(Etensor_matrix)[0], np.shape(Etensor_matrix)[1]))
        Kmat_eff_mat = np.zeros((string_size, np.shape(k_matrix_mat)[0], np.shape(k_matrix_mat)[1]))
        # density_eff_mat = np.zeros((string_size, 1))

        cost_func_vec = np.zeros((string_size, 2))

        #Change the sphere table from old to new
        sphere_table_matrix = sphere_table_matrix_new


    process_size = int(string_size / size)
    sendbuf_scatter = None

    if rank == 0:
        sendbuf_scatter = np.empty([string_size, fortran_table_size, 4], dtype='float')
        sendbuf_scatter = sphere_table_matrix
        # local_result_0 = 0.0

    recvbuf_scatter = np.empty([process_size, fortran_table_size, 4], dtype='float')

    comm.Scatter(sendbuf_scatter, recvbuf_scatter, root=0)

    # send_buf = np.array([[local_1], [local_2], [local_3]])
    local_result_kmat = np.zeros((process_size, 3, 3))
    local_result_etensor = np.zeros((process_size, 6, 6))
    # local_result_density = np.zeros(process_size)

    for i in range(0, process_size):
        [local_result_kmat[i], local_result_etensor[i]] = effective_prop_fem(Nx, k_matrix_mat, k_sphere_mat,
                                                                             Etensor_matrix, Etensor_sphere,
                                                                             recvbuf_scatter[i], beta)

        # inclusion_volume = spheres_volume.sphere_inclusion_volume(recvbuf_scatter[i])
        #
        # local_result_density[i] = density_matrix + (inclusion_volume) * (density_sphere - density_matrix)

    # print 'Local Kmat', local_result_kmat
    # print 'Local Etensor', local_result_etensor

    # Gather the results from the processors
    recvbuf_kmat = None
    recvbuf_etensor = None
    # recvbuf_density = None
    if rank == 0:
        recvbuf_kmat = np.empty([string_size, 3, 3], dtype='float')
        recvbuf_etensor = np.empty([string_size, 6, 6], dtype='float')
        # recvbuf_density = np.empty([string_size,1], dtype='float')

    comm.Gather(local_result_kmat, recvbuf_kmat, root=0)
    comm.Gather(local_result_etensor, recvbuf_etensor, root=0)
    # comm.Gather(local_result_density, recvbuf_density, root=0)
    # recv_buffer = np.zeros(5)


    # Communication
    # comm.Scatter(send_buf, local_result, root = 0)
    if rank == 0:


        Kmat_eff_mat += recvbuf_kmat
        Etensor_eff_mat += recvbuf_etensor
        # density_eff_mat += recvbuf_density


    # Terminate MPI environment
    # comm.Abort()

    #Cost function computation
    if rank == 0:

        for qq in range(0, string_size):

            cost_func_vec[qq, 0] = weight_mech*(np.linalg.norm(Etensor_desired-Etensor_eff_mat[qq]) \
                                           /np.linalg.norm(Etensor_desired)) \
                              + weight_therm*np.linalg.norm(k_mat_desired - Kmat_eff_mat[qq]) \
                                /np.linalg.norm(k_mat_desired)
                                   # + weight_density * np.linalg.norm(density_desired - density_eff_mat[qq])

            cost_func_vec[qq, 1] = qq



        # Sort the cost function
        cost_func_sorted = sort1(cost_func_vec)



        cost_func_min_vec[iterations+1] = np.min(cost_func_sorted[:, 0])



    iterations += 1

if rank == 0:
    tag = 'jan_23_test_n9_500_iter'
    microstuct_ga1 = open('mesoscale_ga_' + tag + '.txt', 'w')
    microstuct_ga1.write('Microstructural Optimization for Elasticity and Thermal Conductivity \n')
    microstuct_ga1.write('Study Tag: ' + tag)
    microstuct_ga1.write('\n')
    microstuct_ga1.write('Final Cost Minima: \n')
    count = 0
    for L2 in cost_func_min_vec:
        microstuct_ga1.write('\t {0}'.format(L2))
        microstuct_ga1.write(', ' + str(count) + ', \n')
        count += 1
    microstuct_ga1.write('\n')

    microstuct_ga1.write('Final Fiber Tables: \n')
    count_tensor = 0
    for L4 in sphere_table_matrix_new:
        microstuct_ga1.write('Next Table')
        for L3 in L4:
            microstuct_ga1.write('\t [{0}, {1}, {2}, {3}],  \n'.format(*L3))

        microstuct_ga1.write('Etensor_mat \n')
        for L5 in Etensor_eff_mat[count_tensor]:
            microstuct_ga1.write('\t [{0}, {1}, {2}, {3}, {4}, {5}], \n'.format(*L5))

        microstuct_ga1.write('\n')

        microstuct_ga1.write('K_eff_mat \n')
        for L6 in Kmat_eff_mat[count_tensor]:
            microstuct_ga1.write('\t [{0}, {1}, {2}], \n'.format(*L6))

        microstuct_ga1.write('\n')

        count_tensor += 1

    microstuct_ga1.write('Done')

    microstuct_ga1.close()






















