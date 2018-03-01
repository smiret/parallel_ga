
#Function call for the effective property calculation
def effective_prop_fem(Nx, k_matrix_mat, k_sphere_mat,Etensor_matrix, Etensor_sphere, sphere_table, beta):


    import numpy as np
    # from numpy.linalg import inv
    # from scipy import interpolate
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    from datetime import datetime

    import elast1_spheres
    import thermo1_spheres
    import strain_temp2_spheres

    startTime = datetime.now()


#####################################################################################################################


####################################################################################################################
# Define the Meshing Function
    def cmesh(n):  # Number of Nodes in Each Direction For Meshing Function

        n_nodes = int(n ** 3)
        n_elem = int((n - 1) ** 3)

        # This code generates the connectivity table for the mesh

        conn = np.zeros(((n - 1) ** 3, 8), dtype=np.int)

        # First row of connectivity table
        conn[0, 0] = 1
        conn[0, 1] = 2
        conn[0, 2] = n + 2
        conn[0, 3] = n + 1
        conn[0, 4] = n ** 2 + 1
        conn[0, 5] = n ** 2 + 2
        conn[0, 6] = n ** 2 + n + 2
        conn[0, 7] = n ** 2 + n + 1

        counter = 0  # counter for row of connectivity table
        for ii in range(0, n - 1):  # outer jumps in z + n**2
            if ii < n - 1:
                conn[counter] = conn[counter - (n - 1) ** 2] + n ** 2
                conn[0, 0] = 1
                conn[0, 1] = 2
                conn[0, 2] = n + 2
                conn[0, 3] = n + 1
                conn[0, 4] = n ** 2 + 1
                conn[0, 5] = n ** 2 + 2
                conn[0, 6] = n ** 2 + n + 2
                conn[0, 7] = n ** 2 + n + 1
                counter = counter + 1
                jj = 0
            else:
                jj = 0
            for jj in range(0, n - 1):  # outer jumps in x + 1
                if jj < n - 2:
                    conn[counter] = conn[counter - 1] + 1
                    counter = counter + 1
                    kk = 0
                else:
                    kk = 0
                    for kk in range(0, n - 2):  # inner jumps in y + 2
                        conn[counter] = conn[counter - 1] + 2
                        counter = counter + 1
                        for jj in range(0, n - 2):  # outer jumps in x + 1
                            conn[counter] = conn[counter - 1] + 1
                            counter = counter + 1

        # Adjust the indexes to 0-base
        conn = conn - 1

        for r in range(0, len(conn)):
            for col in range(0, 4):
                conn[r, col] = int(conn[r, col])  # Make the entries integers

        # print conn

        # Now I need to generate the nodes corresponding to the connectivity table

        nodes = np.zeros((n ** 3, 3))
        nodes2 = np.zeros((n ** 3, 3))

        space = np.linspace(0, 1, n)

        step = space[1] - space[0]

        cc1 = 0

        for zz in range(0, n):
            nodes[cc1, 2] = space[zz]
            nodes[cc1, 1] = 0
            nodes[cc1, 0] = 0
            # cc1 = cc1 + 1
            nodes[0] = 0
            for yy in range(0, n):
                nodes[cc1, 2] = space[zz]
                nodes[cc1, 1] = space[yy]
                # cc1 = cc1 + 1
                for xx in range(0, n):
                    nodes[cc1, 2] = space[zz]
                    nodes[cc1, 1] = space[yy]
                    nodes[cc1, 0] = space[xx]
                    cc1 = cc1 + 1

        nodes3 = nodes

        return nodes3, conn

#####################################################################################################################

#Mechanical Boundary Condition Function - Bottom Z direction

    def mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        u_bc = np.zeros((8,3))
        x_load = np.zeros((8,3))
        conn2 = conn + np.max(conn) + 1
        conn3 = conn2 + np.max(conn) + 1

        Ne = len(conn)

        for e in range(0, (Nx - 1) ** 2):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                u_bc[ii] = np.dot(load_state, x_load[ii])

            Selast[conn[e, 0], :] = 0.0
            Selast[conn[e, 1], :] = 0.0
            Selast[conn[e, 2], :] = 0.0
            Selast[conn[e, 3], :] = 0.0

            Selast[conn[e, 0], conn[e, 0]] = 1.0
            Selast[conn[e, 1], conn[e, 1]] = 1.0
            Selast[conn[e, 2], conn[e, 2]] = 1.0
            Selast[conn[e, 3], conn[e, 3]] = 1.0

            Selast[conn2[e, 0], :] = 0.0
            Selast[conn2[e, 1], :] = 0.0
            Selast[conn2[e, 2], :] = 0.0
            Selast[conn2[e, 3], :] = 0.0

            Selast[conn2[e, 0], conn2[e, 0]] = 1.0
            Selast[conn2[e, 1], conn2[e, 1]] = 1.0
            Selast[conn2[e, 2], conn2[e, 2]] = 1.0
            Selast[conn2[e, 3], conn2[e, 3]] = 1.0

            Selast[conn3[e, 0], :] = 0.0
            Selast[conn3[e, 1], :] = 0.0
            Selast[conn3[e, 2], :] = 0.0
            Selast[conn3[e, 3], :] = 0.0

            Selast[conn3[e, 0], conn3[e, 0]] = 1.0
            Selast[conn3[e, 1], conn3[e, 1]] = 1.0
            Selast[conn3[e, 2], conn3[e, 2]] = 1.0
            Selast[conn3[e, 3], conn3[e, 3]] = 1.0

            Melast[conn[e, 0], :] = 0.0
            Melast[conn[e, 1], :] = 0.0
            Melast[conn[e, 2], :] = 0.0
            Melast[conn[e, 3], :] = 0.0

            Melast[conn[e, 0], conn[e, 0]] = 1.0
            Melast[conn[e, 1], conn[e, 1]] = 1.0
            Melast[conn[e, 2], conn[e, 2]] = 1.0
            Melast[conn[e, 3], conn[e, 3]] = 1.0

            Melast[conn2[e, 0], :] = 0.0
            Melast[conn2[e, 1], :] = 0.0
            Melast[conn2[e, 2], :] = 0.0
            Melast[conn2[e, 3], :] = 0.0

            Melast[conn2[e, 0], conn2[e, 0]] = 1.0
            Melast[conn2[e, 1], conn2[e, 1]] = 1.0
            Melast[conn2[e, 2], conn2[e, 2]] = 1.0
            Melast[conn2[e, 3], conn2[e, 3]] = 1.0

            Melast[conn3[e, 0], :] = 0.0
            Melast[conn3[e, 1], :] = 0.0
            Melast[conn3[e, 2], :] = 0.0
            Melast[conn3[e, 3], :] = 0.0

            Melast[conn3[e, 0], conn3[e, 0]] = 1.0
            Melast[conn3[e, 1], conn3[e, 1]] = 1.0
            Melast[conn3[e, 2], conn3[e, 2]] = 1.0
            Melast[conn3[e, 3], conn3[e, 3]] = 1.0

            Relast[conn[e, 0]] = u_bc[0, 0]
            Relast[conn[e, 1]] = u_bc[1, 0]
            Relast[conn[e, 2]] = u_bc[2, 0]
            Relast[conn[e, 3]] = u_bc[3, 0]

            Relast[conn2[e, 0]] = u_bc[0, 1]
            Relast[conn2[e, 1]] = u_bc[1, 1]
            Relast[conn2[e, 2]] = u_bc[2, 1]
            Relast[conn2[e, 3]] = u_bc[3, 1]

            Relast[conn3[e, 0]] = u_bc[0, 2]
            Relast[conn3[e, 1]] = u_bc[1, 2]
            Relast[conn3[e, 2]] = u_bc[2, 2]
            Relast[conn3[e, 3]] = u_bc[3, 2]

        return Selast, Melast, Relast


#####################################################################################################################

#Mechanical Boundary Condition Function - Top Z direction


    def mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        u_bc = np.zeros((8,3))
        x_load = np.zeros((8,3))
        conn2 = conn + np.max(conn) + 1
        conn3 = conn2 + np.max(conn) + 1

        Ne = len(conn)


        for e in range(Ne - (Nx - 1) ** 2, Ne):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                u_bc[ii] = np.dot(load_state, x_load[ii])

            Selast[conn[e, 4], :] = 0.0
            Selast[conn[e, 5], :] = 0.0
            Selast[conn[e, 6], :] = 0.0
            Selast[conn[e, 7], :] = 0.0

            Selast[conn[e, 4], conn[e, 4]] = 1.0
            Selast[conn[e, 5], conn[e, 5]] = 1.0
            Selast[conn[e, 6], conn[e, 6]] = 1.0
            Selast[conn[e, 7], conn[e, 7]] = 1.0

            Selast[conn2[e, 4], :] = 0.0
            Selast[conn2[e, 5], :] = 0.0
            Selast[conn2[e, 6], :] = 0.0
            Selast[conn2[e, 7], :] = 0.0

            Selast[conn2[e, 4], conn2[e, 4]] = 1.0
            Selast[conn2[e, 5], conn2[e, 5]] = 1.0
            Selast[conn2[e, 6], conn2[e, 6]] = 1.0
            Selast[conn2[e, 7], conn2[e, 7]] = 1.0

            Selast[conn3[e, 4], :] = 0.0
            Selast[conn3[e, 5], :] = 0.0
            Selast[conn3[e, 6], :] = 0.0
            Selast[conn3[e, 7], :] = 0.0

            Selast[conn3[e, 4], conn3[e, 4]] = 1.0
            Selast[conn3[e, 5], conn3[e, 5]] = 1.0
            Selast[conn3[e, 6], conn3[e, 6]] = 1.0
            Selast[conn3[e, 7], conn3[e, 7]] = 1.0

            Melast[conn[e, 4], :] = 0.0
            Melast[conn[e, 5], :] = 0.0
            Melast[conn[e, 6], :] = 0.0
            Melast[conn[e, 7], :] = 0.0

            Melast[conn[e, 4], conn[e, 4]] = 1.0
            Melast[conn[e, 5], conn[e, 5]] = 1.0
            Melast[conn[e, 6], conn[e, 6]] = 1.0
            Melast[conn[e, 7], conn[e, 7]] = 1.0

            Melast[conn2[e, 4], :] = 0.0
            Melast[conn2[e, 5], :] = 0.0
            Melast[conn2[e, 6], :] = 0.0
            Melast[conn2[e, 7], :] = 0.0

            Melast[conn2[e, 4], conn2[e, 4]] = 1.0
            Melast[conn2[e, 5], conn2[e, 5]] = 1.0
            Melast[conn2[e, 6], conn2[e, 6]] = 1.0
            Melast[conn2[e, 7], conn2[e, 7]] = 1.0

            Melast[conn3[e, 4], :] = 0.0
            Melast[conn3[e, 5], :] = 0.0
            Melast[conn3[e, 6], :] = 0.0
            Melast[conn3[e, 7], :] = 0.0

            Melast[conn3[e, 4], conn3[e, 4]] = 1.0
            Melast[conn3[e, 5], conn3[e, 5]] = 1.0
            Melast[conn3[e, 6], conn3[e, 6]] = 1.0
            Melast[conn3[e, 7], conn3[e, 7]] = 1.0

            Relast[conn[e, 4]] = u_bc[4, 0]
            Relast[conn[e, 5]] = u_bc[5, 0]
            Relast[conn[e, 6]] = u_bc[6, 0]
            Relast[conn[e, 7]] = u_bc[7, 0]

            Relast[conn2[e, 4]] = u_bc[4, 1]
            Relast[conn2[e, 5]] = u_bc[5, 1]
            Relast[conn2[e, 6]] = u_bc[6, 1]
            Relast[conn2[e, 7]] = u_bc[7, 1]

            Relast[conn3[e, 4]] = u_bc[4, 2]
            Relast[conn3[e, 5]] = u_bc[5, 2]
            Relast[conn3[e, 6]] = u_bc[6, 2]
            Relast[conn3[e, 7]] = u_bc[7, 2]

        return Selast, Melast, Relast

#####################################################################################################################

#Mechanical Boundary Condition Function - Back X Direction

    def mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        u_bc = np.zeros((8,3))
        x_load = np.zeros((8,3))
        conn2 = conn + np.max(conn) + 1
        conn3 = conn2 + np.max(conn) + 1

        Ne = len(conn)

        for e in range(0, (Ne), Nx - 1):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                u_bc[ii] = np.dot(load_state, x_load[ii])

            Selast[conn[e, 0], :] = 0.0
            Selast[conn[e, 3], :] = 0.0
            Selast[conn[e, 4], :] = 0.0
            Selast[conn[e, 7], :] = 0.0

            Selast[conn[e, 0], conn[e, 0]] = 1.0
            Selast[conn[e, 3], conn[e, 3]] = 1.0
            Selast[conn[e, 4], conn[e, 4]] = 1.0
            Selast[conn[e, 7], conn[e, 7]] = 1.0

            Selast[conn2[e, 0], :] = 0.0
            Selast[conn2[e, 3], :] = 0.0
            Selast[conn2[e, 4], :] = 0.0
            Selast[conn2[e, 7], :] = 0.0

            Selast[conn2[e, 0], conn2[e, 0]] = 1.0
            Selast[conn2[e, 3], conn2[e, 3]] = 1.0
            Selast[conn2[e, 4], conn2[e, 4]] = 1.0
            Selast[conn2[e, 7], conn2[e, 7]] = 1.0

            Selast[conn3[e, 0], :] = 0.0
            Selast[conn3[e, 3], :] = 0.0
            Selast[conn3[e, 4], :] = 0.0
            Selast[conn3[e, 7], :] = 0.0

            Selast[conn3[e, 0], conn3[e, 0]] = 1.0
            Selast[conn3[e, 3], conn3[e, 3]] = 1.0
            Selast[conn3[e, 4], conn3[e, 4]] = 1.0
            Selast[conn3[e, 7], conn3[e, 7]] = 1.0

            Melast[conn[e, 0], :] = 0.0
            Melast[conn[e, 3], :] = 0.0
            Melast[conn[e, 4], :] = 0.0
            Melast[conn[e, 7], :] = 0.0

            Melast[conn[e, 0], conn[e, 0]] = 1.0
            Melast[conn[e, 3], conn[e, 3]] = 1.0
            Melast[conn[e, 4], conn[e, 4]] = 1.0
            Melast[conn[e, 7], conn[e, 7]] = 1.0

            Melast[conn2[e, 0], :] = 0.0
            Melast[conn2[e, 3], :] = 0.0
            Melast[conn2[e, 4], :] = 0.0
            Melast[conn2[e, 7], :] = 0.0

            Melast[conn2[e, 0], conn2[e, 0]] = 1.0
            Melast[conn2[e, 3], conn2[e, 3]] = 1.0
            Melast[conn2[e, 4], conn2[e, 4]] = 1.0
            Melast[conn2[e, 7], conn2[e, 7]] = 1.0

            Melast[conn3[e, 0], :] = 0.0
            Melast[conn3[e, 3], :] = 0.0
            Melast[conn3[e, 4], :] = 0.0
            Melast[conn3[e, 7], :] = 0.0

            Melast[conn3[e, 0], conn3[e, 0]] = 1.0
            Melast[conn3[e, 3], conn3[e, 3]] = 1.0
            Melast[conn3[e, 4], conn3[e, 4]] = 1.0
            Melast[conn3[e, 7], conn3[e, 7]] = 1.0

            Relast[conn[e, 0]] = u_bc[0, 0]
            Relast[conn[e, 3]] = u_bc[3, 0]
            Relast[conn[e, 4]] = u_bc[4, 0]
            Relast[conn[e, 7]] = u_bc[7, 0]

            Relast[conn2[e, 0]] = u_bc[0, 1]
            Relast[conn2[e, 3]] = u_bc[3, 1]
            Relast[conn2[e, 4]] = u_bc[4, 1]
            Relast[conn2[e, 7]] = u_bc[7, 1]

            Relast[conn3[e, 0]] = u_bc[0, 2]
            Relast[conn3[e, 3]] = u_bc[3, 2]
            Relast[conn3[e, 4]] = u_bc[4, 2]
            Relast[conn3[e, 7]] = u_bc[7, 2]

        return Selast, Melast, Relast

#####################################################################################################################

#Mechanical Boundary Condition Function - Front X Direction

    def mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        u_bc = np.zeros((8,3))
        x_load = np.zeros((8,3))
        conn2 = conn + np.max(conn) + 1
        conn3 = conn2 + np.max(conn) + 1

        Ne = len(conn)

        for e in range(Nx - 2, Ne, Nx - 1):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                u_bc[ii] = np.dot(load_state, x_load[ii])

            Selast[conn[e, 1], :] = 0.0
            Selast[conn[e, 2], :] = 0.0
            Selast[conn[e, 5], :] = 0.0
            Selast[conn[e, 6], :] = 0.0

            Selast[conn[e, 1], conn[e, 1]] = 1.0
            Selast[conn[e, 2], conn[e, 2]] = 1.0
            Selast[conn[e, 5], conn[e, 5]] = 1.0
            Selast[conn[e, 6], conn[e, 6]] = 1.0

            Selast[conn2[e, 1], :] = 0.0
            Selast[conn2[e, 2], :] = 0.0
            Selast[conn2[e, 5], :] = 0.0
            Selast[conn2[e, 6], :] = 0.0

            Selast[conn2[e, 1], conn2[e, 1]] = 1.0
            Selast[conn2[e, 2], conn2[e, 2]] = 1.0
            Selast[conn2[e, 5], conn2[e, 5]] = 1.0
            Selast[conn2[e, 6], conn2[e, 6]] = 1.0

            Selast[conn3[e, 1], :] = 0.0
            Selast[conn3[e, 2], :] = 0.0
            Selast[conn3[e, 5], :] = 0.0
            Selast[conn3[e, 6], :] = 0.0

            Selast[conn3[e, 1], conn3[e, 1]] = 1.0
            Selast[conn3[e, 2], conn3[e, 2]] = 1.0
            Selast[conn3[e, 5], conn3[e, 5]] = 1.0
            Selast[conn3[e, 6], conn3[e, 6]] = 1.0

            Melast[conn[e, 1], :] = 0.0
            Melast[conn[e, 2], :] = 0.0
            Melast[conn[e, 5], :] = 0.0
            Melast[conn[e, 6], :] = 0.0

            Melast[conn[e, 1], conn[e, 1]] = 1.0
            Melast[conn[e, 2], conn[e, 2]] = 1.0
            Melast[conn[e, 5], conn[e, 5]] = 1.0
            Melast[conn[e, 6], conn[e, 6]] = 1.0

            Melast[conn2[e, 1], :] = 0.0
            Melast[conn2[e, 2], :] = 0.0
            Melast[conn2[e, 5], :] = 0.0
            Melast[conn2[e, 6], :] = 0.0

            Melast[conn2[e, 1], conn2[e, 1]] = 1.0
            Melast[conn2[e, 2], conn2[e, 2]] = 1.0
            Melast[conn2[e, 5], conn2[e, 5]] = 1.0
            Melast[conn2[e, 6], conn2[e, 6]] = 1.0

            Melast[conn3[e, 1], :] = 0.0
            Melast[conn3[e, 2], :] = 0.0
            Melast[conn3[e, 5], :] = 0.0
            Melast[conn3[e, 6], :] = 0.0

            Melast[conn3[e, 1], conn3[e, 1]] = 1.0
            Melast[conn3[e, 2], conn3[e, 2]] = 1.0
            Melast[conn3[e, 5], conn3[e, 5]] = 1.0
            Melast[conn3[e, 6], conn3[e, 6]] = 1.0

            Relast[conn[e, 1]] = u_bc[1, 0]
            Relast[conn[e, 2]] = u_bc[2, 0]
            Relast[conn[e, 5]] = u_bc[5, 0]
            Relast[conn[e, 6]] = u_bc[6, 0]

            Relast[conn2[e, 1]] = u_bc[1, 1]
            Relast[conn2[e, 2]] = u_bc[2, 1]
            Relast[conn2[e, 5]] = u_bc[5, 1]
            Relast[conn2[e, 6]] = u_bc[6, 1]

            Relast[conn3[e, 1]] = u_bc[1, 2]
            Relast[conn3[e, 2]] = u_bc[2, 2]
            Relast[conn3[e, 5]] = u_bc[5, 2]
            Relast[conn3[e, 6]] = u_bc[6, 2]

        return Selast, Melast, Relast

#####################################################################################################################

#Mechanical Boundary Condition Function - Left Y Direction

    def mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        u_bc = np.zeros((8,3))
        x_load = np.zeros((8,3))
        conn2 = conn + np.max(conn) + 1
        conn3 = conn2 + np.max(conn) + 1

        Ne = len(conn)

        for ee in range(0, Nx - 1):
            for e in range(ee * (Nx - 1) ** 2, ee * (Nx - 1) ** 2 + (Nx - 1)):

                x1[:] = np.array(nodes[conn[e, :], 0])
                x2[:] = np.array(nodes[conn[e, :], 1])
                x3[:] = np.array(nodes[conn[e, :], 2])

                x_load[:, 0] = x1
                x_load[:, 1] = x2
                x_load[:, 2] = x3

                for ii in range(0, 8):
                    u_bc[ii] = np.dot(load_state, x_load[ii])

                Selast[conn[e, 0], :] = 0.0
                Selast[conn[e, 1], :] = 0.0
                Selast[conn[e, 4], :] = 0.0
                Selast[conn[e, 5], :] = 0.0

                Selast[conn[e, 0], conn[e, 0]] = 1.0
                Selast[conn[e, 1], conn[e, 1]] = 1.0
                Selast[conn[e, 4], conn[e, 4]] = 1.0
                Selast[conn[e, 5], conn[e, 5]] = 1.0

                Selast[conn2[e, 0], :] = 0.0
                Selast[conn2[e, 1], :] = 0.0
                Selast[conn2[e, 4], :] = 0.0
                Selast[conn2[e, 5], :] = 0.0

                Selast[conn2[e, 0], conn2[e, 0]] = 1.0
                Selast[conn2[e, 1], conn2[e, 1]] = 1.0
                Selast[conn2[e, 4], conn2[e, 4]] = 1.0
                Selast[conn2[e, 5], conn2[e, 5]] = 1.0

                Selast[conn3[e, 0], :] = 0.0
                Selast[conn3[e, 1], :] = 0.0
                Selast[conn3[e, 4], :] = 0.0
                Selast[conn3[e, 5], :] = 0.0

                Selast[conn3[e, 0], conn3[e, 0]] = 1.0
                Selast[conn3[e, 1], conn3[e, 1]] = 1.0
                Selast[conn3[e, 4], conn3[e, 4]] = 1.0
                Selast[conn3[e, 5], conn3[e, 5]] = 1.0

                Melast[conn[e, 0], :] = 0.0
                Melast[conn[e, 1], :] = 0.0
                Melast[conn[e, 4], :] = 0.0
                Melast[conn[e, 5], :] = 0.0

                Melast[conn[e, 0], conn[e, 0]] = 1.0
                Melast[conn[e, 1], conn[e, 1]] = 1.0
                Melast[conn[e, 4], conn[e, 4]] = 1.0
                Melast[conn[e, 5], conn[e, 5]] = 1.0

                Melast[conn2[e, 0], :] = 0.0
                Melast[conn2[e, 1], :] = 0.0
                Melast[conn2[e, 4], :] = 0.0
                Melast[conn2[e, 5], :] = 0.0

                Melast[conn2[e, 0], conn2[e, 0]] = 1.0
                Melast[conn2[e, 1], conn2[e, 1]] = 1.0
                Melast[conn2[e, 4], conn2[e, 4]] = 1.0
                Melast[conn2[e, 5], conn2[e, 5]] = 1.0

                Melast[conn3[e, 0], :] = 0.0
                Melast[conn3[e, 1], :] = 0.0
                Melast[conn3[e, 4], :] = 0.0
                Melast[conn3[e, 5], :] = 0.0

                Melast[conn3[e, 0], conn3[e, 0]] = 1.0
                Melast[conn3[e, 1], conn3[e, 1]] = 1.0
                Melast[conn3[e, 4], conn3[e, 4]] = 1.0
                Melast[conn3[e, 5], conn3[e, 5]] = 1.0

                Relast[conn[e, 0]] = u_bc[0, 0]
                Relast[conn[e, 1]] = u_bc[1, 0]
                Relast[conn[e, 4]] = u_bc[4, 0]
                Relast[conn[e, 5]] = u_bc[5, 0]

                Relast[conn2[e, 0]] = u_bc[0, 1]
                Relast[conn2[e, 1]] = u_bc[1, 1]
                Relast[conn2[e, 4]] = u_bc[4, 1]
                Relast[conn2[e, 5]] = u_bc[5, 1]

                Relast[conn3[e, 0]] = u_bc[0, 2]
                Relast[conn3[e, 1]] = u_bc[1, 2]
                Relast[conn3[e, 4]] = u_bc[4, 2]
                Relast[conn3[e, 5]] = u_bc[5, 2]

        return Selast, Melast, Relast

#####################################################################################################################

#Mechanical Boundary Condition Function - Right Y Direction

    def mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        u_bc = np.zeros((8,3))
        x_load = np.zeros((8,3))
        conn2 = conn + np.max(conn) + 1
        conn3 = conn2 + np.max(conn) + 1

        Ne = len(conn)

        for ee in range(0, Nx - 1):
            for e in range(ee * (Nx - 1) ** 2 + ((Nx - 1) ** 2 - (Nx - 1)), (ee + 1) * (Nx - 1) ** 2):

                x1[:] = np.array(nodes[conn[e, :], 0])
                x2[:] = np.array(nodes[conn[e, :], 1])
                x3[:] = np.array(nodes[conn[e, :], 2])

                x_load[:, 0] = x1
                x_load[:, 1] = x2
                x_load[:, 2] = x3

                for ii in range(0, 8):
                    u_bc[ii] = np.dot(load_state, x_load[ii])

                Selast[conn[e, 2], :] = 0.0
                Selast[conn[e, 3], :] = 0.0
                Selast[conn[e, 6], :] = 0.0
                Selast[conn[e, 7], :] = 0.0

                Selast[conn[e, 2], conn[e, 2]] = 1.0
                Selast[conn[e, 3], conn[e, 3]] = 1.0
                Selast[conn[e, 6], conn[e, 6]] = 1.0
                Selast[conn[e, 7], conn[e, 7]] = 1.0

                Selast[conn2[e, 2], :] = 0.0
                Selast[conn2[e, 3], :] = 0.0
                Selast[conn2[e, 6], :] = 0.0
                Selast[conn2[e, 7], :] = 0.0

                Selast[conn2[e, 2], conn2[e, 2]] = 1.0
                Selast[conn2[e, 3], conn2[e, 3]] = 1.0
                Selast[conn2[e, 6], conn2[e, 6]] = 1.0
                Selast[conn2[e, 7], conn2[e, 7]] = 1.0

                Selast[conn3[e, 2], :] = 0.0
                Selast[conn3[e, 3], :] = 0.0
                Selast[conn3[e, 6], :] = 0.0
                Selast[conn3[e, 7], :] = 0.0

                Selast[conn3[e, 2], conn3[e, 2]] = 1.0
                Selast[conn3[e, 3], conn3[e, 3]] = 1.0
                Selast[conn3[e, 6], conn3[e, 6]] = 1.0
                Selast[conn3[e, 7], conn3[e, 7]] = 1.0

                Melast[conn[e, 2], :] = 0.0
                Melast[conn[e, 3], :] = 0.0
                Melast[conn[e, 6], :] = 0.0
                Melast[conn[e, 7], :] = 0.0

                Melast[conn[e, 2], conn[e, 2]] = 1.0
                Melast[conn[e, 3], conn[e, 3]] = 1.0
                Melast[conn[e, 6], conn[e, 6]] = 1.0
                Melast[conn[e, 7], conn[e, 7]] = 1.0

                Melast[conn2[e, 2], :] = 0.0
                Melast[conn2[e, 3], :] = 0.0
                Melast[conn2[e, 6], :] = 0.0
                Melast[conn2[e, 7], :] = 0.0

                Melast[conn2[e, 2], conn2[e, 2]] = 1.0
                Melast[conn2[e, 3], conn2[e, 3]] = 1.0
                Melast[conn2[e, 6], conn2[e, 6]] = 1.0
                Melast[conn2[e, 7], conn2[e, 7]] = 1.0

                Melast[conn3[e, 2], :] = 0.0
                Melast[conn3[e, 3], :] = 0.0
                Melast[conn3[e, 6], :] = 0.0
                Melast[conn3[e, 7], :] = 0.0

                Melast[conn3[e, 2], conn3[e, 2]] = 1.0
                Melast[conn3[e, 3], conn3[e, 3]] = 1.0
                Melast[conn3[e, 6], conn3[e, 6]] = 1.0
                Melast[conn3[e, 7], conn3[e, 7]] = 1.0

                Relast[conn[e, 2]] = u_bc[2, 0]
                Relast[conn[e, 3]] = u_bc[3, 0]
                Relast[conn[e, 6]] = u_bc[6, 0]
                Relast[conn[e, 7]] = u_bc[7, 0]

                Relast[conn2[e, 2]] = u_bc[2, 1]
                Relast[conn2[e, 3]] = u_bc[3, 1]
                Relast[conn2[e, 6]] = u_bc[6, 1]
                Relast[conn2[e, 7]] = u_bc[7, 1]

                Relast[conn3[e, 2]] = u_bc[2, 2]
                Relast[conn3[e, 3]] = u_bc[3, 2]
                Relast[conn3[e, 6]] = u_bc[6, 2]
                Relast[conn3[e, 7]] = u_bc[7, 2]

        return Selast, Melast, Relast

#####################################################################################################################

#Thermal Boundary Condition - Lower Z Surface
    def thermal_boundary_bottom_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        t_bc = np.zeros(8)
        x_load = np.zeros((8,3))

        Ne = len(conn)

        # Lower z Surface - bottom

        for e in range(0, (Nx - 1) ** 2):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                t_bc[ii] = np.dot(load_state, x_load[ii])

            Stemp[conn[e, 0], :] = 0.0
            Stemp[conn[e, 1], :] = 0.0
            Stemp[conn[e, 2], :] = 0.0
            Stemp[conn[e, 3], :] = 0.0

            Stemp[conn[e, 0], conn[e, 0]] = 1.0
            Stemp[conn[e, 1], conn[e, 1]] = 1.0
            Stemp[conn[e, 2], conn[e, 2]] = 1.0
            Stemp[conn[e, 3], conn[e, 3]] = 1.0

            Mtemp[conn[e, 0], :] = 0.0
            Mtemp[conn[e, 1], :] = 0.0
            Mtemp[conn[e, 2], :] = 0.0
            Mtemp[conn[e, 3], :] = 0.0

            Mtemp[conn[e, 0], conn[e, 0]] = 1.0
            Mtemp[conn[e, 1], conn[e, 1]] = 1.0
            Mtemp[conn[e, 2], conn[e, 2]] = 1.0
            Mtemp[conn[e, 3], conn[e, 3]] = 1.0

            Rtemp[conn[e, 0]] = t_bc[0]
            Rtemp[conn[e, 1]] = t_bc[1]
            Rtemp[conn[e, 2]] = t_bc[2]
            Rtemp[conn[e, 3]] = t_bc[3]

        return Stemp, Mtemp, Rtemp

#####################################################################################################################
#Thermal Boundary Condition - Top Z Surface

    def thermal_boundary_top_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        t_bc = np.zeros(8)
        x_load = np.zeros((8,3))

        Ne = len(conn)

        # Top z Surface

        for e in range(Ne - (Nx - 1) ** 2, Ne):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                t_bc[ii] = np.dot(load_state, x_load[ii])

            Stemp[conn[e, 4], :] = 0.0
            Stemp[conn[e, 5], :] = 0.0
            Stemp[conn[e, 6], :] = 0.0
            Stemp[conn[e, 7], :] = 0.0

            Stemp[conn[e, 4], conn[e, 4]] = 1.0
            Stemp[conn[e, 5], conn[e, 5]] = 1.0
            Stemp[conn[e, 6], conn[e, 6]] = 1.0
            Stemp[conn[e, 7], conn[e, 7]] = 1.0

            Mtemp[conn[e, 4], :] = 0.0
            Mtemp[conn[e, 5], :] = 0.0
            Mtemp[conn[e, 6], :] = 0.0
            Mtemp[conn[e, 7], :] = 0.0

            Mtemp[conn[e, 4], conn[e, 4]] = 1.0
            Mtemp[conn[e, 5], conn[e, 5]] = 1.0
            Mtemp[conn[e, 6], conn[e, 6]] = 1.0
            Mtemp[conn[e, 7], conn[e, 7]] = 1.0

            Rtemp[conn[e, 4]] = t_bc[4]
            Rtemp[conn[e, 5]] = t_bc[5]
            Rtemp[conn[e, 6]] = t_bc[6]
            Rtemp[conn[e, 7]] = t_bc[7]

        return Stemp, Mtemp, Rtemp

#####################################################################################################################
#Thermal Boundary Condition - Back X Surface

    def thermal_boundary_back_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        t_bc = np.zeros(8)
        x_load = np.zeros((8,3))

        Ne = len(conn)

        # Back x surface - near the axis

        for e in range(0, (Ne), Nx - 1):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                t_bc[ii] = np.dot(load_state, x_load[ii])

            Stemp[conn[e, 0], :] = 0.0
            Stemp[conn[e, 3], :] = 0.0
            Stemp[conn[e, 4], :] = 0.0
            Stemp[conn[e, 7], :] = 0.0

            Stemp[conn[e, 0], conn[e, 0]] = 1.0
            Stemp[conn[e, 3], conn[e, 3]] = 1.0
            Stemp[conn[e, 4], conn[e, 4]] = 1.0
            Stemp[conn[e, 7], conn[e, 7]] = 1.0

            Mtemp[conn[e, 0], :] = 0.0
            Mtemp[conn[e, 3], :] = 0.0
            Mtemp[conn[e, 4], :] = 0.0
            Mtemp[conn[e, 7], :] = 0.0

            Mtemp[conn[e, 0], conn[e, 0]] = 1.0
            Mtemp[conn[e, 3], conn[e, 3]] = 1.0
            Mtemp[conn[e, 4], conn[e, 4]] = 1.0
            Mtemp[conn[e, 7], conn[e, 7]] = 1.0

            Rtemp[conn[e, 0]] = t_bc[0]
            Rtemp[conn[e, 3]] = t_bc[3]
            Rtemp[conn[e, 4]] = t_bc[4]
            Rtemp[conn[e, 7]] = t_bc[7]

        return Stemp, Mtemp, Rtemp

#####################################################################################################################
#Thermal Boundary Condition - Front X Surface (+1.0)

    def thermal_boundary_front_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        t_bc = np.zeros(8)
        x_load = np.zeros((8,3))

        Ne = len(conn)

        # front x surface - away from the axis

        for e in range(Nx - 2, Ne, Nx - 1):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            x_load[:, 0] = x1
            x_load[:, 1] = x2
            x_load[:, 2] = x3

            for ii in range(0, 8):
                t_bc[ii] = np.dot(load_state, x_load[ii])

            Stemp[conn[e, 1], :] = 0.0
            Stemp[conn[e, 2], :] = 0.0
            Stemp[conn[e, 5], :] = 0.0
            Stemp[conn[e, 6], :] = 0.0

            Stemp[conn[e, 1], conn[e, 1]] = 1.0
            Stemp[conn[e, 2], conn[e, 2]] = 1.0
            Stemp[conn[e, 5], conn[e, 5]] = 1.0
            Stemp[conn[e, 6], conn[e, 6]] = 1.0

            Mtemp[conn[e, 1], :] = 0.0
            Mtemp[conn[e, 2], :] = 0.0
            Mtemp[conn[e, 5], :] = 0.0
            Mtemp[conn[e, 6], :] = 0.0

            Mtemp[conn[e, 1], conn[e, 1]] = 1.0
            Mtemp[conn[e, 2], conn[e, 2]] = 1.0
            Mtemp[conn[e, 5], conn[e, 5]] = 1.0
            Mtemp[conn[e, 6], conn[e, 6]] = 1.0

            Rtemp[conn[e, 1]] = t_bc[1]
            Rtemp[conn[e, 2]] = t_bc[2]
            Rtemp[conn[e, 5]] = t_bc[5]
            Rtemp[conn[e, 6]] = t_bc[6]

        return Stemp, Mtemp, Rtemp

#####################################################################################################################
#Thermal Boundary Condition - Left Y Surface (-1.0)

    def thermal_boundary_left_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        t_bc = np.zeros(8)
        x_load = np.zeros((8,3))

        Ne = len(conn)

        # Left y surface - bottom (z = -1.0)

        for ee in range(0, Nx - 1):
            for e in range(ee * (Nx - 1) ** 2, ee * (Nx - 1) ** 2 + (Nx - 1)):

                x1[:] = np.array(nodes[conn[e, :], 0])
                x2[:] = np.array(nodes[conn[e, :], 1])
                x3[:] = np.array(nodes[conn[e, :], 2])

                x_load[:, 0] = x1
                x_load[:, 1] = x2
                x_load[:, 2] = x3

                for ii in range(0, 8):
                    t_bc[ii] = np.dot(load_state, x_load[ii])

                Stemp[conn[e, 0], :] = 0.0
                Stemp[conn[e, 1], :] = 0.0
                Stemp[conn[e, 4], :] = 0.0
                Stemp[conn[e, 5], :] = 0.0

                Stemp[conn[e, 0], conn[e, 0]] = 1.0
                Stemp[conn[e, 1], conn[e, 1]] = 1.0
                Stemp[conn[e, 4], conn[e, 4]] = 1.0
                Stemp[conn[e, 5], conn[e, 5]] = 1.0

                Mtemp[conn[e, 0], :] = 0.0
                Mtemp[conn[e, 1], :] = 0.0
                Mtemp[conn[e, 4], :] = 0.0
                Mtemp[conn[e, 5], :] = 0.0

                Mtemp[conn[e, 0], conn[e, 0]] = 1.0
                Mtemp[conn[e, 1], conn[e, 1]] = 1.0
                Mtemp[conn[e, 4], conn[e, 4]] = 1.0
                Mtemp[conn[e, 5], conn[e, 5]] = 1.0

                Rtemp[conn[e, 0]] = t_bc[0]
                Rtemp[conn[e, 1]] = t_bc[1]
                Rtemp[conn[e, 4]] = t_bc[4]
                Rtemp[conn[e, 5]] = t_bc[5]

        return Stemp, Mtemp, Rtemp

#####################################################################################################################
#Thermal Boundary Condition - Right Y Surface (+1.0)

    def thermal_boundary_right_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state):

        x1 = np.zeros(8)
        x2 = np.zeros(8)
        x3 = np.zeros(8)
        t_bc = np.zeros(8)
        x_load = np.zeros((8,3))

        Ne = len(conn)

        # Right y surface, z = 1.0

        for ee in range(0, Nx - 1):
            for e in range(ee * (Nx - 1) ** 2 + ((Nx - 1) ** 2 - (Nx - 1)), (ee + 1) * (Nx - 1) ** 2):

                x1[:] = np.array(nodes[conn[e, :], 0])
                x2[:] = np.array(nodes[conn[e, :], 1])
                x3[:] = np.array(nodes[conn[e, :], 2])

                x_load[:, 0] = x1
                x_load[:, 1] = x2
                x_load[:, 2] = x3

                for ii in range(0, 8):
                    t_bc[ii] = np.dot(load_state, x_load[ii])

                Stemp[conn[e, 2], :] = 0.0
                Stemp[conn[e, 3], :] = 0.0
                Stemp[conn[e, 6], :] = 0.0
                Stemp[conn[e, 7], :] = 0.0

                Stemp[conn[e, 2], conn[e, 2]] = 1.0
                Stemp[conn[e, 3], conn[e, 3]] = 1.0
                Stemp[conn[e, 6], conn[e, 6]] = 1.0
                Stemp[conn[e, 7], conn[e, 7]] = 1.0

                Mtemp[conn[e, 2], :] = 0.0
                Mtemp[conn[e, 3], :] = 0.0
                Mtemp[conn[e, 6], :] = 0.0
                Mtemp[conn[e, 7], :] = 0.0

                Mtemp[conn[e, 2], conn[e, 2]] = 1.0
                Mtemp[conn[e, 3], conn[e, 3]] = 1.0
                Mtemp[conn[e, 6], conn[e, 6]] = 1.0
                Mtemp[conn[e, 7], conn[e, 7]] = 1.0

                Rtemp[conn[e, 2]] = t_bc[2]
                Rtemp[conn[e, 3]] = t_bc[3]
                Rtemp[conn[e, 6]] = t_bc[6]
                Rtemp[conn[e, 7]] = t_bc[7]

        return Stemp, Mtemp, Rtemp

#####################################################################################################################



#Generate the mesh
    # Nx = 9
    [nodes, conn] = cmesh(Nx)

#Add second and third component to the connectivity to track 2nd and 3rd component of the displacement solution
    conn2 = conn + np.max(conn) + 1
    conn3 = conn2 + np.max(conn) + 1

    n_nodes = int(Nx**3)
    n_elem = int((Nx-1)**3)

    Ne = len(conn)
    NP = len(nodes)

#Thermal FEM matrices

    Stemp = sps.lil_matrix((NP, NP)) #Stiffness Matrix
    Mtemp = sps.lil_matrix((NP, NP)) #Mass Matrix
    fS1temp = sps.lil_matrix((NP, NP))
    fM1temp = sps.lil_matrix((NP, NP))
    fS2temp = sps.lil_matrix((NP, NP))
    fS3temp = sps.lil_matrix((NP, NP))

    Rtemp = np.zeros((NP,1))
    fR1temp = np.zeros((8,1))
    fR2temp = np.zeros((8,1))
    fR3temp = np.zeros((8,1))

#Elasticity FEM matrices
    Selast = sps.lil_matrix((3*NP, 3*NP)) #Stiffness Matrix
    Melast = sps.lil_matrix((3*NP, 3*NP)) #Mass Matrix

    fS1elast = sps.lil_matrix((24, 24))
    fM1elast = sps.lil_matrix((24, 24))
    fS2elast = sps.lil_matrix((24, 24))
    fS3elast = sps.lil_matrix((24, 24))

    Relast = np.zeros((3*NP,1))
    fR1elast = np.zeros((24,1))
    fR2elast = np.zeros((24,1))
    fR3elast = np.zeros((24,1))

#Stress/strain storage matrices (per timestep)
    strainfull0 = np.zeros((n_elem,48))
    stressfull0 = np.zeros((n_elem,48))
    thermstrainfull0 = np.zeros((n_elem,48))
    plasticfull0 = np.zeros((n_elem,48))

    strainfull1 = np.zeros((n_elem,48))
    stressfull1 = np.zeros((n_elem,48))
    thermstrainfull1 = np.zeros((n_elem,48))
    plasticfull1 = np.zeros((n_elem,48))

    therm_deriv0 = np.zeros((n_elem, 24))
    therm_flux0 = np.zeros((n_elem, 24))

    stressmat = np.zeros((n_nodes,6))
    strainmat = np.zeros((n_nodes,6))
    thermstrainmat = np.zeros((n_nodes,6))

    thermderivmat = np.zeros((n_nodes,3))

    thermfluxmat = np.zeros((n_nodes,3))

    plastic_0 = np.zeros((8,6))



####Additional vectors/matrices
    F = np.zeros((3,3))

    x1 = np.zeros(8)
    x2 = np.zeros(8)
    x3 = np.zeros(8)


#Effective Property Matrices
    eff_mech_strain_avg = np.zeros((36,36))
    eff_therm_deriv_avg = np.zeros((9,9))
    zero_6 = np.zeros((1,6))
    strain_avg = np.zeros(6)
    temp_deriv_avg = np.zeros(3)

#mech_states = np.array([beta, beta, beta, beta, beta, beta])
    mech_states_stress = np.zeros(36)
    temp_states = np.zeros(9)

    u_bc = np.zeros((8,3))
    x_load = np.zeros((8,3))
    t_bc = np.zeros(8)

    #beta = 1.01 #mechanical & thermal

#####################################################################################################################
#Initial Conditions
    ufull = np.zeros((len(nodes),3))
    ufullnew = np.zeros((len(nodes),3))

    stressmat0 = np.zeros((n_nodes,6))
    strainmat0 = np.zeros((n_nodes,6))
    thermstrainmat0 = np.zeros((n_nodes,6))


    Tzero = 300.0
    delt = 0.0
    gamma = 10e-5
    arate = 0.0
    damage = 1.0
    damage_stress = 100e12
    yieldstress = 100e12

    T_fem = np.ones(len(Rtemp))*Tzero




#####################################################################################################################
# Load State Stepping loop for Mechanical Loads (Separate Mechanical and Thermal loops)
    for t in range(0, 6):  # 6 different load state calculations for elasticity

        # zero out the matrices and vectors
        Rtemp = np.zeros((NP, 1))
        Stemp = sps.lil_matrix((NP, NP))  # Stiffness Matrix
        Mtemp = sps.lil_matrix((NP, NP))  # Mass Matrix

        Relast = np.zeros((3 * NP, 1))
        Selast = sps.lil_matrix((3 * NP, 3 * NP))  # Stiffness Matrix
        Melast = sps.lil_matrix((3 * NP, 3 * NP))  # Mass Matrix

        # Define the load state
        # Mechanical Load States
        # Load State 1: beta on bottom z, load state2: beta on top z, load state 3: beta on left x, load state 4: beta on right x
        # load state 5: beta on back y, load state 6: beta on front y

        # Elasticity Problem
        for e in range(0, Ne):

            x1[:] = np.array(nodes[conn[e, :], 0])
            x2[:] = np.array(nodes[conn[e, :], 1])
            x3[:] = np.array(nodes[conn[e, :], 2])

            # Integral 1 - Stiffness Matrix - Essential Integral

            fS1elast = elast1_spheres.stiff1(x1, x2, x3, damage, Etensor_matrix, Etensor_sphere, sphere_table)

            for i in range(0, 8):
                for j in range(0, 8):
                    Selast[conn[e, i], conn[e, j]] = Selast[conn[e, i], conn[e, j]] + fS1elast[i, j]
                    Selast[conn2[e, i], conn2[e, j]] = Selast[conn2[e, i], conn2[e, j]] + fS1elast[i + 8, j + 8]
                    Selast[conn3[e, i], conn3[e, j]] = Selast[conn3[e, i], conn3[e, j]] + fS1elast[i + 16, j + 16]

        if t == 0:
            # load state 1

            load_state = np.array([[beta, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

            # Lower z Surface - bottom

            [Selast, Melast, Relast] = mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Top Z Surface

            [Selast, Melast, Relast] = mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Back x surface - near the axis

            [Selast, Melast, Relast] = mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # front x surface - away from the axis

            [Selast, Melast, Relast] = mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Left y surface - bottom (z = -1.0)

            [Selast, Melast, Relast] = mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Right y surface, z = 1.0

            [Selast, Melast, Relast] = mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

        if t == 1:
            # load state 2

            load_state = np.array([[0.0, 0.0, 0.0], [0.0, beta, 0.0], [0.0, 0.0, 0.0]])

            # Lower z Surface - bottom

            [Selast, Melast, Relast] = mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Top Z Surface

            [Selast, Melast, Relast] = mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Back x surface - near the axis

            [Selast, Melast, Relast] = mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # front x surface - away from the axis

            [Selast, Melast, Relast] = mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Left y surface - bottom (z = -1.0)

            [Selast, Melast, Relast] = mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Right y surface, z = 1.0

            [Selast, Melast, Relast] = mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

        if t == 2:
            # load state 3

            load_state = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, beta]])

            # Lower z Surface - bottom

            [Selast, Melast, Relast] = mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Top Z Surface

            [Selast, Melast, Relast] = mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Back x surface - near the axis

            [Selast, Melast, Relast] = mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # front x surface - away from the axis

            [Selast, Melast, Relast] = mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Left y surface - bottom (z = -1.0)

            [Selast, Melast, Relast] = mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Right y surface, z = 1.0

            [Selast, Melast, Relast] = mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)



        if t == 3:
            # load state 4

            load_state = np.array([[0.0, beta, 0.0], [beta, 0.0, 0.0], [0.0, 0.0, 0.0]])

            # Lower z Surface - bottom

            [Selast, Melast, Relast] = mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Top Z Surface

            [Selast, Melast, Relast] = mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Back x surface - near the axis

            [Selast, Melast, Relast] = mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # front x surface - away from the axis

            [Selast, Melast, Relast] = mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Left y surface - bottom (z = -1.0)

            [Selast, Melast, Relast] = mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Right y surface, z = 1.0

            [Selast, Melast, Relast] = mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)


        if t == 4:
            # load state 5

            load_state = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, beta], [beta, 0.0, 0.0]])

            # Lower z Surface - bottom

            [Selast, Melast, Relast] = mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Top Z Surface

            [Selast, Melast, Relast] = mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Back x surface - near the axis

            [Selast, Melast, Relast] = mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # front x surface - away from the axis

            [Selast, Melast, Relast] = mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Left y surface - bottom (z = -1.0)

            [Selast, Melast, Relast] = mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Right y surface, z = 1.0

            [Selast, Melast, Relast] = mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)


        if t == 5:
            # load state 6

            load_state = np.array([[0.0, 0.0, beta], [0.0, 0.0, 0.0], [beta, 0.0, 0.0]])

            # Lower z Surface - bottom

            [Selast, Melast, Relast] = mech_boundary_bottom_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Top Z Surface

            [Selast, Melast, Relast] = mech_boundary_top_zz(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Back x surface - near the axis

            [Selast, Melast, Relast] = mech_boundary_back_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # front x surface - away from the axis

            [Selast, Melast, Relast] = mech_boundary_front_xx(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Left y surface - bottom (z = -1.0)

            [Selast, Melast, Relast] = mech_boundary_left_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

            # Right y surface, z = 1.0

            [Selast, Melast, Relast] = mech_boundary_right_yy(Nx, nodes, conn, Selast, Melast, Relast, load_state)

        ###Solve the elasticity problem - timestep forward to compute fields

        stepvecelast = Relast
        stepmatelast = Selast

        unewvec = spsolve(stepmatelast.tocsc(), stepvecelast)

        # velocity = (unewvec - uvec[:,0])/delt

        udir1 = unewvec[0:n_nodes]
        udir2 = unewvec[n_nodes:2 * n_nodes]
        udir3 = unewvec[2 * n_nodes:3 * n_nodes]

        # WATCH FOR THIS BUG HERE!!! - I think it's fixed with the new limits

        for aa in range(0, len(udir1)):
            if udir1[aa] < 10.0 ** -10 and udir1[aa] > -10.0 ** -10:
                udir1[aa] = 0.0
            if udir2[aa] < 10.0 ** -10 and udir2[aa] > -10.0 ** -10:
                udir2[aa] = 0.0
            if udir3[aa] < 10.0 ** -10 and udir3[aa] > -10.0 ** -10:
                udir3[aa] = 0.0

        ufullnew[:, 0] = udir1
        ufullnew[:, 1] = udir2
        ufullnew[:, 2] = udir3

        # Calculate the new stress state per node - include the temperature derivative on the interior

        # take out the interior to calculate the strain/temperature average on the interior only
        N_ext = 2 * (Nx - 1) ** 2 + 2 * ((Nx - 1) ** 2 - 2 * (Nx - 1)) + 2 * ((Nx - 1) ** 2 - 2 * (Nx - 1) - 2 * (Nx - 3))
        N_int = Ne - N_ext

        # Remove outermost layer of elements

        for eee in range(1, Nx - 2):
            for ee in range(0, Nx - 3):
                for e in range(eee * (Nx - 1) ** 2 + ee * (Nx - 1) + (Nx),
                               eee * (Nx - 1) ** 2 + ee * (Nx - 1) + (Nx + Nx - 3)):

                    # print 'e', e

                    Tfemvec1 = np.array(T_fem[conn[e, :]])

                    strainfull0[e, :] = strainfull1[e, :]
                    stressfull0[e, :] = stressfull1[e, :]
                    thermstrainfull0[e, :] = thermstrainfull1[e, :]
                    plasticfull0[e, :] = plasticfull1[e, :]

                    x1[:] = np.array(nodes[conn[e, :], 0])
                    x2[:] = np.array(nodes[conn[e, :], 1])
                    x3[:] = np.array(nodes[conn[e, :], 2])


                    uelemvec = np.array(ufullnew[conn[e, :], :])

                    [strainel, stressel, thermstrainel, plasticel,therm_deriv_el,
                     therm_flux_el, damage_new] = strain_temp2_spheres.nodes_strains_sphere(x1,x2,x3,uelemvec,Tfemvec1,Tzero,k_matrix_mat,
                                                                         k_sphere_mat,gamma,delt, damage,damage_stress,yieldstress,
                                                                         arate,Etensor_matrix,Etensor_sphere, plastic_0, sphere_table)


                    strainel = strainel.flatten()
                    stressel = stressel.flatten()
                    thermstrainel = thermstrainel.flatten()
                    plasticel = plasticel.flatten()
                    therm_deriv_el = therm_deriv_el.flatten()
                    therm_flux_el = therm_flux_el.flatten()

                    for aa in range(0, len(stressel)):
                        if strainel[aa] < 10.0 ** -10 and strainel[aa] > -10.0 ** -10:
                            strainel[aa] = 0.0
                        if stressel[aa] < 10.0 ** -10 and stressel[aa] > -10.0 ** -10:
                            stressel[aa] = 0.0
                        if thermstrainel[aa] < 10.0 ** -10 and thermstrainel[aa] > -10.0 ** -10:
                            thermstrainel[aa] = 0.0
                        if plasticel[aa] < 10.0 ** -10 and plasticel[aa] > -10.0 ** -10:
                            plasticel[aa] = 0.0
                    for aa in range(0, len(therm_deriv_el)):
                        if therm_deriv_el[aa] < 10.0 ** -10 and therm_deriv_el[aa] > -10.0 ** -10:
                            therm_deriv_el[aa] = 0.0

                    ##Dimensions mismatched - might have to transpose them
                    strainfull1[e, :] = strainel
                    stressfull1[e, :] = stressel
                    thermstrainfull1[e, :] = thermstrainel
                    plasticfull1[e, :] = plasticel
                    therm_deriv0[e, :] = therm_deriv_el
                    therm_flux0[e, :] = therm_flux_el

                    stressmat[conn[e, 0], :] = stressfull1[e, 0:6]
                    stressmat[conn[e, 1], :] = stressfull1[e, 6:12]
                    stressmat[conn[e, 2], :] = stressfull1[e, 12:18]
                    stressmat[conn[e, 3], :] = stressfull1[e, 18:24]
                    stressmat[conn[e, 4], :] = stressfull1[e, 24:30]
                    stressmat[conn[e, 5], :] = stressfull1[e, 30:36]
                    stressmat[conn[e, 6], :] = stressfull1[e, 36:42]
                    stressmat[conn[e, 7], :] = stressfull1[e, 42:48]

                    strainmat[conn[e, 0], :] = strainfull1[e, 0:6]
                    strainmat[conn[e, 1], :] = strainfull1[e, 6:12]
                    strainmat[conn[e, 2], :] = strainfull1[e, 12:18]
                    strainmat[conn[e, 3], :] = strainfull1[e, 18:24]
                    strainmat[conn[e, 4], :] = strainfull1[e, 24:30]
                    strainmat[conn[e, 5], :] = strainfull1[e, 30:36]
                    strainmat[conn[e, 6], :] = strainfull1[e, 36:42]
                    strainmat[conn[e, 7], :] = strainfull1[e, 42:48]

                    thermstrainmat[conn[e, 0], :] = thermstrainfull1[e, 0:6]
                    thermstrainmat[conn[e, 1], :] = thermstrainfull1[e, 6:12]
                    thermstrainmat[conn[e, 2], :] = thermstrainfull1[e, 12:18]
                    thermstrainmat[conn[e, 3], :] = thermstrainfull1[e, 18:24]
                    thermstrainmat[conn[e, 4], :] = thermstrainfull1[e, 24:30]
                    thermstrainmat[conn[e, 5], :] = thermstrainfull1[e, 30:36]
                    thermstrainmat[conn[e, 6], :] = thermstrainfull1[e, 36:42]
                    thermstrainmat[conn[e, 7], :] = thermstrainfull1[e, 42:48]

                    thermderivmat[conn[e, 0], :] = therm_deriv0[e, 0:3]
                    thermderivmat[conn[e, 1], :] = therm_deriv0[e, 3:6]
                    thermderivmat[conn[e, 2], :] = therm_deriv0[e, 6:9]
                    thermderivmat[conn[e, 3], :] = therm_deriv0[e, 9:12]
                    thermderivmat[conn[e, 4], :] = therm_deriv0[e, 12:15]
                    thermderivmat[conn[e, 5], :] = therm_deriv0[e, 15:18]
                    thermderivmat[conn[e, 6], :] = therm_deriv0[e, 18:21]
                    thermderivmat[conn[e, 7], :] = therm_deriv0[e, 21:24]

                    thermfluxmat[conn[e, 0], :] = therm_flux0[e, 0:3]
                    thermfluxmat[conn[e, 1], :] = therm_flux0[e, 3:6]
                    thermfluxmat[conn[e, 2], :] = therm_flux0[e, 6:9]
                    thermfluxmat[conn[e, 3], :] = therm_flux0[e, 9:12]
                    thermfluxmat[conn[e, 4], :] = therm_flux0[e, 12:15]
                    thermfluxmat[conn[e, 5], :] = therm_flux0[e, 15:18]
                    thermfluxmat[conn[e, 6], :] = therm_flux0[e, 18:21]
                    thermfluxmat[conn[e, 7], :] = therm_flux0[e, 21:24]

        # Calculate the average strain in each direction
        strain_avg = [np.sum(strainmat[:, 0]) / N_int, np.sum(strainmat[:, 1]) / N_int, np.sum(strainmat[:, 2]) / N_int,
                      np.sum(strainmat[:, 3]) / N_int, np.sum(strainmat[:, 4]) / N_int, np.sum(strainmat[:, 5]) / N_int]

        stress_avg = [np.sum(stressmat[:, 0]) / N_int, np.sum(stressmat[:, 1]) / N_int, np.sum(stressmat[:, 2]) / N_int,
                      np.sum(stressmat[:, 3]) / N_int, np.sum(stressmat[:, 4]) / N_int, np.sum(stressmat[:, 5]) / N_int]

        # t is dummy variable to number the load state
        eff_mech_strain_avg[t * 6 + 0, 0:6] = strain_avg
        eff_mech_strain_avg[t * 6 + 1, 6:12] = strain_avg
        eff_mech_strain_avg[t * 6 + 2, 12:18] = strain_avg
        eff_mech_strain_avg[t * 6 + 3, 18:24] = strain_avg
        eff_mech_strain_avg[t * 6 + 4, 24:30] = strain_avg
        eff_mech_strain_avg[t * 6 + 5, 30:36] = strain_avg

        mech_states_stress[(t * 6):(t * 6 + 6)] = stress_avg

        # print 'Mech', t
        #
        # print 'calculation time', datetime.now() - startTime



    ################################################################################################################################################
    #Effective property calculations with matrices
    #Elasticity



    #Symmetry of Elasticity Tensor
    eff_mech_strain_avg_sym = np.zeros((36,21))
    mech_states_stress_sym = np.zeros(21)


    #Matching Strains
    #Row 0
    eff_mech_strain_avg_sym[:,0] = eff_mech_strain_avg[:,0]
    eff_mech_strain_avg_sym[:,1] = eff_mech_strain_avg[:,1] + eff_mech_strain_avg[:,6]
    eff_mech_strain_avg_sym[:,2] = eff_mech_strain_avg[:,2] + eff_mech_strain_avg[:,12]
    eff_mech_strain_avg_sym[:,3] = eff_mech_strain_avg[:,3] + eff_mech_strain_avg[:,18]
    eff_mech_strain_avg_sym[:,4] = eff_mech_strain_avg[:,4] + eff_mech_strain_avg[:,24]
    eff_mech_strain_avg_sym[:,5] = eff_mech_strain_avg[:,5] + eff_mech_strain_avg[:,30]
    #Row 1
    eff_mech_strain_avg_sym[:,6] = eff_mech_strain_avg[:,7]
    eff_mech_strain_avg_sym[:,7] = eff_mech_strain_avg[:,8] + eff_mech_strain_avg[:,13]
    eff_mech_strain_avg_sym[:,8] = eff_mech_strain_avg[:,9] + eff_mech_strain_avg[:,19]
    eff_mech_strain_avg_sym[:,9] = eff_mech_strain_avg[:,10] + eff_mech_strain_avg[:,25]
    eff_mech_strain_avg_sym[:,10] = eff_mech_strain_avg[:,11] + eff_mech_strain_avg[:,31]
    #Row 2
    eff_mech_strain_avg_sym[:,11] = eff_mech_strain_avg[:,14]
    eff_mech_strain_avg_sym[:,12] = eff_mech_strain_avg[:,15] + eff_mech_strain_avg[:,20]
    eff_mech_strain_avg_sym[:,13] = eff_mech_strain_avg[:,16] + eff_mech_strain_avg[:,26]
    eff_mech_strain_avg_sym[:,14] = eff_mech_strain_avg[:,17] + eff_mech_strain_avg[:,32]
    #Row 3
    eff_mech_strain_avg_sym[:,15] = eff_mech_strain_avg[:,21]
    eff_mech_strain_avg_sym[:,16] = eff_mech_strain_avg[:,22] + eff_mech_strain_avg[:,27]
    eff_mech_strain_avg_sym[:,17] = eff_mech_strain_avg[:,23] + eff_mech_strain_avg[:,33]
    #Row 4
    eff_mech_strain_avg_sym[:,18] = eff_mech_strain_avg[:,28]
    eff_mech_strain_avg_sym[:,19] = eff_mech_strain_avg[:,29] + eff_mech_strain_avg[:,34]
    #Row 5
    eff_mech_strain_avg_sym[:,20] = eff_mech_strain_avg[:,35]


    #Matching Stresses
    #Row 0
    mech_states_stress_sym[0] = mech_states_stress[0]
    mech_states_stress_sym[1] = mech_states_stress[1] + mech_states_stress[6]
    mech_states_stress_sym[2] = mech_states_stress[2] + mech_states_stress[12]
    mech_states_stress_sym[3] = mech_states_stress[3] + mech_states_stress[18]
    mech_states_stress_sym[4] = mech_states_stress[4] + mech_states_stress[24]
    mech_states_stress_sym[5] = mech_states_stress[5] + mech_states_stress[30]
    #Row 1
    mech_states_stress_sym[6] = mech_states_stress[7]
    mech_states_stress_sym[7] = mech_states_stress[8] + mech_states_stress[13]
    mech_states_stress_sym[8] = mech_states_stress[9] + mech_states_stress[19]
    mech_states_stress_sym[9] = mech_states_stress[10] + mech_states_stress[25]
    mech_states_stress_sym[10] = mech_states_stress[11] + mech_states_stress[31]
    #Row 2
    mech_states_stress_sym[11] = mech_states_stress[14]
    mech_states_stress_sym[12] = mech_states_stress[15] + mech_states_stress[20]
    mech_states_stress_sym[13] = mech_states_stress[16] + mech_states_stress[26]
    mech_states_stress_sym[14] = mech_states_stress[17] + mech_states_stress[32]
    #Row 3
    mech_states_stress_sym[15] = mech_states_stress[21]
    mech_states_stress_sym[16] = mech_states_stress[22] + mech_states_stress[27]
    mech_states_stress_sym[17] = mech_states_stress[23] + mech_states_stress[33]
    #Row 4
    mech_states_stress_sym[18] = mech_states_stress[28]
    mech_states_stress_sym[19] = mech_states_stress[29] + mech_states_stress[34]
    #Row 5
    mech_states_stress_sym[20] = mech_states_stress[35]


    #Mechanical Loop
    eff_mech_vec = np.linalg.lstsq(eff_mech_strain_avg, mech_states_stress)

    eff_mech_vec_sym = np.linalg.lstsq(eff_mech_strain_avg_sym, mech_states_stress)


    #print eff_mech_vec[0]

    A = np.asarray(eff_mech_vec[0])

    AA = np.asarray(eff_mech_vec_sym[0])

    A1 = np.zeros((6,6))








    A2 = np.zeros((6,6))

    A2[0,0:6] = AA[0:6]
    A2[1,0] = AA[1]
    A2[1,1:6] = AA[6:11]
    A2[2,0] = AA[2]
    A2[2,1] = AA[7]
    A2[2,2:6] = AA[11:15]
    A2[3,0] = AA[3]
    A2[3,1] = AA[8]
    A2[3,2] = AA[12]
    A2[3,3:6] = AA[15:18]
    A2[4,0] = AA[4]
    A2[4,1] = AA[9]
    A2[4,2] = AA[13]
    A2[4,3] = AA[16]
    A2[4,4:6] = AA[18:20]
    A2[5,0] = AA[5]
    A2[5,1] = AA[10]
    A2[5,2] = AA[14]
    A2[5,3] = AA[17]
    A2[5,4] = AA[19]
    A2[5,5] = AA[20]

    A1[0,0:6] = A[0:6]
    A1[1,0:6] = A[6:12]
    A1[2,0:6] = A[12:18]
    A1[3,0:6] = A[18:24]
    A1[4,0:6] = A[24:30]
    A1[5,0:6] = A[30:36]

    w,v = np.linalg.eig(A1)

    #Tranpose to make matrix symmetric in approach 1
    A1 = 0.5*(A1 + np.transpose(A1))

    w2,v2 = np.linalg.eig(A2)

    # print 'Elasticity'
    #
    # #print 'Independent', A1
    # print 'Symmetry', A2

    # print 'Eigenvalues'
    # #print w
    # print w2

################################################################################################################################################
#Thermal Problem

#Thermal Problem and its states

    k11_matrix = k_matrix_mat[0,0]
    k22_matrix = k_matrix_mat[1,1]
    k33_matrix = k_matrix_mat[2,2]

    k11_sphere = k_sphere_mat[0,0]
    k22_sphere = k_sphere_mat[1,1]
    k33_sphere = k_sphere_mat[2,2]

    for t in range(0,3):


        #zero out the matrices and vectors
        Rtemp = np.zeros((NP,1))
        Stemp = sps.lil_matrix((NP, NP)) #Stiffness Matrix
        Mtemp = sps.lil_matrix((NP, NP)) #Mass Matrix



        #Define the load state
        #Thermal Load States
        #Load State 1: q on bottom z, load state 2: q on top , load state 3: beta on left x, load state 4: beta on right x
        #load state 5: beta on back y, load state 6: beta on front y


        #Thermal Problem
        for e in range(0,Ne):

            #avtemp = (sum(T[conn[e,:]])/8)

            x1[:] = np.array(nodes[conn[e,:],0])
            x2[:] = np.array(nodes[conn[e,:],1])
            x3[:] = np.array(nodes[conn[e,:],2])

            # Integral 1 - Stiffness Matrix

            fS1temp = thermo1_spheres.stiffthermo1(x1, x2, x3, k11_matrix, k22_matrix, k33_matrix,
                                                  k11_sphere, k22_sphere, k33_sphere, sphere_table)
            for i in range(0, 8):
                for j in range(0, 8):
                    Stemp[conn[e, i], conn[e, j]] = Stemp[conn[e, i], conn[e, j]] + fS1temp[i, j]


        if t == 0:

            load_state = np.array([beta*Tzero, 0.0, 0.0])

            # Lower z Surface - bottom

            [Stemp, Mtemp, Rtemp] = thermal_boundary_bottom_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Top Z Surface

            [Stemp, Mtemp, Rtemp] = thermal_boundary_top_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Back x surface - near the axis

            [Stemp, Mtemp, Rtemp] = thermal_boundary_back_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # front x surface - away from the axis

            [Stemp, Mtemp, Rtemp] = thermal_boundary_front_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Left y surface - bottom (z = -1.0)

            [Stemp, Mtemp, Rtemp] = thermal_boundary_left_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Right y surface, z = 1.0

            [Stemp, Mtemp, Rtemp] = thermal_boundary_right_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)



        if t == 1:

            load_state = np.array([0.0, beta * Tzero, 0.0])

            # Lower z Surface - bottom

            [Stemp, Mtemp, Rtemp] = thermal_boundary_bottom_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Top Z Surface

            [Stemp, Mtemp, Rtemp] = thermal_boundary_top_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Back x surface - near the axis

            [Stemp, Mtemp, Rtemp] = thermal_boundary_back_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # front x surface - away from the axis

            [Stemp, Mtemp, Rtemp] = thermal_boundary_front_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Left y surface - bottom (z = -1.0)

            [Stemp, Mtemp, Rtemp] = thermal_boundary_left_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Right y surface, z = 1.0

            [Stemp, Mtemp, Rtemp] = thermal_boundary_right_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)



        if t == 2:

            load_state = np.array([0.0, 0.0, beta * Tzero])

            # Lower z Surface - bottom

            [Stemp, Mtemp, Rtemp] = thermal_boundary_bottom_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Top Z Surface

            [Stemp, Mtemp, Rtemp] = thermal_boundary_top_zz(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Back x surface - near the axis

            [Stemp, Mtemp, Rtemp] = thermal_boundary_back_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # front x surface - away from the axis

            [Stemp, Mtemp, Rtemp] = thermal_boundary_front_xx(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Left y surface - bottom (z = -1.0)

            [Stemp, Mtemp, Rtemp] = thermal_boundary_left_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)

            # Right y surface, z = 1.0

            [Stemp, Mtemp, Rtemp] = thermal_boundary_right_yy(Nx, nodes, conn, Stemp, Mtemp, Rtemp, load_state)



        # Solve the linear system for the thermal problem

        stepmattemp = Stemp
        stepvectemp = Rtemp

        # print Rtemp

        Tnew = spsolve(stepmattemp.tocsc(), stepvectemp)

        #Changed it here from T_fem[:] = Tnew to see if it goes faster
        T_fem = Tnew



        # Calculate the new stress state per node - include the temperature derivative on the interior

        # take out the interior to calculate the strain/temperature average on the interior only
        N_ext = 2 * (Nx - 1) ** 2 + 2 * ((Nx - 1) ** 2 - 2 * (Nx - 1)) + 2 * ((Nx - 1) ** 2 - 2 * (Nx - 1) - 2 * (Nx - 3))
        N_int = Ne - N_ext

        # Remove outermost layer of elements

        for eee in range(1, Nx - 2):
            for ee in range(0, Nx - 3):
                for e in range(eee * (Nx - 1) ** 2 + ee * (Nx - 1) + (Nx),
                               eee * (Nx - 1) ** 2 + ee * (Nx - 1) + (Nx + Nx - 3)):

                    # print 'e', e

                    Tfemvec1 = np.array(T_fem[conn[e, :]])

                    strainfull0[e, :] = strainfull1[e, :]
                    stressfull0[e, :] = stressfull1[e, :]
                    thermstrainfull0[e, :] = thermstrainfull1[e, :]
                    plasticfull0[e, :] = plasticfull1[e, :]

                    x1[:] = np.array(nodes[conn[e, :], 0])
                    x2[:] = np.array(nodes[conn[e, :], 1])
                    x3[:] = np.array(nodes[conn[e, :], 2])


                    uelemvec = np.array(ufullnew[conn[e, :], :])

                    [strainel, stressel, thermstrainel, plasticel,therm_deriv_el,
                     therm_flux_el, damage_new] = strain_temp2_spheres.nodes_strains_sphere(x1,x2,x3,uelemvec,Tfemvec1,Tzero,k_matrix_mat,
                                                                         k_sphere_mat,gamma,delt, damage,damage_stress,yieldstress,
                                                                         arate,Etensor_matrix,Etensor_sphere, plastic_0, sphere_table)


                    strainel = strainel.flatten()
                    stressel = stressel.flatten()
                    thermstrainel = thermstrainel.flatten()
                    plasticel = plasticel.flatten()
                    therm_deriv_el = therm_deriv_el.flatten()
                    therm_flux_el = therm_flux_el.flatten()

                    for aa in range(0, len(stressel)):
                        if strainel[aa] < 10.0 ** -10 and strainel[aa] > -10.0 ** -10:
                            strainel[aa] = 0.0
                        if stressel[aa] < 10.0 ** -10 and stressel[aa] > -10.0 ** -10:
                            stressel[aa] = 0.0
                        if thermstrainel[aa] < 10.0 ** -10 and thermstrainel[aa] > -10.0 ** -10:
                            thermstrainel[aa] = 0.0
                        if plasticel[aa] < 10.0 ** -10 and plasticel[aa] > -10.0 ** -10:
                            plasticel[aa] = 0.0
                    for aa in range(0, len(therm_deriv_el)):
                        if therm_deriv_el[aa] < 10.0 ** -10 and therm_deriv_el[aa] > -10.0 ** -10:
                            therm_deriv_el[aa] = 0.0

                    ##Dimensions mismatched - might have to transpose them
                    strainfull1[e, :] = strainel
                    stressfull1[e, :] = stressel
                    thermstrainfull1[e, :] = thermstrainel
                    plasticfull1[e, :] = plasticel
                    therm_deriv0[e, :] = therm_deriv_el
                    therm_flux0[e, :] = therm_flux_el

                    stressmat[conn[e, 0], :] = stressfull1[e, 0:6]
                    stressmat[conn[e, 1], :] = stressfull1[e, 6:12]
                    stressmat[conn[e, 2], :] = stressfull1[e, 12:18]
                    stressmat[conn[e, 3], :] = stressfull1[e, 18:24]
                    stressmat[conn[e, 4], :] = stressfull1[e, 24:30]
                    stressmat[conn[e, 5], :] = stressfull1[e, 30:36]
                    stressmat[conn[e, 6], :] = stressfull1[e, 36:42]
                    stressmat[conn[e, 7], :] = stressfull1[e, 42:48]

                    strainmat[conn[e, 0], :] = strainfull1[e, 0:6]
                    strainmat[conn[e, 1], :] = strainfull1[e, 6:12]
                    strainmat[conn[e, 2], :] = strainfull1[e, 12:18]
                    strainmat[conn[e, 3], :] = strainfull1[e, 18:24]
                    strainmat[conn[e, 4], :] = strainfull1[e, 24:30]
                    strainmat[conn[e, 5], :] = strainfull1[e, 30:36]
                    strainmat[conn[e, 6], :] = strainfull1[e, 36:42]
                    strainmat[conn[e, 7], :] = strainfull1[e, 42:48]

                    thermstrainmat[conn[e, 0], :] = thermstrainfull1[e, 0:6]
                    thermstrainmat[conn[e, 1], :] = thermstrainfull1[e, 6:12]
                    thermstrainmat[conn[e, 2], :] = thermstrainfull1[e, 12:18]
                    thermstrainmat[conn[e, 3], :] = thermstrainfull1[e, 18:24]
                    thermstrainmat[conn[e, 4], :] = thermstrainfull1[e, 24:30]
                    thermstrainmat[conn[e, 5], :] = thermstrainfull1[e, 30:36]
                    thermstrainmat[conn[e, 6], :] = thermstrainfull1[e, 36:42]
                    thermstrainmat[conn[e, 7], :] = thermstrainfull1[e, 42:48]

                    thermderivmat[conn[e, 0], :] = therm_deriv0[e, 0:3]
                    thermderivmat[conn[e, 1], :] = therm_deriv0[e, 3:6]
                    thermderivmat[conn[e, 2], :] = therm_deriv0[e, 6:9]
                    thermderivmat[conn[e, 3], :] = therm_deriv0[e, 9:12]
                    thermderivmat[conn[e, 4], :] = therm_deriv0[e, 12:15]
                    thermderivmat[conn[e, 5], :] = therm_deriv0[e, 15:18]
                    thermderivmat[conn[e, 6], :] = therm_deriv0[e, 18:21]
                    thermderivmat[conn[e, 7], :] = therm_deriv0[e, 21:24]

                    thermfluxmat[conn[e, 0], :] = therm_flux0[e, 0:3]
                    thermfluxmat[conn[e, 1], :] = therm_flux0[e, 3:6]
                    thermfluxmat[conn[e, 2], :] = therm_flux0[e, 6:9]
                    thermfluxmat[conn[e, 3], :] = therm_flux0[e, 9:12]
                    thermfluxmat[conn[e, 4], :] = therm_flux0[e, 12:15]
                    thermfluxmat[conn[e, 5], :] = therm_flux0[e, 15:18]
                    thermfluxmat[conn[e, 6], :] = therm_flux0[e, 18:21]
                    thermfluxmat[conn[e, 7], :] = therm_flux0[e, 21:24]



        # print 'Temp', t
        #
        # print 'calculation time', datetime.now() - startTime

        # Calculate the average temperature derivate in each spatial direction
        temp_deriv_avg = [np.sum(thermderivmat[:, 0]) / N_int, np.sum(thermderivmat[:, 1]) / N_int,
                          np.sum(thermderivmat[:, 2]) / N_int]

        temp_flux_avg = [np.sum(thermfluxmat[:, 0]) / N_int, np.sum(thermfluxmat[:, 1]) / N_int,
                         np.sum(thermfluxmat[:, 2]) / N_int]

        ##t denotes temperature load state
        eff_therm_deriv_avg[t * 3 + 0, 0:3] = temp_deriv_avg
        eff_therm_deriv_avg[t * 3 + 1, 3:6] = temp_deriv_avg
        eff_therm_deriv_avg[t * 3 + 2, 6:9] = temp_deriv_avg

        temp_states[(t * 3):(t * 3 + 3)] = temp_flux_avg

    ################################################################################################################################################
    #Effective property calculations with matrices
    #Thermal Problem


    #Setting Corresponding Conductivity Matrix Elements to 0
    #eff_therm_deriv_avg_2 = eff_therm_deriv_avg
    eff_therm_deriv_avg[:,1] = 0.0
    eff_therm_deriv_avg[:,2] = 0.0
    eff_therm_deriv_avg[:,3] = 0.0
    eff_therm_deriv_avg[:,5] = 0.0
    eff_therm_deriv_avg[:,6] = 0.0
    eff_therm_deriv_avg[:,7] = 0.0

    #Thermal Loop
    eff_temp_vec = np.linalg.lstsq(eff_therm_deriv_avg, temp_states)
    #eff_temp_vec_2 = np.linalg.lstsq(eff_therm_deriv_avg_2, temp_states)

    # print 'Thermal'
    # print eff_temp_vec[0]

    KK_mat = np.zeros((3,3))

    KK_0 = np.asarray(eff_temp_vec[0])

    KK_mat[0,0:3] = KK_0[0:3]
    KK_mat[1,0:3] = KK_0[3:6]
    KK_mat[2,0:3] = KK_0[6:9]

    return KK_mat, A2




################################################################################################################################################
#
#
# import numpy as np
#
# #Material Properties
#
# #Fiber and Matrix
# k11_fiber = 0.0 #W/mk
# k22_fiber = 0.0 #W/mk
# k33_fiber = 3.0 #W/mk
#
# k_fiber_mat = np.array([[k11_fiber, 0.0, 0.0],
#                         [0.0, k22_fiber, 0.0],
#                         [0.0, 0.0, k33_fiber]])
#
# k11_matrix = 137.0
# k22_matrix = 15.0
# k33_matrix = 221.0
#
# k_matrix_mat = np.array([[k11_matrix, 0.0, 0.0],
#                         [0.0, k22_matrix, 0.0],
#                         [0.0, 0.0, k33_matrix]])
#
# #Elasticity forces and parameters
#
# c11 = 7.0/3.0 #107.3e9 #Gpa
# c12 = 1.0/3.0 #28.3e9 #Gpa
# c44 = 1.0 #60.9e9 #Gpa
#
# #Fiber
# k_fiber = 260.0 #Gpa
# u_fiber = 40.0 #Gpa
# c11_fiber = (k_fiber + (4.0/3.0)*u_fiber)
# c12_fiber = (k_fiber - (2.0/3.0)*u_fiber)
# c44_fiber = u_fiber
#
# #Matrix
# k_matrix = 30.0 #Gpa
# u_matrix = 25.0 #Gpa
# c11_matrix = (k_matrix + (4.0/3.0)*u_matrix)
# c12_matrix = (k_matrix - (2.0/3.0)*u_matrix)
# c44_matrix = u_matrix
#
#
#
# Etensor_fiber = np.array([[c11_fiber, c12_fiber, c12_fiber, 0.0, 0.0, 0.0],
#                           [c12_fiber, c11_fiber, c12_fiber, 0.0, 0.0, 0.0],
#                           [c12_fiber, c12_fiber, c11_fiber, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, c44_fiber, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, c44_fiber, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, c44_fiber]])
#
# Etensor_matrix = np.array([[c11_matrix, c12_matrix, c12_matrix, 0.0, 0.0, 0.0],
#                           [c12_matrix, c11_matrix, c12_matrix, 0.0, 0.0, 0.0],
#                           [c12_matrix, c12_matrix, c11_matrix, 0.0, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, c44_matrix, 0.0, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, c44_matrix, 0.0],
#                           [0.0, 0.0, 0.0, 0.0, 0.0, c44_matrix]])
#
# fiber_table = np.zeros((150,4))
#
# beta = 1.01
#
# Nx = 9
#
#
# [Kmat_eff, Etensor_eff] = effective_prop_fem(Nx, k_matrix_mat, k_fiber_mat,Etensor_matrix, Etensor_fiber, fiber_table, beta)