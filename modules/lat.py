"""
    Author: Sergio Chaves Garcia-Mascaraque
    E-mail: sergiozteskate@gmail.com

    Module containing the functions to create the lattice points for a
    finite temporal lattice or an infinite temporal one.
"""

import numpy as np

__all__ = [ 'create_own_tinf_lattice',
            'create_own_fin_lattice' ]

'''
    10 de Mayo de 2018
    ---> Sergio Chaves
    ---> Javier Ugarrio

    ##############################################################################
    Function that creates a lattice in 5 dimensions using periodic or antiperiodic
    boundary conditions. It uses a parallelepiped of LX*LY*LZ*T_1*T_2 divided each
    side in N_X*N_Y*N_Z*N_T1*N_T2
    It asigns the partitions of each dimension according to the number of blocks you
    want to create, which could be used to divide the lattice in N_c cores in parallel.

    The lattice length MUST be a number so that L_i, T_i = 2^n with n integer and
    the number of cores must be also a number of cores so that
    N_c = 2^n <  L_X * L_Y * ... * T_2

    ##############################################################################
    '''

def create_own_tinf_lattice( L, N, mini_N, coords, mini_coords ):
    '''
        Create the lattice for a inifite time direction calculation.
        In this case, we do not have temporal component as it has been
        previously integrated out.

        Must be consistent that product of N[i] = number of cores used
    '''

    # Get the lenths of the lattice in the spatial direction
    L_X, L_Y, L_Z = int( L[0] ), int( L[1] ), int( L[2] )
    N_X, N_Y, N_Z = int( N[0] ), int( N[1] ), int( N[2] )

    # Get the sublenghts of the lattice to avoid memory problems
    sub_length_X = int( L_X / N_X )
    sub_length_Y = int( L_Y / N_Y )
    sub_length_Z = int( L_Z / N_Z )

    ## coordinates that define the cube
    coord_X = int( coords[0] )
    coord_Y = int( coords[1] )
    coord_Z = int( coords[2] )

    volume_p = int( sub_length_X * sub_length_Y * sub_length_Z )
    sub_array = [ sub_length_X, sub_length_Y, sub_length_Z ]

    ## minilengths of the minilattices
    mini_X = int( sub_length_X / mini_N[0] )
    mini_Y = int( sub_length_Y / mini_N[1] )
    mini_Z = int( sub_length_Z / mini_N[2] )

    mini_coord_X = int( mini_coords[0] )
    mini_coord_Y = int( mini_coords[1] )
    mini_coord_Z = int( mini_coords[2] )

    volume_mini = int( mini_X * mini_Y * mini_Z )

    # Fill the matrix with the lattice points
    data = np.empty( [volume_mini, 3] )
    alfa = 0    # Auxiliar variable

    # Define the cube used for each rank
    lenXRankIn = sub_length_X * coord_X + mini_coord_X * mini_X
    lenXRankFn = sub_length_X * coord_X + mini_X * ( 1 + mini_coord_X )
    lenYRankIn = sub_length_Y * coord_Y + mini_coord_Y * mini_Y
    lenYRankFn = sub_length_Y * coord_Y + mini_Y * ( 1 + mini_coord_Y )
    lenZRankIn = sub_length_Z * coord_Z + mini_coord_Z * mini_Z
    lenZRankFn = sub_length_Z * coord_Z + mini_Z * ( 1 + mini_coord_Z )

    for i in range( lenXRankIn, lenXRankFn ):
        value_X = i - L_X / 2 + 1
        for j in range( lenYRankIn, lenYRankFn ):
            value_Y = j - L_Y / 2 + 1
            for k in range( lenZRankIn, lenZRankFn ):
                value_Z = k - L_Z / 2 + 1
                data[alfa,0] = ( 2 * np.pi * value_X ) / L_X
                data[alfa,1] = ( 2 * np.pi * value_Y ) / L_Y
                data[alfa,2] = ( 2 * np.pi * value_Z ) / L_Z
                alfa += 1

    return data

def create_own_fin_lattice( L, N, mini_N, coords, mini_coords ):
    '''
        Create the lattice for a finite time direction calculation.
        In this case, we do have a 5-dimensional lattice to integrate.

        Must be consistent that product of N[i] = number of cores used
    '''

    # Get the size of the lattice
    L_X, L_Y, L_Z = int( L[0] ), int( L[1] ), int( L[2] ),
    T_1, T_2 = int( L[3] ), int( L[4] )

    # Get the number of divisions of the lattice
    N_X, N_Y, N_Z = int( N[0] ), int( N[1] ), int( N[2] ),
    N_1, N_2 = int( N[3] ), int( N[4] )


    ## sublengths of the sublattices
    sub_length_X = int( L_X / N_X )
    sub_length_Y = int( L_Y / N_Y )
    sub_length_Z = int( L_Z / N_Z )

    # subtime extents of the sublattices
    sub_time_T1 = int( T_1 / N_1 )
    sub_time_T2 = int( T_2 / N_2 )

    ## coordinates that define the cube
    coord_X = int( coords[0] )
    coord_Y = int( coords[1] )
    coord_Z = int( coords[2] )
    coord_T1 = int( coords[3] )
    coord_T2 = int( coords[4] )

    volume_p = int( sub_length_X * sub_length_Y * sub_length_Z  )
    volume_p *= int( sub_time_T1 * sub_time_T2 )

    sub_array = [
                    sub_length_X,
                    sub_length_Y,
                    sub_length_Z,
                    sub_time_T1,
                    sub_time_T2
                ]

    ## minilengths of the minilattices
    mini_X = int( sub_length_X / mini_N[0] )
    mini_Y = int( sub_length_Y / mini_N[1] )
    mini_Z = int( sub_length_Z / mini_N[2] )

    ## minitime extents of the minilattices
    mini_T1 = int( sub_time_T1 / mini_N[3] )
    mini_T2 = int( sub_time_T2 / mini_N[4] )

    mini_coord_X = int( mini_coords[0] )
    mini_coord_Y = int( mini_coords[1] )
    mini_coord_Z = int( mini_coords[2] )
    mini_coord_T1 = int( mini_coords[3] )
    mini_coord_T2 = int( mini_coords[4] )

    volume_mini = int( mini_X * mini_Y * mini_Z * mini_T1 * mini_T2 )

    # Fill the matrix with the lattice points
    data = np.empty( [volume_mini, 5] )
    alfa = 0    # Auxiliar variable

    # Define the cube used for each rank
    lenXRankIn = sub_length_X * coord_X + mini_coord_X * mini_X
    lenXRankFn = sub_length_X * coord_X + mini_X * ( 1 + mini_coord_X )
    lenYRankIn = sub_length_Y * coord_Y + mini_coord_Y * mini_Y
    lenYRankFn = sub_length_Y * coord_Y + mini_Y * ( 1 + mini_coord_Y )
    lenZRankIn = sub_length_Z * coord_Z + mini_coord_Z * mini_Z
    lenZRankFn = sub_length_Z * coord_Z + mini_Z * ( 1 + mini_coord_Z )
    len1RankIn = sub_time_T1 * coord_T1 + mini_coord_T1 * mini_T1
    len1RankFn = sub_time_T1 * coord_T1 + mini_T1 * ( 1 + mini_coord_T1 )
    len2RankIn = sub_time_T2 * coord_T2 + mini_coord_T2 * mini_T2
    len2RankFn = sub_time_T2 * coord_T2 + mini_T2 * ( 1 + mini_coord_T2 )

    for i in range( lenXRankIn, lenXRankFn ):
        value_X = i - L_X / 2 + 1
        for j in range( lenYRankIn, lenYRankFn ):
            value_Y = j - L_Y / 2 + 1
            for k in range( lenZRankIn, lenZRankFn ):
                value_Z = k - L_Z / 2 + 1
                for t_1 in range( len1RankIn, len1RankFn ):
                    value_T1 = t_1 - T_1 / 2 + 1
                    for t_2 in range( len2RankIn, len2RankFn ):
                        value_T2 = t_2 - T_2 / 2 + 1
                        data[alfa,0] = ( 2 * np.pi * value_X ) / L_X
                        data[alfa,1] = ( 2 * np.pi * value_Y ) / L_Y
                        data[alfa,2] = ( 2 * np.pi * value_Z ) / L_Z
                        data[alfa,3] = ( (2 * value_T1 + 1) * np.pi ) / T_1
                        data[alfa,4] = ( (2 * value_T2 + 1) * np.pi ) / T_2
                        alfa += 1


    return data

if __name__ == '__main__':
    pass
