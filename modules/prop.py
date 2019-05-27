"""
    Author: Sergio Chaves Garcia-Mascaraque
    E-mail: sergiozteskate@gmail.com

    Module containing the definitions of the propagators in all the different
    boundary conditions, periodic, SF/open/Dirichlet and infinite time extent
"""

import numpy as np

# Load matrix module
from .matrix import *

__all__ = [ 'wilson_tm_prop_integrated',
            'wilson_tm_propagator',
            'SF_wilson_TM' ]

def wilson_tm_prop_integrated(
                               mass,
                               twisted_mass,
                               spatial_momenta,
                               time,
                               up_down,
                               r_param
                             ):
    '''
        @arg (float): Standard mass of the quark
        @arg (float): Twisted mass of the quark
        @arg (list) : Final momenta of the correlation function. Set it to zero, if
                      you wished to add momenta use theta BC in the correlation
                      function.
        @arg (list) : Initial and final time in which the propagator is calculated.
        @arg (int)  : SU(2) projection. +1 for UP quarks, -1 for DOWN quarks.
        @arg (float): Wilson r-parameter.

        return      : Wilson-Twisted-Mass quark propagator in the free theory in the
                      limit in which the temporal boundary condition tends to infinity.
                      The spatial boundary conditions are set to be periodic.
    '''

    sum_cos, sum_sin_sq, sum_sin = 0, 0, 0

    for i in range( 0, len(spatial_momenta) ):
        sum_cos += (1 - np.cos(spatial_momenta[i] ) )
        sum_sin += np.sin( spatial_momenta[i] ) * gamma_matrix( i + 1 )
        sum_sin_sq += np.sin( spatial_momenta[i] ) ** 2

    # Define the elements of the propagator
    Delta  = mass + r_param * ( 1 + sum_cos )

    omega = np.arccosh(
            r_param * \
            ( Delta ** 2 + 1 + sum_sin_sq + twisted_mass ** 2) / ( 2 * Delta ) \
                      )

    sinh_omega = np.sinh( omega )

    identity_part = ( Delta - r_param * np.cosh( omega ) ) * identity_matrix()
    twisted_part = twisted_mass * gamma_matrix( 5 )

    numerator = identity_part + gamma_matrix( 4 ) * sinh_omega - \
                1j * sum_sin - up_down * 1j * twisted_part

    denominator = 2 *  Delta * sinh_omega

    numerator *= np.exp( - omega * ( time[0] - time[1] ) )
    return np.matrix( numerator / denominator, dtype = np.complex128 )

def wilson_tm_propagator(
                          mass,
                          twisted_mass,
                          spatial_momenta,
                          temporal_momenta,
                          up_down,
                          r = 1
                        ):
    '''
        @arg (float): Standard mass of the quark
        @arg (float): Twisted mass of the quark
        @arg (list) : Final momenta of the correlation function. Set it to zero, if
                      you wished to add momenta use theta BC in the correlation
                      function.
        @arg (list) : Initial and final time in which the propagator is calculated.
        @arg (int)  : SU(2) projection. +1 for UP quarks, -1 for DOWN quarks.
        @arg (float): Wilson r-parameter.

        return      : Wilson-Twisted-Mass quark propagator in the free theory when
                      the boundary conditions in the time direction are antiperiodic.
                      The spatial boundary conditions are set to be periodic.
    '''

    sum_cos =  ( 1 - np.cos( temporal_momenta ) )
    sum_sin_sq = ( np.sin( temporal_momenta )) ** 2

    for i in range( 0, len(spatial_momenta) ):
        sum_cos += ( 1 - np.cos( spatial_momenta[i] ) )
        sum_sin_sq += ( np.sin( spatial_momenta[i] ) ) ** 2

    gamma_part = gamma_matrix( 4 ) * np.sin( temporal_momenta )
    twisted_part = gamma_matrix( 5 ) * twisted_mass

    for mu in range(0, len(spatial_momenta) ):
        gamma_part += gamma_matrix( mu + 1 ) * np.sin( spatial_momenta[ mu ] )

    denominator = ( mass + r * sum_cos ) ** 2  + \
                  sum_sin_sq + twisted_mass ** 2
    num_matrix = ( mass + r * sum_cos ) * identity_matrix() - \
                 1j *  gamma_part - up_down * 1j * twisted_part

    return num_matrix / denominator

def G_func( mass, twisted_mass, spatial_momenta, time, T ):
    '''
        @arg (float): Standard mass of the quark
        @arg (float): Twisted mass of the quark
        @arg (list) : Final momenta of the correlation function. Set it to zero, if
                      you wished to add momenta use theta BC in the correlation
                      function.
        @arg (list) : Initial and final time in which the propagator is calculated.
        @arg (int)  : Temporal extent of the lattice.

        return      : G_function used to calculate the quark propagator in the
                      free theory when the boundary conditions in the time direction
                      are open/Dirichlet/SF. The spatial boundary conditions are
                      set to be periodic.
    '''

    sum_sin_sq, sum_sin_half = 0, 0

    for i in range( 0, len(spatial_momenta) ):
        sum_sin_sq += np.sin( spatial_momenta[i] ) ** 2
        sum_sin_half += ( 2 * np.sin( spatial_momenta[i] / 2 ) ) ** 2

    omega = 2 * np.arcsinh(
            0.5 * \
            np.sqrt(
                ( sum_sin_sq + twisted_mass ** 2 + \
                ( mass + 0.5 * sum_sin_half ) ** 2 ) / \
                ( 1 + mass + 0.5 * sum_sin_half )
                   )
            )

    while omega > 2 * np.pi:
        omega = omega - 2 * np.pi

    p_0 = 1j * omega
    sin_temp, sin_temp_half = np.sin( p_0 ), 2 * np.sin( p_0 / 2 )

    M = mass + 0.5 * ( sum_sin_half + sin_temp_half ** 2 )
    R = M * ( 1 - np.exp( - 2 * omega * T )) - \
            1j * sin_temp * ( 1 + np.exp( -2 * omega * T ) )
    A = 1 + ( mass + 0.5 * sum_sin_half )

    ## Now we construct the G function
    N = 1 / ( -2j * sin_temp * A * R )
    M_plus = M + 1j * sin_temp
    M_min = M - 1j * sin_temp

    G_func_P = N * ( \
            M_min * ( np.exp( -omega * (abs( time[0] - time[1] ) ) ) - \
            np.exp( omega * ( time[0] + time[1] - 2 * T ) ) ) + \
            M_plus * ( np.exp( omega * ( abs( time[0] - time[1] ) - 2 * T ) ) - \
            np.exp( -omega * ( time[0] + time[1] ) ) )
            )

    G_func_M = N * ( \
            M_min * ( np.exp( -omega * (abs( time[1] - time[0] ) ) ) - \
            np.exp( - omega * ( time[1] + time[0] ) ) ) + \
            M_plus * ( np.exp( omega * ( abs( time[1] - time[0] ) - 2 * T ) ) - \
            np.exp( -omega * ( 2 * T - time[1] - time[0]) ) )
            )

    P_plus = 0.5 * ( identity_matrix() - gamma_matrix( 4 ) )
    P_minus = 0.5 * ( identity_matrix() + gamma_matrix( 4 ) )

    return  ( G_func_M * P_minus + G_func_P * P_plus )

def SF_wilson_TM( mass, twisted_mass, spatial_momenta , time, T, up_down ):
    '''
        @arg (float): Standard mass of the quark
        @arg (float): Twisted mass of the quark
        @arg (list) : Final momenta of the correlation function. Set it to zero, if
                      you wished to add momenta use theta BC in the correlation
                      function.
        @arg (list) : Initial and final time in which the propagator is calculated.
        @arg (int)  : Temporal extent of the lattice.
        @arg (int)  : SU(2) projection. +1 for UP quarks, -1 for DOWN quarks.

        return      : Wilson-Twisted-Mass quark propagator in the free theory when
                      the boundary conditions in the time direction are open/Dirichlet/
                      SF. The spatial boundary conditions are set to be periodic.
    '''

    ## To match openQCD data T -> N_T - a
    T = T - 1

    ## Now we calculate the finite part corresponding to ( D + M ) * G, M_P = Delta
    part_sin, part_sin_sq = 0, 0

    for i in range( 0, len( spatial_momenta ) ):
        part_sin += gamma_matrix( i + 1 ) * np.sin( spatial_momenta[i] )
        part_sin_sq += ( 2 * np.sin( spatial_momenta[i] / 2 ) ) ** 2

    ## Propagator is defined as:
    ## G_+ = G(p,y_4+1,x_4) , G_- = G(p,y_4-1,x_4), G_c = G(p,y_4,x_4)

    ## 0.5 * g4 * [G_+ - G_-] - 0.5 * ( G_+ + G_- - 2G_c ) * I
    ## - i gamma_i sin(p_i) *G_c
    ## + I * 2 * sin^2(p_i/2) * G_c + m * G_c - i * g5 * mu * G_c

    #if time[0] + 1 == T + 1 or time[0] - 1 == 0:
    #    G_p, G_m = 0, 0

    G_p = G_func( mass, twisted_mass, spatial_momenta, [time[0] + 1, time[1]], T )
    G_m = G_func( mass, twisted_mass, spatial_momenta, [time[0] - 1, time[1]], T )
    G_c = G_func( mass, twisted_mass, spatial_momenta, time, T )

    ## Try to place a +i as gamma_\mu^\dagger = gamma_\mu
    prop =  0.5 * gamma_matrix( 4 ) * ( G_p - G_m ) - 0.5 * ( G_p + G_m - 2 * G_c ) \
    + ( -1j * part_sin + part_sin_sq / 2 * identity_matrix() ) * G_c + mass * G_c \
    - 1j * gamma_matrix( 5 ) * twisted_mass * G_c * up_down

    return prop


if __name__ == '__main__':
    pass
