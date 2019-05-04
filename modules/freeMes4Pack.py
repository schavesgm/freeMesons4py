"""
    Author: Sergio Chaves Garcia-Mascaraque
    E-mail: sergiozteskate@gmail.com

    Module containing the integration of the correlation functions using
    different boundary conditions
"""

import numpy as np
import timeit

from .prop import *
from .matrix import *
from .lat import *

__all__ =  [ 'create_trace_and_integrate', 'operator_dictionary' ]

def create_trace_and_integrate(
                                meson_type, bound_elec,
                                attributes, N,
                                coords, operators,
                                t_source, L,
                                T, rank,
                                mini_N, theta_1,
                                theta_2, type_1,
                                type_2, time_calc,
                                r_1, r_2
                              ):

    """
        The size of N and the size of mini_N must be consistent with the
        boundary conditions elected.
    """

    # Load the attributes
    mass_1, mass_2,  = attributes[0], attributes[1]
    twisted_mass_1, twisted_mass_2 = attributes[2], attributes[3]
    final_momenta = attributes[4:]

    # Antiperiodic boundary conditions
    if bound_elec == 0:

        operator_1, operator_2 = operators[0], operators[1]
        bar_operator_2 = gamma_matrix( 4 ) * operator_2 * gamma_matrix( 4 )

        L_array = [L,L,L,T,T]

        # Generate the mini-lattices, using mini_N
        N_mini_div = np.prod( mini_N )

        mini_coords = []
        for i in range( 0, mini_N[0] ):
            for j in range( 0, mini_N[1] ):
                for k in range( 0, mini_N[2] ):
                    for t_1 in range( 0, mini_N[3] ):
                        for t_2 in range( 0, mini_N[4] ):
                            mini_coords.append( [i,j,k,t_1,t_2] )

        time_array = np.arange( time_calc[0], time_calc[1] )

        time_results_real = np.empty( [len(time_array), len(mini_coords)],
                                      dtype = 'float' )
        time_results_imag = np.zeros( [len(time_array), len(mini_coords)],
                                      dtype = 'complex' )

        start = timeit.default_timer()

        # For each mini-lattice calculate all times
        for i in range( 0, len( mini_coords ) ):

            # Create all the minicontributions for each time
            lattice_points = create_own_fin_lattice(
                    L_array, N, mini_N, coords, mini_coords[i] )

            #print( 'rank number ', rank, ' in his ', i + 1, ' subdivision has ',
            #        len(lattice_points) * ( time_calc[1] - time_calc[0] ),
            #        ' points.' )

            trace_array = []

            # Calculate different correlators depending on meson_type

            # Two flavor meson -- Charged pions
            if meson_type == "2FLAVORS":

                for l in range( 0, len( lattice_points ) ):

                    prop_A = wilson_tm_propagator( mass_1, twisted_mass_1,
                            lattice_points[l,0:3] + theta_1,
                            lattice_points[l,3], type_1, r_1 )

                    prop_B = wilson_tm_propagator( mass_2, twisted_mass_2,
                            lattice_points[l,0:3] - final_momenta + theta_2,
                            lattice_points[l,4], type_2,  r_2)

                    trace_array.append( - np.trace( operator_1 * prop_A * \
                                                   bar_operator_2 * prop_B  ) )

                for t in range( 0, len( time_array ) ):

                    suma = 0
                    for k in range( 0, len( trace_array ) ):
                        exponential = np.exp( 1j * \
                                ( lattice_points[k,3] - lattice_points[k,4] ) * \
                                ( time_array[t] - t_source ) )

                        suma +=  trace_array[k] * exponential

                    time_results_real[t,i] = suma.real
                    time_results_imag[t,i] = suma.imag
                    suma = 0

            # Blob disconnected diagram -- Propagator (x,x)
            elif meson_type == "BLOB":

                for l in range( 0, len( lattice_points ) ):

                    prop_A = wilson_tm_propagator( mass_1, twisted_mass_1,
                            lattice_points[l,0:3] + theta_1,
                            lattice_points[l,3], type_1, r_1 )

                    trace_array.append( -np.trace( operator_1 * prop_A  ) )

                for t in range( 0, len( time_array ) ):

                    suma = 0
                    for k in range( 0, len( trace_array ) ):

                        suma +=  trace_array[k]
                        # Exponential in a blob is always 1 as t - t = 0

                    time_results_real[t,i] = suma.real
                    time_results_imag[t,i] = suma.imag
                    suma = 0

    if bound_elec == 1 or bound_elec == 2:

        operator_1, operator_2 = operators[0], operators[1]
        bar_operator_2 = gamma_matrix( 4 ) * operator_2 * gamma_matrix( 4 )

        L_array = [L,L,L]

        # Generate the mini-lattices, using mini_N

        sub_array = np.empty( 3 )
        for i in range( 0, 3 ):
            sub_array[i] = int( L_array[i] / N[i] )

        mini_coords = []
        for i in range( 0, mini_N[0] ):
            for j in range( 0, mini_N[1] ):
                for k in range( 0, mini_N[2] ):
                    mini_coords.append( [i,j,k] )

        time_array = np.arange( time_calc[0], time_calc[1] )

        time_results_real = np.empty( [len(time_array), len(mini_coords) ], \
                                      dtype = complex )
        time_results_imag = np.empty( [len(time_array), len(mini_coords) ], \
                                      dtype = complex )

        start = timeit.default_timer()

        for i in range( 0, len( mini_coords ) ):

            # Create all the minicontributions for each time
            lattice_points = create_own_tinf_lattice( L_array, N, mini_N, coords, mini_coords[i] )

            print(  'rank number ', rank, ' in his ', i + 1, ' subdivision has ',
                    len(lattice_points) * ( time_calc[1] - time_calc[0] ),
                    ' points.' )

            # T infinite WTM
            if bound_elec == 1:

                if meson_type == "2FLAVORS":

                    for t in range( 0, len(time_array) ):
                        suma = 0
                        for l in range( 0, len( lattice_points ) ):

                            prop_A = wilson_tm_prop_integrated( mass_1,
                                    twisted_mass_1,
                                    lattice_points[l,:] + theta_1,
                                    [time_array[t], t_source],
                                    type_1, r_1 )

                            prop_B = np.conj( wilson_tm_prop_integrated(
                                mass_2, twisted_mass_2,
                                lattice_points[l,:] - final_momenta + theta_2,
                                [time_array[t], t_source], -type_2, r_2 ) )

                            suma +=  np.trace( -operator_1 * prop_A * \
                                    bar_operator_2 * gamma_matrix( 5 ) * \
                                    prop_B * gamma_matrix( 5 ) )

                        time_results_real[t,i] = suma.real
                        time_results_imag[t,i] = suma.imag
                        suma = 0

                if meson_type == "BLOB":

                    for t in range( 0, len(time_array) ):
                        suma = 0
                        for l in range( 0, len( lattice_points ) ):

                            prop_A = wilson_tm_prop_integrated( mass_1,
                                    twisted_mass_1,
                                    lattice_points[l,:] + theta_1,
                                    [time_array[t], time_array[t]], type_1, r_1 )

                            suma +=  np.trace( -operator_1 * prop_A  )

                        time_results_real[t,i] = suma.real
                        time_results_imag[t,i] = suma.imag
                        suma = 0

            # Dirichlet - SF - open Boundary conditions in time extent
            if bound_elec == 2:

                if meson_type == "2FLAVORS":

                    for t in range( 0, len(time_array)):

                        # Fix corr functions at 0 and T to be zero
                        if time_array[t] == 0  or time_array[t] == T - 1:
                            time_results_real[t,i] = 0
                            time_results_imag[t,i] = 0

                        else:
                            suma = 0
                            for l in range( 0, len( lattice_points ) ):

                                prop_A = SF_wilson_TM( mass_1, twisted_mass_1,
                                        lattice_points[l,:] + theta_1,
                                        [time_array[t], t_source], T, type_1 )

                                prop_B = SF_wilson_TM( mass_2, twisted_mass_2,
                                        lattice_points[l,:] - final_momenta + theta_2,
                                        [t_source, time_array[t]] , T, type_2 )

                                suma +=  np.trace( - operator_1 * prop_A * \
                                        bar_operator_2 * prop_B  )

                            time_results_real[t,i] = suma.real
                            time_results_imag[t,i] = suma.imag

                            suma = 0

                if meson_type == "BLOB":

                    for t in range( 0, len(time_array)):

                        # Fix corr functions at 0 and T to be zero
                        if time_array[t] == 0  or time_array[t] == T - 1:
                            time_results_real[t,i] = 0
                            time_results_imag[t,i] = 0

                        else:
                            suma = 0
                            for l in range( 0, len( lattice_points ) ):

                                prop_A = SF_wilson_TM( mass_1, twisted_mass_1,
                                    lattice_points[l,:] + theta_1,
                                    [time_array[t], time_array[t]], T, type_1 )

                                suma += - np.trace( operator_1 * prop_A )

                            time_results_real[t,i] = suma.real
                            time_results_imag[t,i] = suma.imag

                            suma = 0

    # print( 'The rank number ', rank , ' has created and integrated',
    #        ' his lattice in ', timeit.default_timer() - start, ' seconds.' )

    # Sum all the contributions to the integral
    timeCorrReal, timeCorrImag = [], []
    for t in range( 0, len( time_array) ):
        timeCorrReal.append( np.sum( time_results_real[t,:].real ) )
        timeCorrImag.append( np.sum( time_results_imag[t,:].imag ) )

    return timeCorrReal, timeCorrImag

if __name__ == '__main__':
    pass


