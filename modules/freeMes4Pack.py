'''
    -> Package to build free mesonic observables using
       antiperiodic, Dirichlet and T -> infty in time direction.


    -> Sergio Chaves Garcia-Mascaraque. sergiozteskate@gmail.com

    -> Marzo de 2018 - Septiembre de 2018
'''

import numpy as np
import timeit

__all__ =  [ 'create_trace_and_integrate', 'operator_dictionary' ]

## FUNCTIONS THAT ARE IMPORTED IN THE MODULE

def create_trace_and_integrate( meson_type, bound_elec, attributes, N, coords,
        operators, t_source, L, T, rank, mini_N, theta_1, theta_2, type_1,
        type_2, time_calc, r_1, r_2 ):

    '''
        THE SIZE OF N AND THE SIZE OF MINI_N MUST BE CONSISTENT WITH THE BOUNDARY ELECTION.
        --> 5 ELEMENTS FOR THE ANTIPERIODIC CASE AND 3 FOR THE PERIODIC
    '''
    mass_1, mass_2,  = attributes[0], attributes[1]
    twisted_mass_1, twisted_mass_2 = attributes[2], attributes[3]
    final_momenta = attributes[4:]

    ### ANTIPERIODIC BOUNDARY CONDITIONS
    if bound_elec == 0:

        operator_1, operator_2 = operators[0], operators[1]
        bar_operator_2 = gamma_matrix( 4 ) * operator_2 * gamma_matrix( 4 )

        L_array = [L,L,L,T,T]

        ### Generate the mini-lattices, using mini_N
        N_mini_div = np.prod( mini_N )

        mini_coords = []
        for i in range( 0, mini_N[0] ):
            for j in range( 0, mini_N[1] ):
                for k in range( 0, mini_N[2] ):
                    for t_1 in range( 0, mini_N[3] ):
                        for t_2 in range( 0, mini_N[4] ):
                            mini_coords.append( [i,j,k,t_1,t_2] )

        time_array = np.arange( time_calc[0], time_calc[1] )

        time_results_real = np.empty( [len(time_array), len(mini_coords)], dtype = 'float' )
        time_results_imag = np.zeros( [len(time_array), len(mini_coords)], dtype = 'complex' )

        start = timeit.default_timer()

        ### For each mini-lattice calculate all times
        for i in range( 0, len( mini_coords ) ):

            ### Create all the minicontributions for each time
            lattice_points = create_own_fin_lattice( L_array, N, mini_N, coords, mini_coords[i] )

            print( 'rank number ', rank, ' in his ', i + 1, ' subdivision has ',
                    len(lattice_points) * ( time_calc[1] - time_calc[0] ),
                    ' points.' )

            trace_array = []

            ## Calculate different correlators depending on meson_type

            ## Two flavor meson -- Charged pions
            if meson_type == "2FLAVORS":

                for l in range( 0, len( lattice_points ) ):
                    prop_A = wilson_tm_propagator( mass_1, twisted_mass_1,
                            lattice_points[l,0:3] + theta_1,
                            lattice_points[l,3], type_1, r_1 )

                    prop_B = wilson_tm_propagator( mass_2, twisted_mass_2,
                            lattice_points[l,0:3] - final_momenta + theta_2,
                            lattice_points[l,4], type_2,  r_2)

                    trace_array.append( -np.trace( operator_1 * prop_A *
                                                   bar_operator_2 * prop_B  ) )

                for t in range( 0, len( time_array ) ):
                    suma = 0
                    for k in range( 0, len( trace_array ) ):
                        exponential = np.exp( 1j *
                                ( lattice_points[k,3] - lattice_points[k,4] ) *
                                ( time_array[t] - t_source ) )

                        suma +=  trace_array[k] * exponential
                    time_results_real[t,i] = suma.real
                    time_results_imag[t,i] = suma.imag
                    suma = 0

            ## Blob disconnected diagram -- Propagator (x,x)
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
                        ## Exponential in a blob is always 1 as t - t = 0
                        #exponential = np.exp( 1j *
                        #        ( lattice_points[k,3] - lattice_points[k,4] ) *
                        #        ( time_array[t] - time_array[t] ) )

                    time_results_real[t,i] = suma.real
                    time_results_imag[t,i] = suma.imag
                    suma = 0

    if bound_elec == 1 or bound_elec == 2:

        operator_1, operator_2 = operators[0], operators[1]
        bar_operator_2 = gamma_matrix( 4 ) * operator_2 * gamma_matrix( 4 )

        L_array = [L,L,L]

        ### Generate the mini-lattices, using mini_N
        #N_mini_div = np.prod( mini_N )

        sub_array = np.empty( 3 )
        for i in range( 0, 3 ):
            sub_array[i] = int( L_array[i] / N[i] )

        mini_coords = []
        for i in range( 0, mini_N[0] ):
            for j in range( 0, mini_N[1] ):
                for k in range( 0, mini_N[2] ):
                    mini_coords.append( [i,j,k] )

        time_array = np.arange( time_calc[0], time_calc[1] )

        time_results_real = np.empty( [len(time_array), len(mini_coords) ], dtype = complex )
        time_results_imag = np.empty( [len(time_array), len(mini_coords) ], dtype = complex )

        start = timeit.default_timer()

        for i in range( 0, len( mini_coords ) ):

            ### Create all the minicontributions for each time
            lattice_points = create_own_tinf_lattice( L_array, N, mini_N, coords, mini_coords[i] )

            print('rank number ', rank, ' in his ', i + 1, ' subdivision has ',
                    len(lattice_points) * ( time_calc[1] - time_calc[0] ),
                    ' points.' )

            ### T infinite WTM
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

                            suma +=  np.trace( -operator_1 * prop_A *
                                    bar_operator_2 * gamma_matrix( 5 ) *
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

            ### Dirichlet - SF - open Boundary conditions in time extent
            if bound_elec == 2:

                if meson_type == "2FLAVORS":

                    for t in range( 0, len(time_array)):

                        ### FIX CORRELATION FUNCTIONS AT 0 AND T TO BE ZERO
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

                                suma +=  np.trace( - operator_1 * prop_A *
                                        bar_operator_2 * prop_B  )

                            time_results_real[t,i] = suma.real
                            time_results_imag[t,i] = suma.imag

                            suma = 0

                if meson_type == "BLOB":

                    for t in range( 0, len(time_array)):

                        ### FIX CORRELATION FUNCTIONS AT 0 AND T TO BE ZERO
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

    print( 'The rank number ', rank , ' has created and integrated',
           ' his lattice in ', timeit.default_timer() - start, ' seconds.' )

    timeCorrReal, timeCorrImag = [], []
    for t in range( 0, len( time_array) ):
        timeCorrReal.append( np.abs( np.sum( time_results_real[t,:] ) ) )
        timeCorrImag.append( np.abs( np.sum( time_results_imag[t,:] ) ) )

    return timeCorrReal, timeCorrImag

''' ############### Create lattices functions ################# '''

''' 10 de Mayo de 2018
    ---> Sergio Chaves
    ---> Javier Ugarrio

    ##############################################################################
    Function that creates a lattice in 5 dimensions using periodic or antiperiodic
    boundary conditions. It uses a parallelepiped of LX*LY*LZ*T_1*T_2 divided each
    side in N_X*N_Y*N_Z*N_T1*N_T2
    It asigns the partitions of each dimension according to the number of blocks you
    want to create, which could be used to divide the lattice in N_c cores in parallel.

    The lattice length MUST be a number so that L_i, T_i = 2^n with n integer and the number
    of cores must be also a number of cores so that N_c = 2^n <  L_X * L_Y * ... * T_2
    ##############################################################################
    '''

def create_own_tinf_lattice( L, N, mini_N, coords, mini_coords ): ## FOR INFINTE TIME EXTENT LATTICE
    ''' Must be consistent that product of N[i] = number of cores used '''

    L_X, L_Y, L_Z = int( L[0] ), int( L[1] ), int( L[2] )
    N_X, N_Y, N_Z = int( N[0] ), int( N[1] ), int( N[2] )

    ## sublengths of the sublattices
    sub_length_X = int( L_X / N_X )
    sub_length_Y = int( L_Y / N_Y )
    sub_length_Z = int( L_Z / N_Z )

    ## coordinates that define the cube
    coord_X , coord_Y, coord_Z = int(coords[0]), int(coords[1]), int(coords[2])

    volume_p = int( sub_length_X * sub_length_Y * sub_length_Z )
    sub_array = [ sub_length_X, sub_length_Y, sub_length_Z ]

    ## minilengths of the minilattices
    mini_X = int( sub_length_X / mini_N[0] )
    mini_Y = int( sub_length_Y / mini_N[1] )
    mini_Z = int( sub_length_Z / mini_N[2] )

    mini_coord_X, mini_coord_Y, mini_coord_Z = int(mini_coords[0]), int(mini_coords[1]), int(mini_coords[2])

    volume_mini = int( mini_X * mini_Y * mini_Z )

    data = np.empty( [volume_mini, 3] )
    alfa = 0

    for i in range( sub_length_X * coord_X + mini_coord_X * mini_X, sub_length_X * coord_X + mini_X * ( 1 + mini_coord_X ) ):
        value_X = i - L_X / 2 + 1
        for j in range( sub_length_Y * coord_Y + mini_coord_Y * mini_Y, sub_length_Y * coord_Y + mini_Y * ( 1 + mini_coord_Y ) ):
            value_Y = j - L_Y / 2 + 1
            for k in range( sub_length_Z * coord_Z + mini_coord_Z * mini_Z, sub_length_Z * coord_Z + mini_Z * ( 1 + mini_coord_Z ) ):
                value_Z = k - L_Z / 2 + 1
                data[alfa,0] = ( 2 * np.pi * value_X ) / L_X
                data[alfa,1] = ( 2 * np.pi * value_Y ) / L_Y
                data[alfa,2] = ( 2 * np.pi * value_Z ) / L_Z
                alfa += 1

    return data

def create_own_fin_lattice( L, N, mini_N, coords, mini_coords ):
    ''' Must be consistent that product of N[i] = number of cores used '''

    L_X, L_Y, L_Z, T_1, T_2 = int( L[0] ), int( L[1] ), int( L[2] ), int( L[3] ), int( L[4] )
    N_X, N_Y, N_Z, N_1, N_2 = int( N[0] ), int( N[1] ), int( N[2] ), int( N[3] ), int( N[4] )


    ## sublengths of the sublattices
    sub_length_X = int( L_X / N_X )
    sub_length_Y = int( L_Y / N_Y )
    sub_length_Z = int( L_Z / N_Z )

    # subtime extents of the sublattices
    sub_time_T1 = int( T_1 / N_1 )
    sub_time_T2 = int( T_2 / N_2 )

    ## coordinates that define the cube
    coord_X , coord_Y, coord_Z = int(coords[0]), int(coords[1]), int(coords[2])
    coord_T1, coord_T2 = int(coords[3]), int(coords[4])

    volume_p = int( sub_length_X * sub_length_Y * sub_length_Z * sub_time_T1 * sub_time_T2 )
    sub_array = [ sub_length_X, sub_length_Y, sub_length_Z, sub_time_T1, sub_time_T2 ]

    ## minilengths of the minilattices
    mini_X = int( sub_length_X / mini_N[0] )
    mini_Y = int( sub_length_Y / mini_N[1] )
    mini_Z = int( sub_length_Z / mini_N[2] )

    ## minitime extents of the minilattices
    mini_T1 = int( sub_time_T1 / mini_N[3] )
    mini_T2 = int( sub_time_T2 / mini_N[4] )

    mini_coord_X, mini_coord_Y, mini_coord_Z = int(mini_coords[0]), int(mini_coords[1]), int(mini_coords[2])
    mini_coord_T1, mini_coord_T2 = int(mini_coords[3]), int(mini_coords[4])

    volume_mini = int( mini_X * mini_Y * mini_Z * mini_T1 * mini_T2 )

    data = np.empty( [volume_mini, 5] )
    alfa = 0

    for i in range( sub_length_X * coord_X + mini_coord_X * mini_X, sub_length_X * coord_X + mini_X * ( 1 + mini_coord_X ) ):
        value_X = i - L_X / 2 + 1
        for j in range( sub_length_Y * coord_Y + mini_coord_Y * mini_Y, sub_length_Y * coord_Y + mini_Y * ( 1 + mini_coord_Y ) ):
            value_Y = j - L_Y / 2 + 1
            for k in range( sub_length_Z * coord_Z + mini_coord_Z * mini_Z, sub_length_Z * coord_Z + mini_Z * ( 1 + mini_coord_Z ) ):
                value_Z = k - L_Z / 2 + 1
                for t_1 in range( sub_time_T1 * coord_T1 + mini_coord_T1 * mini_T1, sub_time_T1 * coord_T1 + mini_T1 * ( 1 + mini_coord_T1 ) ):
                    value_T1 = t_1 - T_1 / 2 + 1
                    for t_2 in range( sub_time_T2 * coord_T2 + mini_coord_T2 * mini_T2, sub_time_T2 * coord_T2 + mini_T2 * ( 1 + mini_coord_T2 ) ):
                        value_T2 = t_2 - T_2 / 2 + 1
                        data[alfa,0] = ( 2 * np.pi * value_X ) / L_X
                        data[alfa,1] = ( 2 * np.pi * value_Y ) / L_Y
                        data[alfa,2] = ( 2 * np.pi * value_Z ) / L_Z
                        data[alfa,3] = ( (2 * value_T1 + 1) * np.pi ) / T_1
                        data[alfa,4] = ( (2 * value_T2 + 1) * np.pi ) / T_2
                        alfa += 1


    return data

''' OPERATOR DICTIONARY '''

def operator_dictionary( operator_signal, number ):
    ''' Dictionary of operators:'''

    ## Needed to be consistent with input_assigner output
    operator_signal = operator_signal[number - 1 ]

    if operator_signal == 'I':
        return identity_matrix()
    if operator_signal == 'G1':
        return gamma_matrix( 1 )
    if operator_signal == 'G2':
        return gamma_matrix( 2 )
    if operator_signal == 'G3':
        return gamma_matrix( 3 )
    if operator_signal == 'G4':
        return gamma_matrix( 4 )
    if operator_signal == 'G5':
        return gamma_matrix( 5 )
    if operator_signal == 'G4G1':
        return gamma_matrix( 4 ) * gamma_matrix( 1 )
    if operator_signal == 'G4G2':
        return gamma_matrix( 4 ) * gamma_matrix( 2 )
    if operator_signal == 'G4G3':
        return gamma_matrix( 4 ) * gamma_matrix( 3 )
    if operator_signal == 'G4G5':
        return gamma_matrix( 4 ) * gamma_matrix( 5 )
    if operator_signal == 'G1G2':
        return gamma_matrix( 1 ) * gamma_matrix( 2 )
    if operator_signal == 'G1G3':
        return gamma_matrix( 1 ) * gamma_matrix( 3 )
    if operator_signal == 'G1G5':
        return gamma_matrix( 1 ) * gamma_matrix( 5 )
    if operator_signal == 'G2G3':
        return gamma_matrix( 2 ) * gamma_matrix( 3 )
    if operator_signal == 'G2G5':
        return gamma_matrix( 2 ) * gamma_matrix( 5 )
    if operator_signal == 'G3G5':
        return gamma_matrix( 3 ) * gamma_matrix( 5 )

## PRIVATE FUNCTIONS USED BY THE MODULE IMPORTED

def gamma_matrix( mu ):
    ''' Returns the gamma euclidean matrix given the mu index '''
    zeros, imagp, imagn = 0.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j
    onesp, onesn = 1.0 + 0.0j, -1.0 + 0.0j
    if ( mu == 0 or mu > 5 ):
        print( 'ERROR. The index runs from 1 to 5' )
    if mu == 1:
        return np.matrix( [[zeros, zeros, zeros, imagn], [zeros, zeros, imagn, zeros], [zeros, imagp, zeros, zeros], [imagp, zeros, zeros, zeros]], dtype = complex )
    if mu == 2:
        return np.matrix( [[zeros, zeros, zeros, onesn], [zeros, zeros, onesp, zeros], [zeros, onesp, zeros, zeros], [onesn, zeros, zeros, zeros]], dtype = complex )
    if mu == 3:
        return np.matrix( [[zeros, zeros, imagn, zeros], [zeros, zeros, zeros, imagp], [imagp, zeros, zeros, zeros], [zeros, imagn, zeros, zeros]], dtype = complex )
    if mu == 4:
        return np.matrix( [[zeros, zeros, onesp, zeros], [zeros, zeros, zeros, onesp], [onesp, zeros, zeros, zeros], [zeros, onesp, zeros, zeros]], dtype = complex )
    if mu == 5:
        return np.matrix( [[onesp, zeros, zeros, zeros], [zeros, onesp, zeros, zeros], [zeros, zeros, onesn, zeros], [zeros, zeros, zeros, onesn]], dtype = complex )

def identity_matrix( dim = 4 ):
    ''' Returns the identity delta kronnecker given the dimension ( default 4 ) '''
    delta = np.zeros( [dim,dim] )
    for i in range( 0, dim ):
        delta[i,i] = 1.0

    return np.matrix( delta, dtype = complex )


''' ####################### Propagators used in the calculations ######################## '''

def wilson_tm_prop_integrated( mass, twisted_mass, spatial_momenta, time, up_down, r_param ):
    ''' Returns the quark propagator using twisted mass quarks integrated over time
        up_down must be 1 if the propagator corresponds to the UP quark and -1 if it
        corresponds to the DOWN proyection in flavor space. '''

    sum_cos, sum_sin_sq, sum_sin = 0, 0, 0

    for i in range( 0, len(spatial_momenta) ):
        sum_cos += (1 - np.cos(spatial_momenta[i] ) )
        sum_sin += np.sin( spatial_momenta[i] ) * gamma_matrix( i + 1 )
        sum_sin_sq += np.sin( spatial_momenta[i] ) ** 2

    Delta  = mass + r_param * ( 1 + sum_cos )
    omega = np.arccosh( r_param * ( Delta ** 2 + 1 + sum_sin_sq + twisted_mass ** 2) / ( 2 * Delta ) ) #
    sinh_omega = np.sinh( omega )

    identity_part = ( Delta - r_param * np.cosh( omega ) ) * identity_matrix()
    twisted_part = twisted_mass * gamma_matrix( 5 )

    numerator = identity_part + gamma_matrix( 4 ) * sinh_omega - 1j * sum_sin - up_down * 1j * twisted_part

    denominator = 2 *  Delta * sinh_omega

    return np.matrix( ( numerator * np.exp( - omega * ( time[0] - time[1] ) ) ) / denominator, dtype = np.complex128 )

def G_func( mass, twisted_mass, spatial_momenta, time, T ):

    sum_sin_sq, sum_sin_half = 0, 0
    for i in range( 0, len(spatial_momenta) ):
        sum_sin_sq += np.sin( spatial_momenta[i] ) ** 2
        sum_sin_half += ( 2 * np.sin( spatial_momenta[i] / 2 ) ) ** 2

    omega = 2 * np.arcsinh( 0.5 * np.sqrt( (sum_sin_sq + twisted_mass ** 2 + ( mass + 0.5 * sum_sin_half ) ** 2 ) / ( 1 + mass + 0.5 * sum_sin_half ) ) )

    while omega > 2 * np.pi:
        omega = omega - 2 * np.pi

    p_0 = 1j * omega
    sin_temp, sin_temp_half = np.sin( p_0 ), 2 * np.sin( p_0 / 2 )

    M = mass + 0.5 * ( sum_sin_half + sin_temp_half ** 2 )
    R = M * ( 1 - np.exp( - 2 * omega * T )) - 1j * sin_temp * ( 1 + np.exp( -2 * omega * T ) )
    A = 1 + ( mass + 0.5 * sum_sin_half )

    ## Now we construct the G function
    N = 1 / ( -2j * sin_temp * A * R )
    M_plus = M + 1j * sin_temp
    M_min = M - 1j * sin_temp

    G_func_P = N * ( M_min * ( np.exp( -omega * (abs( time[0] - time[1] ) ) ) - np.exp( omega * ( time[0] + time[1] - 2 * T ) ) ) + M_plus * ( np.exp( omega * ( abs( time[0] - time[1] ) - 2 * T ) ) - np.exp( -omega * ( time[0] + time[1] ) ) ) )

    G_func_M = N * ( M_min * ( np.exp( -omega * (abs( time[1] - time[0] ) ) ) - np.exp( - omega * ( time[1] + time[0] ) ) ) + M_plus * ( np.exp( omega * ( abs( time[1] - time[0] ) - 2 * T ) ) - np.exp( -omega * ( 2 * T - time[1] - time[0]) ) ) )

    P_plus = 0.5 * ( identity_matrix() - gamma_matrix( 4 ) )
    P_minus = 0.5 * ( identity_matrix() + gamma_matrix( 4 ) )

    return  ( G_func_M * P_minus + G_func_P * P_plus )

def SF_wilson_TM( mass, twisted_mass, spatial_momenta , time, T, up_down ):
    """ Return the quark propagator using SF twisted mass regularization in finite time
        extennt. up_down = 1 represents an up_quark and up_down = -1 represents a down
        quark. """

    ## To match openQCD data T -> N_T - a
    T = T - 1

    ## Now we calculate the finite part corresponding to ( D + M ) * G, M_P = Delta
    part_sin, part_sin_sq = 0, 0

    for i in range( 0, len( spatial_momenta ) ):
        part_sin += gamma_matrix( i + 1 ) * np.sin( spatial_momenta[i] )
        part_sin_sq += ( 2 * np.sin( spatial_momenta[i] / 2 ) ) ** 2

    ## Propagator is defined as:
    ## G_+ = G(p,y_4+1,x_4) , G_- = G(p,y_4-1,x_4), G_c = G(p,y_4,x_4)
    ## 0.5 * g4 * [G_+ - G_-] - 0.5 * ( G_+ + G_- - 2G_c ) * I - i gamma_i sin(p_i) *G_c
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

def wilson_tm_propagator( mass, twisted_mass, spatial_momenta, temporal_momenta, up_down, r = 1 ):
    ''' Returns the quark propagator given the mass of the quark, the twisted mass and some spatial
        momenta k and some temporal momenta k_4. The operator corresponds to a Wilson-Twisted-Mass
        free theory. The value of r can be changed, r = 1 as default.
        up_down must be +1 if the quark correspond to the UP proyection and -1 if corresponds to the
        DOWN proyection. '''

    sum_cos =  ( 1 - np.cos( temporal_momenta ) )
    sum_sin_sq = ( np.sin( temporal_momenta )) ** 2

    for i in range( 0, len(spatial_momenta) ):
        sum_cos += ( 1 - np.cos( spatial_momenta[i] ) )
        sum_sin_sq += ( np.sin( spatial_momenta[i] ) ) ** 2

    gamma_part = gamma_matrix( 4 ) * np.sin( temporal_momenta )
    twisted_part = gamma_matrix( 5 ) * twisted_mass

    for mu in range(0, len(spatial_momenta) ):
        gamma_part += gamma_matrix( mu + 1 ) * np.sin( spatial_momenta[ mu ] )

    denominator = ( mass + r * sum_cos ) ** 2  + sum_sin_sq + twisted_mass ** 2
    num_matrix = ( mass + r * sum_cos ) * identity_matrix() - 1j *  gamma_part - up_down * 1j * twisted_part

    return num_matrix / denominator


if __name__ == '__main__':
    pass


