''' Main program to calculate the meson correlator in parallel

    -> Sergio Chaves Garcia-Mascaraque
    -> Marzo de 2018 - Octubre de 2018
'''
from __future__ import division

## IMPORT MODULE CREATED TO ALLOW CALCULATIONS
import modules as mod

from mpi4py import MPI
import numpy as np
import os
import sys
import timeit

# Open the communicator -> MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':

    if rank == 0:

        start = timeit.default_timer()

        file_name = str( sys.argv[1] )

        ''' OUTPUT NAME '''
        output_name = mod.input_assigner( 'output_name', file_name )[0]

        ''' IMPORT ATTRIBUTES OF THE SIMULATION '''

        final_momenta = mod.float_converter(
                        mod.input_assigner( 'final_momenta', file_name ) )

        mass = mod.float_converter(
                        mod.input_assigner( 'mass_1 mass_2', file_name ) )

        twisted_mass = mod.float_converter(
                        mod.input_assigner( 'twisted_mass_1 twisted_mass_2',
                                            file_name ) )

        spatial_length = mod.int_converter(
                        mod.input_assigner( 'spatial_length', file_name ) )

        temporal_extent = mod.int_converter(
                        mod.input_assigner( 'temporal_extent', file_name ) )

        t_source = mod.int_converter(
                        mod.input_assigner( 't_source', file_name ) )

        operator_1 = mod.operator_dictionary(
                        mod.input_assigner( 'operator_1 operator_2',
                                file_name ), 1 )

        operator_2 = mod.operator_dictionary(
                        mod.input_assigner( 'operator_1 operator_2',
                                file_name ), 2 )

        flavor_tm = mod.float_converter(
                        mod.input_assigner( 'flavor_tm_1 flavor_tm_2',
                                file_name ) )

        ## Get meson type to calculate
        meson_type = mod.input_assigner( 'meson_type', file_name )[0]

        ''' CALCULATE A BUNCH OF TIMES IF WANTED '''
        time_calc = mod.int_converter(
                        mod.input_assigner( 'time_init time_finit', file_name ) )

        if time_calc[0] == 0 and time_calc[1] == 0:
            time_calc = [0, temporal_extent]

        ''' THETA BOUNDARY CONDITIONS '''
        theta_1 = mod.float_converter(
                        mod.input_assigner( 'theta_1', file_name ) )

        theta_2 = mod.float_converter(
                        mod.input_assigner( 'theta_2', file_name ) )

        theta_1 = np.array( theta_1 ) / spatial_length
        theta_2 = np.array( theta_2 ) / spatial_length

        ''' IMPORT SIMULATION VALUES '''

        r = mod.float_converter( mod.input_assigner( 'r_1 r_2', file_name ) )

        boundary_elec = mod.int_converter(
                        mod.input_assigner( 'boundary_elec', file_name ) )

        ''' IMPORT AVOID MEMORY ERROR VALUE '''
        N_mini = mod.int_converter( mod.input_assigner( 'N_mini', file_name ) )

        ## Data that are arrays
        flavor_tm_1, flavor_tm_2 = flavor_tm[0], flavor_tm[1]

        ''' MASSES AND IMPROVEMENT '''
        improv = mod.int_converter( mod.input_assigner( 'improv', file_name ) )

        ## NOT IMPROVING
        if improv == 0:
            mass_1 = mass[0] / spatial_length
            mass_2 = mass[1] / spatial_length
            twisted_mass_1 = twisted_mass[0] / spatial_length
            twisted_mass_2 = twisted_mass[1] / spatial_length

        ## IMPROVING
        elif improv == 1:
            mass_1 = ( 1 - np.sqrt( 1 - 2 * ( mass[0] / spatial_length
                    + 0.5 * ( twisted_mass[0] / spatial_length ) ** 2 ) ) )

            mass_2 = ( 1 - np.sqrt( 1 - 2 * ( mass[1] / spatial_length
                    + 0.5 * ( twisted_mass[1] / spatial_length ) ** 2 ) ) )

            twisted_mass_1 = twisted_mass[0] / spatial_length
            twisted_mass_2 = twisted_mass[1] / spatial_length

        print( 'MASSES USED IN YOUR SIMULATION' )
        print( mass_1, mass_2, twisted_mass_1, twisted_mass_2 )

        r_1, r_2 = r[0], r[1]

        ''' SELECT THE TIME EXTENT BOUNDARY CONDITION IN THE SIMULATION '''

        if boundary_elec == 0:
            print( 'You are using ANTIPERIODIC BOUNDARY CONDITONS' )
            theory_used = 'ANTI-PER'
            L = [ spatial_length, spatial_length, spatial_length,
                            temporal_extent, temporal_extent ]

            attributes = [ mass_1, mass_2, twisted_mass_1,  twisted_mass_2 ]

        if boundary_elec == 1:
            print( 'You are using the limit T -> infinite' )
            theory_used = 'TINF'
            L = [ spatial_length, spatial_length, spatial_length ]

            attributes = [ mass_1, mass_2, twisted_mass_1, twisted_mass_2 ]

        if boundary_elec == 2:
            print( 'You are using DIRICHLET/OPEN/SF BOUNDARY CONDITIONS' )
            theory_used = 'DIR/OPEN/SF'
            L = [ spatial_length, spatial_length, spatial_length ]

            attributes = [ mass_1, mass_2, twisted_mass_1, twisted_mass_2    ]

        attributes.extend( final_momenta )
        operators = [ operator_1, operator_2 ]

        ''' SIDE PARTITIONS AND SUB-LATTICE BLOCKS COORDINATES '''

        if boundary_elec == 0:

            N_X = mod.int_converter(
                            mod.input_assigner( 'N_X', file_name ) )

            N_Y = mod.int_converter(
                            mod.input_assigner( 'N_Y', file_name ) )

            N_Z = mod.int_converter(
                            mod.input_assigner( 'N_Z', file_name ) )

            N_T1 = mod.int_converter(
                            mod.input_assigner( 'N_T1', file_name ) )

            N_T2 = mod.int_converter(
                            mod.input_assigner( 'N_T2', file_name ) )

            N = [ N_X, N_Y, N_Z, N_T1, N_T2 ]
            if ( N_X * N_Y * N_Z * N_T1 * N_T2 ) != size:
                print( 'Number of divisions of the lattice',
                        ' does not match the number of cores used' )

            ''' MINI_LATTICES BLOCKS TO AVOID MEMORY PROBLEMS '''
            N_mini_X = mod.int_converter(
                            mod.input_assigner( 'N_mini_X', file_name ) )

            N_mini_Y = mod.int_converter(
                            mod.input_assigner( 'N_mini_Y', file_name ) )

            N_mini_Z = mod.int_converter(
                            mod.input_assigner( 'N_mini_Z', file_name ) )

            N_mini_T1 = mod.int_converter(
                            mod.input_assigner( 'N_mini_T1', file_name ) )

            N_mini_T2 = mod.int_converter(
                            mod.input_assigner( 'N_mini_T2', file_name ) )

            N_mini = [ N_mini_X, N_mini_Y, N_mini_Z, N_mini_T1, N_mini_T2 ]
            print( 'Total volume is', np.product( L ) )

            ''' Generate the coordinates that label each block '''

            coords_mpi = []
            for i in range( 0, N[0] ):
                for j in range( 0, N[1] ):
                    for k in range( 0, N[2] ):
                        for t_1 in range( 0, N[3] ):
                            for t_2 in range( 0, N[4] ):
                                coords_mpi.append( [i,j,k,t_1,t_2] )

        elif boundary_elec == 1 or boundary_elec == 2:

            N_X = mod.int_converter( mod.input_assigner( 'N_X', file_name ) )
            N_Y = mod.int_converter( mod.input_assigner( 'N_Y', file_name ) )
            N_Z = mod.int_converter( mod.input_assigner( 'N_Z', file_name ) )

            N = [ N_X, N_Y, N_Z ]
            if ( N_X * N_Y * N_Z ) != size:
                print( 'Number of divisions of the lattice',
                       ' does not match the number of cores used' )

            ''' MINI_LATTICES BLOCKS TO AVOID MEMORY PROBLEMS '''
            N_mini_X = mod.int_converter(
                            mod.input_assigner( 'N_mini_X', file_name ) )

            N_mini_Y = mod.int_converter(
                            mod.input_assigner( 'N_mini_Y', file_name ) )

            N_mini_Z = mod.int_converter(
                            mod.input_assigner( 'N_mini_Z', file_name ) )

            N_mini = [ N_mini_X, N_mini_Y, N_mini_Z ]
            print( 'Total volume is', np.product( L ) )

            ''' Generate the coordinates that label each block '''

            coords_mpi = []
            for i in range( 0, N[0] ):
                for j in range( 0, N[1] ):
                    for k in range( 0, N[2] ):
                        coords_mpi.append( [i,j,k] )


        ''' GENERATE ARRAY TO SCATTER AT EACH RANK FROM MASTER rank == 0 '''

        ### Simulation data
        attributes_mpi = []
        spatial_length_mpi = []
        temporal_extent_mpi = []
        N_mpi = []
        t_source_mpi = []
        operators_mpi = []
        r_1_mpi = []
        r_2_mpi = []
        theta_1_mpi = []
        theta_2_mpi = []
        flavor_tm_1_mpi = []
        flavor_tm_2_mpi = []
        time_calc_mpi = []
        boundary_elec_mpi = []
        meson_type_mpi = []

        ## Avoid memory problems for larger lattices
        N_mini_mpi = []

        ### Result array
        result_sumReal_mpi, result_sumImag_mpi = [], []

        ### Generate N == size arrays to scatter data from master rank
        for i in range( 0, size ):

            ### Simulation data
            attributes_mpi.append( attributes )
            spatial_length_mpi.append( spatial_length )
            temporal_extent_mpi.append( temporal_extent )
            t_source_mpi.append( t_source )
            operators_mpi.append( operators )
            r_1_mpi.append( r_1 )
            r_2_mpi.append( r_2 )
            theta_1_mpi.append( theta_1 )
            theta_2_mpi.append( theta_2 )
            flavor_tm_1_mpi.append( flavor_tm_1 )
            flavor_tm_2_mpi.append( flavor_tm_2 )
            time_calc_mpi.append( time_calc )
            boundary_elec_mpi.append( boundary_elec )
            meson_type_mpi.append( meson_type )

            ### Lattice data
            N_mpi.append( N )
            N_mini_mpi.append( N_mini )

            ### Result array
            result_sumReal_mpi.append( 0 )
            result_sumImag_mpi.append( 0 )

    else:

        ''' GENERATE THE SAME ARRAYS WITH None AT EACH RANK TO AVOID PROBLEMS '''

        ### Simulation data
        attributes_mpi = None
        spatial_length_mpi = None
        temporal_extent_mpi = None
        t_source_mpi = None
        operators_mpi = None
        r_1_mpi = None
        r_2_mpi = None
        theta_1_mpi = None
        theta_2_mpi = None
        flavor_tm_1_mpi = None
        flavor_tm_2_mpi = None
        time_calc_mpi = None
        boundary_elec_mpi = None
        meson_type_mpi = None

        ### Lattice data
        N_mpi = None
        coords_mpi = None

        ## Avoid memory problems
        N_mini_mpi = None

        ### Result array
        result_sumReal_mpi = None
        result_sumImag_mpi = None

    ''' SCATTER ALL THE DATA TO EACH RANK '''

    ### Simulation data
    attributes_scatter = comm.scatter( attributes_mpi, root = 0 )
    spatial_length_scatter = comm.scatter( spatial_length_mpi, root = 0 )
    temporal_extent_scatter = comm.scatter( temporal_extent_mpi, root = 0 )
    t_source_scatter = comm.scatter( t_source_mpi, root = 0 )
    operators_scatter = comm.scatter( operators_mpi, root = 0 )
    r_1_scatter = comm.scatter( r_1_mpi, root = 0 )
    r_2_scatter = comm.scatter( r_2_mpi, root = 0 )
    theta_1_scatter = comm.scatter( theta_1_mpi, root = 0 )
    theta_2_scatter = comm.scatter( theta_2_mpi, root = 0 )
    flavor_tm_1_scatter = comm.scatter( flavor_tm_1_mpi, root = 0 )
    flavor_tm_2_scatter = comm.scatter( flavor_tm_2_mpi, root = 0 )
    time_calc_scatter = comm.scatter( time_calc_mpi, root = 0 )
    boundary_elec_scatter = comm.scatter( boundary_elec_mpi, root = 0 )
    meson_type_scatter = comm.scatter( meson_type_mpi, root = 0 )

    ### Lattice data
    N_scatter = comm.scatter( N_mpi, root = 0 )
    N_mini_scatter = comm.scatter( N_mini_mpi, root = 0 )
    coords_scatter = comm.scatter( coords_mpi, root = 0 )

    ### Result array
    result_sumReal_scatter = comm.scatter( result_sumReal_mpi, root = 0 )
    result_sumImag_scatter = comm.scatter( result_sumImag_mpi, root = 0 )

    ''' PERFORM THE CALCULATION AT EACH RANK '''

    result_sumReal_scatter, result_sumImag_scatter = mod.create_trace_and_integrate(
            meson_type_scatter, boundary_elec_scatter, attributes_scatter,
            N_scatter, coords_scatter, operators_scatter,
            t_source_scatter, spatial_length_scatter, temporal_extent_scatter,
            rank, N_mini_scatter, theta_1_scatter, theta_2_scatter,
            flavor_tm_1_scatter, flavor_tm_2_scatter, time_calc_scatter,
            r_1_scatter, r_2_scatter )

    ''' GATHER ALL THE DATA AT RANK 0 '''
    result_sumReal = comm.gather( result_sumReal_scatter, root = 0 )
    result_sumImag = comm.gather( result_sumImag_scatter, root = 0 )


    ''' OBTAIN RESULTS FROM DATA AND CREATE AN OUTPUT FILE '''

    if rank == 0:

        ### NORMALIZATION TO MATCH OPEN QCD, ADIMENSIONAL QUANTITIES

        ## THE NORMALIZATION FACTOR DEPENDS ON THE STRUCTURE CALCULATED
        if meson_type == '2FLAVORS':
            if boundary_elec == 0:
                normalization = 3 / ( temporal_extent ** 2 )
            else:
                normalization = 3 / ( spatial_length ** 3 )
        elif meson_type == 'BLOB':
            if boundary_elec == 0:
                normalization = 3 / temporal_extent
            else:
                normalization = 3  #/ ( np.sqrt( spatial_length ** 3 ) )

        stop = timeit.default_timer()
        ## Access the data, it's created as tuples inside a list
        time = []
        dataAuxReal, dataAuxImag = 0, 0
        dataCorrReal, dataCorrImag = [], []

        boundDictionary = {
                0 : 'ANTI-PERIODIC',
                1 : 'T INFINITE',
                2 : 'DIRICHLET/OPEN/SF'
                          }

        improvDictionary = {
                0 : 'NOT-IMPROVED',
                1 : 'IMPROVED'
                            }

        for i in range( 0, int( time_calc[1] - time_calc[0] ) ):
            time.append( i )
            for j in range( 0, size ): # Sobre todos los procesadores
                dataAuxReal += [x[i] for x in result_sumReal ][j]
                dataAuxImag += [x[i] for x in result_sumImag ][j]
            dataCorrReal.append( dataAuxReal * normalization )
            dataCorrImag.append( dataAuxImag * normalization )
            data_aux = 0

        print( '---------------------------' )
        print( 'REAL PART OF THE DATA' )
        print( dataCorrReal )
        print( '---------------------------' )
        print( 'IMAGINARY PART OF THE DATA' )
        print( dataCorrImag  )
        print( '---------------------------' )
        print( 'THE SIMULATION HAS FINISHED' )

        path_here = os.path.dirname( os.path.relpath(__file__) )
        subdir = 'RESULTS_DATA'
        filepath = os.path.join( path_here, subdir, output_name )

        #create the subdirectory if does not exist
        if not os.path.exists( subdir ):
            os.mkdir( os.path.join(path_here, subdir) )


        file_ = open( filepath, 'w')

        ## WRITE DOWN THE DATA INTO A FILE
        file_.write( '# L = ' + str(spatial_length) +
                        ' T = ' + str(temporal_extent) + '\n' )
        file_.write( '# meson type calculated : ' + meson_type + '\n' )

        file_.write( '# boundary election : '
                        + str( boundDictionary[boundary_elec] ) + '\n' )

        file_.write( '# mass improvement : '
                        + str( improvDictionary[improv] ) + '\n' )

        file_.write( '# standard mass (m_0 / L) = ' + str(mass_1)
                        + ' ' + str(mass_2) + '\n' )

        file_.write( '# twisted mass (mu / L) = ' + str(twisted_mass_1)
                        + ' ' + str(twisted_mass_2) + '\n' )

        file_.write( '# source position = ' + str(t_source) + '\n' )

        file_.write( '# theta_1 / L = ' + str(theta_1) + '\n' )

        file_.write( '# theta_2 / L = ' + str(theta_2) + '\n' )

        file_.write( '# Computation lasted : ' + str(round( stop - start, 2 ))
                        + ' s' + '\n' )

        file_.write( '# Computation used : ' + str( size ) + 'cores ' + '\n' )

        file_.write( '# ----------------------------------------' + '\n' )

        file_.write( '# y_0' + '\t' + '|C(y_0).real|'
                        + '\t' + '|C(y_0).imag|' + '\n' )

        for i in range( 0, len( dataCorrReal ) ):
            file_.write( str( time[i] ) + '\t' + str( dataCorrReal[i] ) +
                         '\t' + str( dataCorrImag[i] ) + '\n' )

