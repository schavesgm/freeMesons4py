"""
    Author: Sergio Chaves Garcia-Mascaraque
    E-mail: sergiozteskate@gmail.com

    Main script to perform the calculation in parallel and obtain the
    mesonic correlation function
"""

from __future__ import division

# Import module inside ./modules folder
import modules as mod

# Load the modules needed
from mpi4py import MPI
import numpy as np
import os, sys, timeit

# Open the communicator -> MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':

    if rank == 0:

        start = timeit.default_timer()

        # File where the input file is saved
        fileName = str( sys.argv[1] )
        # Output file name
        output_name = mod.input_assigner( 'output_name', fileName )[0]

        # Attributes of the simulation
        spatial_length = mod.int_converter(
                        mod.input_assigner( 'spatial_length', fileName ) )

        temporal_extent = mod.int_converter(
                        mod.input_assigner( 'temporal_extent', fileName ) )

        t_source = mod.int_converter(
                        mod.input_assigner( 't_source', fileName ) )

        mass = mod.float_converter(
                        mod.input_assigner( 'mass_1 mass_2', fileName ) )

        twisted_mass = mod.float_converter(
                        mod.input_assigner( 'twisted_mass_1 twisted_mass_2',
                                            fileName ) )

        final_momenta = mod.float_converter(
                        mod.input_assigner( 'final_momenta', fileName ) )

        r = mod.float_converter( mod.input_assigner( 'r_1 r_2', fileName ) )

        nameOp = mod.input_assigner( 'operator_1 operator_2', fileName )

        operator_1 = mod.operator_dictionary(
                        mod.input_assigner( 'operator_1 operator_2',
                                fileName ), 1 )

        operator_2 = mod.operator_dictionary(
                        mod.input_assigner( 'operator_1 operator_2',
                                fileName ), 2 )

        flavor_tm = mod.float_converter(
                        mod.input_assigner( 'flavor_tm_1 flavor_tm_2',
                                fileName ) )

        # Get meson type to calculate
        meson_type = mod.input_assigner( 'meson_type', fileName )[0]

        # Calculate a bunch of time if wanted
        time_calc = mod.int_converter(
                        mod.input_assigner( 'time_init time_finit', fileName ) )

        if time_calc[0] == 0 and time_calc[1] == 0:
            time_calc = [0, temporal_extent]

        # Theta boundary conditions
        theta_1 = mod.float_converter(
                        mod.input_assigner( 'theta_1', fileName ) )

        theta_2 = mod.float_converter(
                        mod.input_assigner( 'theta_2', fileName ) )

        theta_1 = np.array( theta_1 ) / spatial_length
        theta_2 = np.array( theta_2 ) / spatial_length

        # Boundary condition selected
        boundary_elec = mod.int_converter(
                        mod.input_assigner( 'boundary_elec', fileName ) )

        # Avoid memory errors in large cases
        N_mini = mod.int_converter( mod.input_assigner( 'N_mini', fileName ) )

        # Data that are arrays
        flavor_tm_1, flavor_tm_2 = flavor_tm[0], flavor_tm[1]

        # Check for improvement
        improv = mod.int_converter( mod.input_assigner( 'improv', fileName ) )

        if improv == 0:
            mass_1 = mass[0] / spatial_length
            mass_2 = mass[1] / spatial_length
            twisted_mass_1 = twisted_mass[0] / spatial_length
            twisted_mass_2 = twisted_mass[1] / spatial_length

        # Improve the calculation
        elif improv == 1:
            mass_1 = ( 1 - np.sqrt( 1 - 2 * ( mass[0] / spatial_length \
                    + 0.5 * ( twisted_mass[0] / spatial_length ) ** 2 ) ) )

            mass_2 = ( 1 - np.sqrt( 1 - 2 * ( mass[1] / spatial_length \
                    + 0.5 * ( twisted_mass[1] / spatial_length ) ** 2 ) ) )

            twisted_mass_1 = twisted_mass[0] / spatial_length
            twisted_mass_2 = twisted_mass[1] / spatial_length

        print( 'MASSES USED IN YOUR SIMULATION' )
        print( mass_1, mass_2, twisted_mass_1, twisted_mass_2 )

        r_1, r_2 = r[0], r[1]

        # Select the boundary condition used in the calculation
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

        # Parallelize the calculation dividing in blocks the lattice
        if boundary_elec == 0:

            N_X = mod.int_converter(
                            mod.input_assigner( 'N_X', fileName ) )

            N_Y = mod.int_converter(
                            mod.input_assigner( 'N_Y', fileName ) )

            N_Z = mod.int_converter(
                            mod.input_assigner( 'N_Z', fileName ) )

            N_T1 = mod.int_converter(
                            mod.input_assigner( 'N_T1', fileName ) )

            N_T2 = mod.int_converter(
                            mod.input_assigner( 'N_T2', fileName ) )

            N = [ N_X, N_Y, N_Z, N_T1, N_T2 ]

            if ( N_X * N_Y * N_Z * N_T1 * N_T2 ) != size:
                print( 'Number of divisions of the lattice',
                        ' does not match the number of cores used' )

            # Mini-lattices inside each block to avoid memory problems
            # when the number of processors is limited
            N_mini_X = mod.int_converter(
                            mod.input_assigner( 'N_mini_X', fileName ) )

            N_mini_Y = mod.int_converter(
                            mod.input_assigner( 'N_mini_Y', fileName ) )

            N_mini_Z = mod.int_converter(
                            mod.input_assigner( 'N_mini_Z', fileName ) )

            N_mini_T1 = mod.int_converter(
                            mod.input_assigner( 'N_mini_T1', fileName ) )

            N_mini_T2 = mod.int_converter(
                            mod.input_assigner( 'N_mini_T2', fileName ) )

            N_mini = [ N_mini_X, N_mini_Y, N_mini_Z, N_mini_T1, N_mini_T2 ]
            print( 'Total volume is', np.product( L ) )

            # Generate the coordinates that label each block
            coords_mpi = []
            for i in range( 0, N[0] ):
                for j in range( 0, N[1] ):
                    for k in range( 0, N[2] ):
                        for t_1 in range( 0, N[3] ):
                            for t_2 in range( 0, N[4] ):
                                coords_mpi.append( [i,j,k,t_1,t_2] )

        elif boundary_elec == 1 or boundary_elec == 2:

            N_X = mod.int_converter( mod.input_assigner( 'N_X', fileName ) )
            N_Y = mod.int_converter( mod.input_assigner( 'N_Y', fileName ) )
            N_Z = mod.int_converter( mod.input_assigner( 'N_Z', fileName ) )

            N = [ N_X, N_Y, N_Z ]
            if ( N_X * N_Y * N_Z ) != size:
                print( 'Number of divisions of the lattice',
                       ' does not match the number of cores used' )


            # Mini-lattices inside each block to avoid memory problems
            # when the number of processors is limited
            N_mini_X = mod.int_converter(
                            mod.input_assigner( 'N_mini_X', fileName ) )

            N_mini_Y = mod.int_converter(
                            mod.input_assigner( 'N_mini_Y', fileName ) )

            N_mini_Z = mod.int_converter(
                            mod.input_assigner( 'N_mini_Z', fileName ) )

            N_mini = [ N_mini_X, N_mini_Y, N_mini_Z ]
            print( 'Total volume is', np.product( L ) )

            # Generate the coordinates that label each block
            coords_mpi = []
            for i in range( 0, N[0] ):
                for j in range( 0, N[1] ):
                    for k in range( 0, N[2] ):
                        coords_mpi.append( [i,j,k] )

    else:

        # Generate the variables in all the ranks to be broadcaster

        # Scattered data
        coords_mpi = None

        # Broadcasted data
        attributes = None
        spatial_length = None
        temporal_extent = None
        t_source = None
        operators = None
        r_1 = None
        r_2 = None
        theta_1 = None
        theta_2 = None
        flavor_tm_1 = None
        flavor_tm_2 = None
        time_calc = None
        boundary_elec = None
        meson_type = None

        N_mini = None
        N = None

    # Barrier to the processors to keep things tidy and clean
    comm.Barrier()

    # Broadcast the data to all ranks

    # Simulation data
    attributes = comm.bcast( attributes, root = 0 )
    spatial_length = comm.bcast( spatial_length, root = 0 )
    temporal_extent = comm.bcast( temporal_extent, root = 0 )
    t_source = comm.bcast( t_source, root = 0 )
    operators = comm.bcast( operators, root = 0 )
    r_1 = comm.bcast( r_1, root = 0 )
    r_2 = comm.bcast( r_2, root = 0 )
    theta_1 = comm.bcast( theta_1, root = 0 )
    theta_2 = comm.bcast( theta_2, root = 0 )
    flavor_tm_1 = comm.bcast( flavor_tm_1, root = 0 )
    flavor_tm_2 = comm.bcast( flavor_tm_2, root = 0 )
    time_calc = comm.bcast( time_calc, root = 0 )
    boundary_elec = comm.bcast( boundary_elec, root = 0 )
    meson_type = comm.bcast( meson_type, root = 0 )

    # Parallelization data
    N = comm.bcast( N, root = 0 )
    N_mini = comm.bcast( N_mini, root = 0 )

    # Only thing needed to be scattered
    coords_scatter = comm.scatter( coords_mpi, root = 0 )

    # Barrier to the processors to keep things tidy and clean
    comm.Barrier()

    # Perform the integration at each rank
    result_sumReal, result_sumImag = mod.create_trace_and_integrate(
            meson_type, boundary_elec, attributes,
            N, coords_scatter, operators,
            t_source, spatial_length, temporal_extent,
            rank, N_mini, theta_1, theta_2,
            flavor_tm_1, flavor_tm_2, time_calc,
            r_1, r_2 )

    # Gather data to rank number 0 to extract the corr function
    result_sumReal = comm.gather( result_sumReal, root = 0 )
    result_sumImag = comm.gather( result_sumImag, root = 0 )


    # Obtain the results and flush them into the output file
    if rank == 0:

        # Normalization factors to match openQCD. Adimensional quantities
        # Normalization depends on the calculated correlation function and
        # the boundary conditions used
        if meson_type == '2FLAVORS':
            if boundary_elec == 0:
                normalization = 3 / ( temporal_extent ** 2 )
            else:
                normalization = 3

        elif meson_type == 'BLOB':
            if boundary_elec == 0:
                normalization = 3 / temporal_extent
            else:
                normalization = 3

        stop = timeit.default_timer()
        # Access the data, it's created as tuples inside a list
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
            time.append( time_calc[0] + i )

            for j in range( 0, size ): # Sobre todos los procesadores
                dataAuxReal += [x[i] for x in result_sumReal ][j]
                dataAuxImag += [x[i] for x in result_sumImag ][j]

            dataCorrReal.append( dataAuxReal * normalization )
            dataCorrImag.append( dataAuxImag * normalization )
            dataAuxReal = 0
            dataAuxImag = 0

        # Display the results on screen for a quick check
        print( '---------------------------' )
        print( 'REAL PART OF THE DATA' )
        print( dataCorrReal )
        print( '---------------------------' )
        print( 'IMAGINARY PART OF THE DATA' )
        print( dataCorrImag  )
        print( '---------------------------' )
        print( 'THE SIMULATION HAS FINISHED' )

        path_here = os.path.dirname( os.path.relpath(__file__) )
        subdir = 'results'
        filepath = os.path.join( path_here, subdir, output_name )

        # Create the subdirectory if does not exist
        if not os.path.exists( subdir ):
            os.mkdir( os.path.join(path_here, subdir) )


        ouFile = open( filepath, 'w')

        # Flush the data into a file
        ouFile.write( '# L = ' + str(spatial_length) +
                        ' T = ' + str(temporal_extent) + '\n' )

        ouFile.write( '# meson type calculated : ' + meson_type + '\n' )

        ouFile.write( '# Operators used : ' + str(nameOp[0]) + \
                     ' ' + str(nameOp[1]) + '\n' )

        ouFile.write( '# boundary election : '
                        + str( boundDictionary[boundary_elec] ) + '\n' )

        ouFile.write( '# mass improvement : '
                        + str( improvDictionary[improv] ) + '\n' )

        ouFile.write( '# standard mass (m_0 / L) = ' + str(mass_1)
                        + ' ' + str(mass_2) + '\n' )

        ouFile.write( '# twisted mass (mu / L) = ' + str(twisted_mass_1)
                        + ' ' + str(twisted_mass_2) + '\n' )

        ouFile.write( '# source position = ' + str(t_source) + '\n' )

        ouFile.write( '# theta_1 / L = ' + str(theta_1) + '\n' )

        ouFile.write( '# theta_2 / L = ' + str(theta_2) + '\n' )

        ouFile.write( '# Computation lasted : ' + str(round( stop - start, 2 ))
                        + ' s' + '\n' )

        ouFile.write( '# Computation used : ' + str( size ) + 'cores ' + '\n' )

        ouFile.write( '# ----------------------------------------' + '\n' )

        ouFile.write( '# y_0' + '\t' + '|C(y_0).real|'
                        + '\t' + '|C(y_0).imag|' + '\n' )

        # Write the actual data, first column is time, second is the real
        # part of the correlation function and third is the imaginary part
        for i in range( 0, len( dataCorrReal ) ):
            ouFile.write( str( time[i] ) + '\t' + str( dataCorrReal[i] ) +
                         '\t' + str( dataCorrImag[i] ) + '\n' )

