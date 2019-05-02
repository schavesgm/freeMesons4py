"""
    Author: Sergio Chaves García-Mascaraque
    E-mail: sergiozteskate@gmail.com

    Module containing all matrix dependencies in freeMeson4py
"""

import numpy as np

__all__ = [ 'operator_dictionary',
            'gamma_matrix',
            'identity_matrix' ]

def operator_dictionary( operator_signal, number ):
    '''
        Dictionary of operators:

        Converts an string with to a given matrix operator.
        Called in the main file to generate the matrix needed.

    '''

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

def gamma_matrix( mu ):
    '''
    Returns the gamma euclidean matrix given the mu index
    '''

    # Define the variables used -- avoid precision problems
    zeros = 0.0 + 0.0j
    imagp = 0.0 + 1.0j
    imagn = 0.0 - 1.0j
    onesp = 1.0 + 0.0j
    onesn = -1.0 + 0.0j

    # Check for possible errors
    if ( mu == 0 or mu > 5 ):
        print( 'ERROR. The index runs from 1 to 5' )
    # Return the matrices
    if mu == 1:  # \gamma_1
        return np.matrix( [[zeros, zeros, zeros, imagn], \
                           [zeros, zeros, imagn, zeros], \
                           [zeros, imagp, zeros, zeros], \
                           [imagp, zeros, zeros, zeros]], \
                           dtype = complex )
    if mu == 2:  # \gamma_2
        return np.matrix( [[zeros, zeros, zeros, onesn], \
                           [zeros, zeros, onesp, zeros], \
                           [zeros, onesp, zeros, zeros], \
                           [onesn, zeros, zeros, zeros]], \
                           dtype = complex )
    if mu == 3:  # \gamma_3
        return np.matrix( [[zeros, zeros, imagn, zeros], \
                           [zeros, zeros, zeros, imagp], \
                           [imagp, zeros, zeros, zeros], \
                           [zeros, imagn, zeros, zeros]], \
                           dtype = complex )
    if mu == 4:  # \gamma_4 == \gamma_0
        return np.matrix( [[zeros, zeros, onesp, zeros], \
                           [zeros, zeros, zeros, onesp], \
                           [onesp, zeros, zeros, zeros], \
                           [zeros, onesp, zeros, zeros]], \
                           dtype = complex )
    if mu == 5:  # \gamma_5
        return np.matrix( [[onesp, zeros, zeros, zeros], \
                           [zeros, onesp, zeros, zeros], \
                           [zeros, zeros, onesn, zeros], \
                           [zeros, zeros, zeros, onesn]], \
                           dtype = complex )

def identity_matrix( dim = 4 ):
    '''
        Returns the identity delta kronnecker given the dimension.
        The default value is the identity in 4 dimensions.

        This module is the same as np.eye( 4 )
    '''

    delta = np.zeros( [dim,dim] )
    for i in range( 0, dim ):
        delta[i,i] = 1.0

    return np.matrix( delta, dtype = complex )

if __name__ == '__main__':
    pass
