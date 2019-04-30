import numpy as np

__all__ = [ 'float_converter', 'int_converter', 'input_assigner', 'look_for_word' ]

''' ############## Functions to extract data from input ################## '''

def float_converter( my_list ):
    ''' Converts a list of strings into floats'''
    data_conv = []
    for i in range( 0, len( my_list ) ):
        data_conv.append( float( my_list[i] ) )

    if len( data_conv ) == 1:
        return data_conv

    else:
        return data_conv

def int_converter( my_list ):
    ''' Converts a list of strings into integers'''
    data_conv = []
    for i in range( 0, len( my_list ) ):
        data_conv.append( int( my_list[i] ) )

    if len( data_conv ) == 1:
        return data_conv[0]

    else:
        return data_conv

def look_for_word( word_to_find, file_name = 'input.in' ):
    ''' Looks for a word inside a txt and returns the line that contains
        the word we are looking to
    '''
    with open( file_name ) as data:
        content = data.read().splitlines()

    word_line = []

    for i in range(0, len( content ) ):

        if word_to_find in content[i]:
            word_line = ( content[i] )


    return word_line.split()

def input_assigner( word_to_find, file_name = 'input.in' ):
    ''' Returns the values associated to the words given '''

    line_word = look_for_word( word_to_find, file_name )
    word_to_find = word_to_find.split()

    data_extract = []
    for i in range( 0, len(line_word) ):
        if len( word_to_find ) > i:
            pass
        else:
            data_extract.append( line_word[i] )

    return data_extract

if __name__ == '__main__':
    pass
