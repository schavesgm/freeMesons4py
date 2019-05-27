import numpy as np

__all__ = [ 'float_converter', 'int_converter', 'input_assigner', 'look_for_word' ]

''' Functions to extract data from input '''

def float_converter( my_list ):
    '''
        @arg (list) : List of strings to be converted into list of floats.

        return      : List containing floats with the values inside my_list.
    '''

    data_conv = []
    for i in range( 0, len( my_list ) ):
        data_conv.append( float( my_list[i] ) )

    if len( data_conv ) == 1:
        return data_conv

    else:
        return data_conv

def int_converter( my_list ):
    '''
        @arg (list) : List of strings to be converted into list of integers.

        return      : List containing integers with the values inside my_list.
    '''


    data_conv = []
    for i in range( 0, len( my_list ) ):
        data_conv.append( int( my_list[i] ) )

    if len( data_conv ) == 1:
        return data_conv[0]

    else:
        return data_conv

def look_for_word( word_to_find, file_name = 'input.in' ):
    '''
        @arg (string): String containing the words to find inside the file.

        Optional:
        @arg (string): String with the name of the file to look into.

        return       : Line in which the word wanted appears. If it appears more
                       than once, it would return the last case.
    '''

    with open( file_name ) as data:
        content = data.read().splitlines()

    word_line = []
    for i in range(0, len( content ) ):
        if word_to_find in content[i]:
            word_line = ( content[i] )

    return word_line.split()

def input_assigner( word_to_find, file_name = 'input.in' ):
    '''
        @arg (string): Word to be found inside the file

        Optional
        @arg (string): Name of the file in which we want to look for the word.

        return       : Values associated with the word given. A list will be
                       returned in case we have more than one strings after
                       word_to_find in the file. Note that it looks for the pattern
                       'word_to_find A B C...'
    '''

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
