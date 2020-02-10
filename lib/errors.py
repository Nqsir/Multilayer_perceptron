def display_errors_dict(code_error):
    """
    Display an error message
    :param code_error: The list returned from check_data()
    """

    dictionary = {
        'endswith': f'Invalid extension\n\n',
        'exists': f'File does not exist\n\n',
        'isfile': f'Not a file\n\n',
        'check_nb_column': f'Wrong number of columns\n\n',
        'no_order': f'No order received\n\n',
        'no_network': f'No "Network.save", please run train mode first\n\n',
        'wrong_load': f'Loading file does not exist\n\n',
    }
    print(f'\n{dictionary[code_error]}')
