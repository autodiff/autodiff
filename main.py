def declare_variables(variables, macro):
    """
    This is the hook for the functions

    - variables: the dictionary that contains the variables
    - macro: a decorator function, to declare a macro.
    """

    @macro
    def inputcode(filename, language):
        f = open(filename, 'r')
        text = f.read()
        textblock = f'```{language}\n{text}\n```'
        return textblock

    @macro
    def inputcpp(filename):
        return inputcode(filename, 'cpp')
