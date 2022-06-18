# +
def logify(x, dx):
    """
    Base-10 log of quantity `x` and its error `dx`
    """
    return np.log10(x), np.log10(np.e)*dx/x

def logratio(x, dx, y, dy):
    '''
    Base-10 log of ratio (`x` +/-`dx`) / (`y` +/-`dy`) plus error
    '''
    r = x/y
    dr_over_r = np.sqrt((dx/x)**2 + (dy/y)**2)
    return np.log10(r), np.log10(np.e)*dr_over_r
