import numpy as np

def Vol_by_Pr(a,b):
    return (a*b)/1000000000 # show in billions

def All_stocks_Value(a,b,c,d,e,f):
    return a+b+c+d+e+f

def PC_Format(a):
    return (str(np.round(a, 3) * 100) + '%')

