'''
File: print.py
Author: Xiaoyi Qu
--------------------------------------------
Description: Helper functions
'''

import numpy as np

def print_header(outID):
    if outID is not None:
        filename = './log/{}.txt'.format(outID)
    else:
        filename = './log/log/log.txt'
    column_titles = ' {Iter:^5s} {f:^11s} {fr:^11s} {g:^11s} {x:^11s} {v:^11s} {u:^11s} \
                      {s:^11s} {c:^11s} {alpha:^11s} {KKT:^11s} {tau:^11s} {phi:^11s} \n'.format(Iter='Iter',
                    f='f', fr='f+r', g='g', x='|x|', v='|v|', u='|u|', s = '|s|', c ='|c|', alpha = 'alpha', 
                    KKT='KKT', tau = 'tau', phi = 'Merit fval')
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


def print_iteration(iteration, fval, frval, normg, normx, normv, normu, norms, normc, alpha, KKTnorm,
                    tau, meritf, outID):
    if outID is not None:
        filename = './log/{}.txt'.format(outID)
    else:
        filename = './log/log.txt'

    contents = "{it:5d} {fval:8.5e} {frval:8.5e} {normg:8.5e} {normx:8.5e} {normv:8.5e} \
                {normu:8.5e} {norms:8.5e} {normc:8.5e} {alpha:8.5e} {KKT:8.5e} {tau:8.5e} \
                {meritf:8.5e} |\n".format(it=iteration, fval=fval, frval=frval, normg=normg, 
                                          normx=normx, normv = normv, normu = normu, norms = norms, 
                                          normc = normc, alpha = alpha, KKT = KKTnorm, tau = tau, 
                                          meritf = meritf)
    with open(filename, "a") as logfile:
        logfile.write(contents)

    

