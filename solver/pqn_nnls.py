#!/usr/bin/python
# -*- coding: utf-8 -*-

# PQN-NNLS algorithm (Kim, Sra, Dhillon 2006)
import numpy as np

def pqn_nnls(A, b, err, limit = 300):
    """return x which is >= 0 and minimizes ||Ax - b||"""
    m, n = A.shape;
    AtA = np.dot(A.T, A)
    Atb = np.dot(A.T, b)

    def proj(x):
        return x * (x>=0)

    curS = np.eye(n)
    x = np.zeros(n, 1)

    for iteration in range(limit):
        # gradient for current x 
        grad = AtA*x - Atb

        fixed_set = []
        free_set = []
        for i in range(n):
            if (abs(x[i]) < err) and (grad[i] > err):
                fixed_set.append(i)
            else:
                free_set.append(i)

        cur_y = x[free_set]
        grad_y = grad[free_set];
 
        subS = curS[free_set, free_set]
        subA = A(:, free_set);

        def obj(x):
            return 1/2 * norm(subA*x - b) ** 2
        def gamma(bt):
            return proj(cur_y - bt*subS*grad_y)

        # using APA rule
        alpha = 1;
        sigma = 1;
        s = 1/2;
        tau = 1/4;
        m = 0;
        storedgamma = gamma(s^m*sigma)
        while (obj(cur_y)-obj(storedgamma)) < tau*grad_y.T * (cur_y - storedgamma)
             m = m+1;
             storedgamma = gamma(s^m*sigma);

        d = storedgamma - cur_y;

        u = x;
        u[free_set] = alpha*d;
        u[fixed_set] = 0;

        pre_b = np.dot(A, u)
        temp2 = np.dot(At, pre_b)

        temp3 = np.dot(u, u.T)
        temp4 = np.dot(pre_b.T, pre_b)
        temp5 = np.dot(curS, temp2, u.T)

        curS = ((1 + temp2.T*curS*temp2/temp4)*temp3 - ...
            (temp5+temp5.\'))/temp4 + curS;

        curs += 
        if norm(x[free_set] - cur_y - alpha*d) < err
            break;

        x[free_set] = cur_y + alpha*d;

    return x;