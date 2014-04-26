from __future__ import division
from __future__ import print_function
import numpy as np
from math import sqrt, cos, acos, pi

def eig_symm(A):
    # Given a real symmetric 3x3 matrix A, compute the eigenvalues
    p1 = np.sum(np.tril(A, -1)**2)
    if p1 == 0: 
        # A is diagonal.
        eig1, eig2, eig3 = np.diag(A)
    else:
        q = np.trace(A)/3
        p2 = np.sum((np.diag(A)-q)**2) + 2*p1
        p = sqrt(p2/6)
        B = (1/p) * (A - q*np.eye(3))
        r = np.linalg.det(B) / 2
        # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if (r <= -1):
            phi = pi / 3
        elif (r >= 1):
            phi = 0
        else:
            phi = acos(r) / 3
        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        eig1 = q + 2 * p * cos(phi)
        eig3 = q + 2 * p * cos(phi + (2*pi/3))
        eig2 = 3 * q - eig1 - eig3     # since trace(A) = eig1 + eig2 + eig3
    # compute eigenvectors
    eigv1 = np.cross(A[:,0]-np.array([eig1, 0, 0]), A[:,1]-np.array([0, eig1, 0]))
    eigv2 = np.cross(A[:,0]-np.array([eig2, 0, 0]), A[:,1]-np.array([0, eig2, 0]))
    eigv3 = np.cross(eigv1, eigv2)
    if np.linalg.norm(eigv1): eigv1 = eigv1/np.linalg.norm(eigv1)
    if np.linalg.norm(eigv2): eigv2 = eigv2/np.linalg.norm(eigv2)
    if np.linalg.norm(eigv3): eigv3 = eigv3/np.linalg.norm(eigv3)
    return np.array([eig1, eig2, eig3]), np.array([eigv1, eigv2, eigv3]).transpose()

if __name__ == '__main__':
    #v = np.array([1, 1, 0.001])
    #A = v[:, np.newaxis] * v[np.newaxis, :]
    #A = np.array([[  2.63783090e-06,  -2.03845602e-07,   1.82752444e-06],
    #              [ -2.03845602e-07,   2.79856908e-06,   2.13257385e-06],
    #              [  1.82752444e-06,   2.13257385e-06,   8.06261323e-06]])
    #A = np.array([[ 3,  2,  6],
    #              [ 2,  2,  5],
    #              [-2, -1,  4]])
    A = np.array([[ 7, -2,  0],
                  [-2,  6, -2],
                  [ 0, -2,  5]])
    print(A)
    print("eigsymm(A):")
    eigvals, eigvecs = eig_symm(A)
    print(eigvals)
    print(eigvecs)
    print("np.linalg.eigh(A):")
    eigvals, eigvecs = np.linalg.eigh(A)
    print(eigvals)
    print(eigvecs)
