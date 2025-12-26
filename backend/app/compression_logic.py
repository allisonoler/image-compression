"""Volume 1: The SVD and Image Compression."""

import numpy as np
from imageio.v3 import imread
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    # Calculate the eigenvalues and eigenvectors of AHA
    e_vals, e_vecs = la.eig(A.conj().T@A)
    # Calculate the singular values
    s_vals = np.sqrt(e_vals)
    # Sort the singular values from greatest to smallest, shifting the eigenvalues accordingly.
    correct_indexes = np.argsort(s_vals)[::-1]
    s_vals = s_vals[correct_indexes]
    e_vecs = e_vecs[:, correct_indexes]
    # Find the nonzero singular values
    r = 0
    for val in s_vals:
        if val > tol:
            r += 1
        else:
            break
    # Drop all the zero singular values and corresponding eigenvectors
    s_vals = s_vals[:r]
    e_vecs = e_vecs[:, :r]
    # Calculate U1
    U1 = A @ e_vecs @ np.diag(1/s_vals)

    # Return the compact SVD
    return U1, s_vals, e_vecs.conj().T



# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    # Create the theta values
    thetas = np.linspace(0, 2*np.pi, 100)
    # Create S and E
    x = np.cos(thetas)
    y = np.sin(thetas)
    S = np.vstack((x, y))
    E = np.array([[1, 0, 0], [0, 0, 1]])
    # Calculate the SVD
    U, sigma, V = la.svd(A)

    # Calculate the different transformed version of S and E
    VHS = V.conj().T @ S
    VHE = V.conj().T @ E

    sigVHS = np.diag(sigma) @ VHS
    sigVHE = np.diag(sigma) @ VHE

    UsigVHS = U @ sigVHS
    UsigVHE = U @ sigVHE


    # Plot S, E
    plt.clf()
    plt.subplot(221)
    plt.axis("equal")
    plt.plot(S[0], S[1])
    plt.plot(E[0], E[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("S")

    # Plot VHS, VHE
    plt.subplot(222)
    plt.axis("equal")
    plt.plot(VHS[0], VHS[1])
    plt.plot(VHE[0], VHE[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("VHS")

    # Plot sigVHS, sigVHE
    plt.subplot(223)
    plt.axis("equal")
    plt.plot(sigVHS[0], sigVHS[1])
    plt.plot(sigVHE[0], sigVHE[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("SigmaVHS")

    # Plot UsigVHS, UsigVHE
    plt.subplot(224)
    plt.axis("equal")
    plt.plot(UsigVHS[0], UsigVHS[1])
    plt.plot(UsigVHE[0], UsigVHE[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("USigmaS")

    plt.suptitle("Visualizing the SVD")
    plt.savefig("prob2")


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Check that s is not greater than the rank
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s is larger than the rank of A")

    # Calculate the SVD
    U, sigma, V = la.svd(A, full_matrices=False)

    # Reduce down U, V, and sigma
    U = U[:, :s]
    V = V[:s, :]
    sigma = sigma[:s]

    # Calculate the approximation
    approx_A = U @ np.diag(sigma) @ V

    # Calculate the number of entries
    num_entries = U.size + sigma.size + V.size

    return approx_A, num_entries


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Calculate the SVD
    U, sigma, V = la.svd(A, full_matrices=False)

    # Find all the indices that the s.v. are larger than the error
    indices = np.where(sigma >= err)[0]
    if len(indices) == 0:
        raise ValueError("A cannot be approximated within the tolerance")

    # Calculate s
    s = len(indices)

    # Reduce down U, V, and sigma
    U = U[:, :s]
    V = V[:s, :]
    sigma = sigma[:s]

    # Calculate the approximation
    approx_A = U @ np.diag(sigma) @ V

    # Calculate the number of entries
    num_entries = U.size + sigma.size + V.size

    return approx_A, num_entries


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    # Determine if the image is in color or not
    # color = False
    # image = imread(filename) / 255
    # if len(image.shape) == 3:
    #     color = True
    #
    # # Plot the original image
    # plt.clf()
    # plt.subplot(121)
    # plt.axis("off")
    # if color:
    #     plt.imshow(image)
    # else:
    #     plt.imshow(image, cmap="gray")
    #
    # # Set qualities of the graph
    # plt.title("Original")
    # plt.subplot(122)
    # plt.axis("off")
    #
    # # Initialize variables
    # compressed = None
    # num_entries = 0
    #
    # # The case if the image is in color
    # if color:
    #     # Seperate the layers
    #     R = image[:, :, 0]
    #     G = image[:, :, 1]
    #     B = image[:, :, 2]
    #     # Compress each layer
    #     com_R, R_entries = svd_approx(R, s)
    #     com_G, G_entries = svd_approx(G, s)
    #     com_B, B_entries = svd_approx(B, s)
    #     # Calculate total entries nd put them back togehter
    #     num_entries = R_entries + G_entries + B_entries
    #     compressed = np.stack((com_R, com_G, com_B), axis=2)
    #     # Make sure the clip problematic values
    #     compressed = np.clip(compressed, 0, 1)
    #     plt.imshow(compressed)
    # else:
    #     # Black and white case, just compress and show
    #     compressed, num_entries = svd_approx(image, s)
    #     plt.imshow(compressed, cmap="gray")
    # plt.title("Compressed")
    #
    # # Save the figure.
    # plt.suptitle(f"Compressed has {image.size - num_entries} less entries than the original")
    # plt.savefig("compression")

    return "compressed image hehe"

# if __name__ == '__main__':
#     A = np.array([[3,1],[1,3]])
#     visualize_svd(A)
    # compress_image("hubble.jpg", 20)
    # pass
