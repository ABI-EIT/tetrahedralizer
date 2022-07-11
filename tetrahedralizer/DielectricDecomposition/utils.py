from LookupTables import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def chi_lookup(materials, frequencies, include_k0=False):
    """
    Calculates the chi = k0^2(er^2 - 1) spectra for the given materials and frequencies.
    :param materials:
    :param frequencies:
    :return:
    """

    if include_k0:
        k0_squared = [(2 * np.pi * f / SPEED_OF_LIGHT_VACUUM) ** 2 for f in frequencies]
    else:
        k0_squared = [1 for _ in frequencies]  # i.e. use identity to ignore

    n_freq = len(frequencies)
    chi_list = {material: [] for material in materials}

    for material in materials:
        perm, cond = permittivity_lookup(material, frequencies)

        for i in range(n_freq):
            er = np.complex(perm[i], - cond[i] / (2 * np.pi * frequencies[i]))
            chi = k0_squared[i] * (er * er - 1)
            chi_list[material].append(chi)

    return chi_list


def normalise_chi(chi):

    norm_chi = {}

    for key in chi.keys():
        scale_factor = np.sqrt(max([(v * v.conjugate()).real for v in chi[key]]))  # scales by largest amplitude
        if scale_factor > 0:
            norm_chi[key] = [v / scale_factor for v in chi[key]]
        else:
            print('Excluding material {} due to ill-conditioned normalisation.'.format(key))

    return norm_chi


def plot_chi(chi, frequencies, component='Re', tag=''):

    x = [f / 1e9 for f in frequencies]  # GHz
    title = component if tag == '' else tag + ' - ' + component

    fig, ax = plt.subplots(figsize=(8, 8))

    for key in chi.keys():

        if component == 'Re':
            y = [v.real for v in chi[key]]
        else:  # component == 'Im'
            y = [v.imag for v in chi[key]]
        plt.plot(x, y, label=key)

    plt.legend()
    plt.title(title)
    return fig, ax


def svd_basis(chi, component='Re', tol=0.01):
    """

    :param chi:
    :param component:
    :param tol: Significant Singular Value threshold - fraction of max.
    :return:
    """

    # Construct Spectral Matrix
    materials = [k for k in chi.keys()]
    n_materials = len(materials)
    n_freq = len(chi[materials[0]])
    if component in ['Re', 'Im']:
        m = np.zeros(shape=(n_freq, n_materials))
    else:
        m = np.zeros(shape=(n_freq, n_materials), dtype=complex)


    for idx, material in enumerate(materials):
        if component == 'Re':
            m[:, idx] = [v.real for v in chi[material]]
        elif component == 'Im':
            m[:, idx] = [v.imag for v in chi[material]]
        else:  # Use complex number
            m[:, idx] = [v for v in chi[material]]

    # Apply SVD
    u, s, vh = np.linalg.svd(m, full_matrices=False)

    # Determine Most Significant Singular Values
    keep = s >= (s.max() * tol)
    return u[:, keep]


if __name__ == '__main__':
    _materials = VALID_MATERIALS
    _step_GHz = 0.1
    _frequencies = [f * 1e9 for f in np.arange(MIN_FREQUENCY / 1e9, MAX_FREQUENCY / 1e9, _step_GHz)]
    _chi = chi_lookup(_materials, _frequencies)
    _norm_chi = normalise_chi(_chi)

    basis = svd_basis(_norm_chi, component='Re', )
    n_basis = basis.shape[1]
    basis = {i: basis[:, i] for i in range(n_basis)}
    plot_chi(basis, _frequencies, 'Re', tag='SVD')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())

    basis = svd_basis(_norm_chi, component='Im')
    n_basis = basis.shape[1]
    basis = {i: basis[:, i] for i in range(n_basis)}
    plot_chi(basis, _frequencies, 'Im', tag='SVD')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())

    plot_chi(_norm_chi, _frequencies, 'Re', r'Normalised $\chi$ - Real Component')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())
    plot_chi(_chi, _frequencies, 'Im', r'Normalised $\chi$ - Imaginary Component')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())

    basis = svd_basis(_norm_chi, component='Re')
    n_basis = basis.shape[1]
    basis = {i: basis[:, i] for i in range(basis)}
    plot_chi(_chi, _frequencies, 'Re')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())

    print("sdfsdf")




