import os
import numpy as np

SPEED_OF_LIGHT_VACUUM = 2.99792458e8  # m/s
PERMEABILITY_OF_FREE_SPACE = 4e-7 * np.pi  # H/m
PERMITTIVITY_OF_FREE_SPACE = 1 / (PERMEABILITY_OF_FREE_SPACE * SPEED_OF_LIGHT_VACUUM**2)  # F/m
IMPEDANCE_OF_FREE_SPACE = np.sqrt(PERMEABILITY_OF_FREE_SPACE / PERMITTIVITY_OF_FREE_SPACE)
INV_IMPEDANCE_OF_FREE_SPACE = 1 / IMPEDANCE_OF_FREE_SPACE

# Note that permittivity is frequency dependent.
VALID_MATERIALS = [
    'air',
    'free space',
    'water',
    'blood',
    'bone (cancellous)',
    'bone (cortical)',
    'bone marrow (red)',
    'bone marrow (yellow)',
    'breast fat',
    'cartilage',
    'fat',
    'lung (deflated)',
    'lung (inflated)',
    'muscle',
    'skin'
]

MIN_FREQUENCY = 1e9   # 1 GHz
MAX_FREQUENCY = 8e9  # 10 GHz


def permittivity_lookup(material, frequencies):
    """
    Extract electrical permittivity and conductivity values for phantom materials from lookup tables. These
    values are returned in dictionaries of the same format as the material dictionary generated alongside
    the phantom specification in generate_phantom().
    :param material: material index dictionary produced alongside the phantom specification.
    :param frequencies: radiation frequency (Hz).
    :return: permittivity_dict, conductivity_dict
    """

    # --------------- #
    # Validate Inputs #
    # --------------- #
    if material not in VALID_MATERIALS:
        raise ValueError('Unknown material: ' + str(material))

    frequencies = frequencies if hasattr(frequencies, '__iter__') else [frequencies]  # Convert freq to list if singleton.
    if min(frequencies) < MIN_FREQUENCY or max(frequencies) > MAX_FREQUENCY:
        raise ValueError('Requested frequencies not within available range [{:0.1f}GHz, {:0.1f}GHz].'.format(MIN_FREQUENCY, MAX_FREQUENCY))

    permittivity = []
    conductivity = []

    for freq in frequencies:

        # Special Cases
        # -------------
        if material == 'air' or material == 'free space':
            _er, _cond = 1.0, 0.0
        elif material == 'bone phantom 1':  # Experimental Measurements
            _er, _cond = 18.0, 3.7 * PERMITTIVITY_OF_FREE_SPACE * freq
        elif material == 'fat phantom 1':  # Experimental Measurements
            _er, _cond = 9.3, 1.5 * PERMITTIVITY_OF_FREE_SPACE * freq
        else:
            # Load Sources
            # ------------
            source_path = os.path.join(os.path.split(__file__)[0], 'permittivity', material + '.txt')
            assert os.path.isfile(source_path), "Cannot find '{}' in lookup_tables.".format(material)
            src_freq, src_perm, src_cond = list(), list(), list()
            with open(source_path, 'r') as f:
                lines = f.readlines()
            for i in range(2, len(lines)):
                line = lines[i].replace('\n', '')
                if line == '':
                    continue
                fr, er, sig = line.split('\t')
                src_freq.append(float(fr))
                src_perm.append(float(er))
                src_cond.append(float(sig))

            # Regress around target frequency
            # -------------------------------
            idx = (np.abs(np.array(src_freq) - freq)).argmin()  # index of closest frequency in lookup
            if src_freq[idx] > freq:
                f1 = src_freq[idx - 1]
                e1 = src_perm[idx - 1]
                c1 = src_cond[idx - 1]
                f2 = src_freq[idx]
                e2 = src_perm[idx]
                c2 = src_cond[idx]
                _er = (e2 - e1) / (f2 - f1) * (freq - f1) + e1
                _cond = (c2 - c1) / (f2 - f1) * (freq - f1) + c1
            elif src_freq[idx] < freq:
                f1 = src_freq[idx]
                e1 = src_perm[idx]
                c1 = src_cond[idx]
                f2 = src_freq[idx + 1]
                e2 = src_perm[idx + 1]
                c2 = src_cond[idx + 1]
                _er = (e2 - e1) / (f2 - f1) * (freq - f1) + e1
                _cond = (c2 - c1) / (f2 - f1) * (freq - f1) + c1
            else:
                _er, _cond = src_perm[idx], src_cond[idx]

        # Update Dictionaries
        permittivity.append(_er)
        conductivity.append(_cond)

    return permittivity, conductivity