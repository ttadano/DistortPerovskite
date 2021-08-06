# import argparse
import spglib
import numpy as np
import math
import copy
import collections


class DistortPerovskite(object):

    def __init__(self,
                 elements=None,
                 bond_length=2.0,
                 cellsize=None):

        if elements is None:
            elements = ['Sr', 'Ti', 'O']

        if cellsize is None:
            cellsize = [2, 2, 2]

        self._distorsion_angles = None
        self._elements = elements
        self._bond_length = bond_length
        self._structure_type = None
        self._lattice_vector = None
        self._cellsize = cellsize

        # If you increase bond_length, the appropriate value of the symprec would change.
        self._symprec = 0.5e-2 * bond_length

        # If the length of elements is 3,
        # we assume that they form the ABX3 perovskite.
        # It may be interesting to extend this class for double-perovskite A2BB'X6.
        if len(self._elements) == 3:
            self._structure_type = "perovskite"

        if not self._structure_type:
            raise RuntimeError("The input elements do not form ABX3 perovskite")

        self._element_list = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                              "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                              "K", "Ca",
                              "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                              "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                              "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
                              "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                              "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
                              "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
                              "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
                              "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
                              "Pa", "U", "Np", "Pu"]

        self._element_index = []
        for elem in elements:
            elem_lower = elem.lower()
            found = False
            for i, elem2 in enumerate(self._element_list):
                if elem_lower == elem2.lower():
                    self._element_index.append(i)
                    found = True
                    break
            if not found:
                raise RuntimeError("The input element name does not exist in the element_list.")

        self._element_nums = []
        self._fractional_coordinate = None
        self._cartesian_coordinate = None
        self._supercell = None

        self._build_primitivecell()
        self._build_supercell()
#        self._build_network()
        self._kvecs = None
        self._distorted_supercell = None


    def _build_primitivecell(self):

        lattice_vector = None
        fractional_coordinate = None
        cartesian_coordinate = None
        element_nums = None

        nat_primitive = None

        if self._structure_type == "perovskite":
            lattice_vector = np.array([[self._bond_length * 2.0, 0.0, 0.0],
                                       [0.0, self._bond_length * 2.0, 0.0],
                                       [0.0, 0.0, self._bond_length * 2.0]])

            fractional_coordinate = np.array([[0.5, 0.5, 0.5],  # A cation
                                              [0.0, 0.0, 0.0],  # B cation
                                              [0.5, 0.0, 0.0],  # Anion 1
                                              [0.0, 0.5, 0.0],  # Anion 2
                                              [0.0, 0.0, 0.5],  # Anion 3
                                              [-0.5, 0.0, 0.0],  # Anion 1 in the next cell
                                              [0.0, -0.5, 0.0],  # Anion 2 in the next cell
                                              [0.0, 0.0, -0.5]])  # Anion 3 in the next cell

            element_nums = ([self._element_index[0],
                             self._element_index[1],
                             self._element_index[2],
                             self._element_index[2],
                             self._element_index[2]])

            cartesian_coordinate = np.dot(fractional_coordinate, lattice_vector)
            nat_primitive = 5

        self._lattice_vector = lattice_vector.transpose()
        self._fractional_coordinate = fractional_coordinate
        self._cartesian_coordinate = cartesian_coordinate
        self._element_nums = element_nums
        self._nat_primitive = nat_primitive

    def _build_supercell(self):

        if len(self._cellsize) == 3:
            transformation_matrix = np.zeros((3, 3))
            transformation_matrix[0, 0] = self._cellsize[0]
            transformation_matrix[1, 1] = self._cellsize[1]
            transformation_matrix[2, 2] = self._cellsize[2]
        else:
            raise RuntimeError("The dimension of cellsize must be 3.")

        natom = len(self._fractional_coordinate)
        fractional_coordinate = np.zeros((natom, 3))

        lattice_vector = np.dot(self._lattice_vector, transformation_matrix)

        self._supercell = {'lattice_vector': lattice_vector,
                           'shifts': [],
                           'fractional_coordinates': [],
                           'cartesian_coordinates': []}

        for i in range(self._cellsize[0]):
            for j in range(self._cellsize[1]):
                for k in range(self._cellsize[2]):
                    shift = [i, j, k]
                    for iat in range(natom):
                        x = (self._fractional_coordinate[iat][0] + float(i)) / float(self._cellsize[0])
                        y = (self._fractional_coordinate[iat][1] + float(j)) / float(self._cellsize[1])
                        z = (self._fractional_coordinate[iat][2] + float(k)) / float(self._cellsize[2])
                        fractional_coordinate[iat, 0] = x
                        fractional_coordinate[iat, 1] = y
                        fractional_coordinate[iat, 2] = z

                    self._supercell['shifts'].append(shift)
                    self._supercell['fractional_coordinates'].append(copy.deepcopy(fractional_coordinate))
                    self._supercell['cartesian_coordinates'].append(np.dot(fractional_coordinate, lattice_vector))

    @staticmethod
    def get_rotation_matrix(axis='z', angle=5.0):

        angle_rad = math.pi * angle / 180.0

        if axis == 'x':
            rotmat = np.array([[1.0, 0.0, 0.0],
                               [0.0, math.cos(angle_rad), -math.sin(angle_rad)],
                               [0.0, math.sin(angle_rad), math.cos(angle_rad)]])
        elif axis == 'y':
            rotmat = np.array([[math.cos(angle_rad), 0.0, math.sin(angle_rad)],
                               [0.0, 1.0, 0.0],
                               [-math.sin(angle_rad), 0.0, math.cos(angle_rad)]])
        elif axis == 'z':
            rotmat = np.array([[math.cos(angle_rad), -math.sin(angle_rad), 0.0],
                               [math.sin(angle_rad), math.cos(angle_rad), 0.0],
                               [0.0, 0.0, 1.0]])
        else:
            raise RuntimeError("Invalid rotation axis %s" % axis)

        return rotmat

    @staticmethod
    def get_rotation_matrix_around_axis(axis_vector, angle=5.0):

        angle_rad = math.pi * angle / 180.0
        norm = math.sqrt(np.dot(axis_vector, axis_vector))
        if norm < 1.0e-12:
            raise RuntimeError("The norm of the rotation vector is zero. Return identity matrix.")

        axis_vector /= math.sqrt(np.dot(axis_vector, axis_vector))
        rotmat = np.identity(3) * math.cos(angle_rad) \
                 + (1.0 - math.cos(angle_rad)) * np.outer(axis_vector, axis_vector)

        # matrix representation of cross product n x r
        cross_prod_matrix = np.zeros((3, 3), dtype=float)
        cross_prod_matrix[0, 1] += -axis_vector[2]
        cross_prod_matrix[0, 2] += axis_vector[1]
        cross_prod_matrix[1, 0] += axis_vector[2]
        cross_prod_matrix[1, 2] += -axis_vector[0]
        cross_prod_matrix[2, 0] += -axis_vector[1]
        cross_prod_matrix[2, 1] += axis_vector[0]
        cross_prod_matrix *= math.sin(angle_rad)

        rotmat += cross_prod_matrix

        return rotmat

    def rotate_single_octahedron(self, angles):

        if len(angles) != 3:
            raise RuntimeError("The angles must be an array with 3 elements.")

        rot_x = self.get_rotation_matrix('x', angles[0])
        rot_y = self.get_rotation_matrix('y', angles[1])
        rot_z = self.get_rotation_matrix('z', angles[2])
        rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))
        rotation_matrix2 = np.dot(rot_z, np.dot(rot_y, rot_x))

        for entry in self._cartesian_coordinate[2:]:
            rotated_entry = np.dot(entry, rotation_matrix.transpose())
            rotated_entry2 = np.dot(entry, rotation_matrix2.transpose())

            print(rotated_entry)
            print(rotated_entry2)

        print(rotation_matrix)
        print(rotation_matrix2)

    def rotate_octahedra(self, angles, tilt_patterns):

        if len(angles) != 3 or len(tilt_patterns) != 3:
            raise RuntimeError("The number of entries of angles and tilt_patterns must be 3.")

        self._kvecs = []

        for i, pattern in enumerate(tilt_patterns):
            if angles[i] == 0.0 and pattern != '0':
                print("Warning: The pattern was forced to be '0' since the angle is 0.")
                pattern = '0'

            if pattern == '0' and angles[i] != 0.0:
                print("Warning: The tilt angle is nonzero even when the given pattern is '0'.\n"
                      "         The corresponding angle is set to 0.")
                angles[i] = 0.0

            kvec_tmp = [1, 1, 1]
            if pattern == '0':
                kvec_tmp = [0, 0, 0]
            elif pattern == '+':
                kvec_tmp[i] = 0
            elif pattern == '-':
                kvec_tmp = [1, 1, 1]

            self._kvecs.append(np.array(kvec_tmp))

        self._distorsion_angles = angles

        disp = self._get_displacements(basis='C')

        new_lattice_vector, \
        new_cartesian_coordinates, \
        new_fractional_coordinates = self._adjust_network(disp)

        self._distorted_supercell \
            = {'lattice_vector': new_lattice_vector,
                'shifts': self._supercell['shifts'],
                'fractional_coordinates': new_fractional_coordinates[:,:self._nat_primitive,:],
                'cartesian_coordinates': new_cartesian_coordinates[:,:self._nat_primitive,:]}


    def _get_displacements(self, rigid=True, basis='F', rodrigues=False):
        """Generate displacements in supercell

        This method computes the displacements in the supercell associated with
        the pure rotations around x, y, and z axes.
        Assuming that the rotational
        angle is small, we consider the rotations in three axes independently
        and take the summation of the displacements at the end.
        :param rigid: If true, perform rigid rotation in 3D. If false,
                      the rotation in three axes are performed independently
                      and the final displacements are summed.
        :type rigid: bool
        :param basis: If basis = 'F', the rotation with performed in the fractional
                      coordinate of the primitive lattice.
                      If basis = 'C', the rotation will be done in the Cartesian coordinate.
        :type basis: str
        :return: The displacements in the supercell in the input coordinate
        """

        disp_super = np.zeros((len(self._supercell['shifts']), len(self._cartesian_coordinate), 3))
        rotation_axes = ['x', 'y', 'z']
        nat = len(self._fractional_coordinate)

        if rigid:
            if basis == 'F':
                pos_now = copy.deepcopy(self._fractional_coordinate)
            elif basis == 'C':
                pos_now = copy.deepcopy(self._cartesian_coordinate)

            pos_new = np.zeros((nat, 3))

            if rodrigues:

                disp_now = np.zeros((nat, 3))

                omegavec = self._distorsion_angles[:]
                norm_omegavec = np.sqrt(np.dot(omegavec, omegavec))
                normalized_omegavec = omegavec / norm_omegavec

                rotmat = self.get_rotation_matrix_around_axis(normalized_omegavec, norm_omegavec).transpose()

                for i in range(2):
                    pos_new[i, :] = pos_now[i, :]
                for i in range(2, nat):
                    pos_new[i, :] = np.dot(pos_new[i, :], rotmat)

                disp_now[:, :] = pos_new - pos_now

                # how to handle this part?
                # for ishift, entry in enumerate(self.get_supercell()['shifts']):
                #     disp_super[:, :, ishift] += disp_now[:, :] \
                #                                 * math.cos(math.pi * np.dot(entry, self._kvecs[iax]))

            else:
                disp_now = np.zeros((nat, 3, 3))

                for iax, axis in enumerate(rotation_axes):
                    rotmat = self.get_rotation_matrix(axis,
                                                      self._distorsion_angles[iax]).transpose()
                    for i in range(2):
                        pos_new[i, :] = pos_now[i, :]
                    for i in range(2, nat):
                        pos_new[i, :] = np.dot(pos_now[i, :], rotmat)
                    disp_now[:, :, iax] = pos_new - pos_now
                    pos_now[:, :] = pos_new[:, :]

                    for ishift, entry in enumerate(self.get_supercell()['shifts']):
                        disp_super[ishift, :, :] += disp_now[:, :, iax] \
                                                     * math.cos(math.pi * np.dot(entry, self._kvecs[iax]))

        else:
            for iax, axis in enumerate(rotation_axes):
                rotmat = self.get_rotation_matrix(axis, self._distorsion_angles[iax]).transpose()

                if basis == 'F':
                    pos_new = copy.deepcopy(self._fractional_coordinate)
                elif basis == 'C':
                    pos_new = copy.deepcopy(self._cartesian_coordinate)

                for i in range(2, nat):
                    pos_new[i, :] = np.dot(pos_new[i, :], rotmat)

                if basis == 'F':
                    disp_primitive = pos_new - self._fractional_coordinate
                elif basis == 'C':
                    disp_primitive = pos_new - self._cartesian_coordinate

                for ishift, entry in enumerate(self.get_supercell()['shifts']):
                    disp_super[ishift, :, :] += disp_primitive \
                                                * math.cos(math.pi * np.dot(entry, self._kvecs[iax]))
        return disp_super

    def _adjust_network(self, disp_orig):

        # Brute force way to match the vertex sites

        shifted_cartesian = copy.deepcopy(self._supercell['cartesian_coordinates'])
        shifted_cartesian += disp_orig

        lattice_translation_array = np.array(self._supercell['shifts'])
        neighbor_lists = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        for index_centercell in range(len(self._supercell['cartesian_coordinates'])):

            for idirec, shift_neighbor in enumerate(neighbor_lists):
                lattice_tranlation_adjacent = lattice_translation_array[index_centercell] + shift_neighbor

                index_adjacent_cell = None

                for j, trans in enumerate(lattice_translation_array):
                    diff = lattice_tranlation_adjacent - trans
                    if np.dot(diff, diff) == 0:
                        index_adjacent_cell = j
                        break

                if index_adjacent_cell is None:
                    continue

                xshift = shifted_cartesian[index_adjacent_cell, 5 + idirec, :] \
                         - shifted_cartesian[index_centercell, 2 + idirec, :]

                shifted_cartesian[index_adjacent_cell, :, :] -= xshift

        terminal_lattice_x = np.array([self._cellsize[0]-1, 0, 0])
        terminal_lattice_y = np.array([0, self._cellsize[1]-1, 0])
        terminal_lattice_z = np.array([0, 0, self._cellsize[2]-1])

        index_terminal = []

        for j, trans in enumerate(lattice_translation_array):
            diff = terminal_lattice_x - trans
            if np.dot(diff, diff) == 0:
                index_terminal.append(j)
                break

        for j, trans in enumerate(lattice_translation_array):
            diff = terminal_lattice_y - trans
            if np.dot(diff, diff) == 0:
                index_terminal.append(j)
                break

        for j, trans in enumerate(lattice_translation_array):
            diff = terminal_lattice_z - trans
            if np.dot(diff, diff) == 0:
                index_terminal.append(j)
                break

        new_lattice_vector = copy.deepcopy(self._supercell['lattice_vector'])
        new_lattice_vector[0, :] = 2.0 * (shifted_cartesian[index_terminal[0], 1, :] - shifted_cartesian[0, 1, :])
        new_lattice_vector[1, :] = 2.0 * (shifted_cartesian[index_terminal[1], 1, :] - shifted_cartesian[0, 1, :])
        new_lattice_vector[2, :] = 2.0 * (shifted_cartesian[index_terminal[2], 1, :] - shifted_cartesian[0, 1, :])

        inv_lattice_vector = np.linalg.inv(new_lattice_vector)
        new_fractional_coordinate = np.dot(shifted_cartesian, inv_lattice_vector)

        return new_lattice_vector, shifted_cartesian, new_fractional_coordinate

    def get_symmetrized_structure(self, to_primitive=False):
        supercell, elems = self.get_distorted_structure()
        lattice = supercell['lattice_vector']
        xfrac = np.reshape(supercell['fractional_coordinates'], (len(elems), 3))
        cell = (lattice, xfrac, elems)
        syminfo = spglib.get_spacegroup(cell, symprec=self._symprec)
        lattice, scaled_positions, numbers = spglib.standardize_cell(cell,
                                                                     to_primitive=to_primitive,
                                                                     no_idealize=False,
                                                                     symprec=self._symprec)
        return syminfo, lattice, scaled_positions, numbers

    def write_vasp_poscar(self, fname, to_primitive=False):

        syminfo, lattice, scaled_positions, numbers \
            = self.get_symmetrized_structure(to_primitive=to_primitive)

        formula = self._elements[0] + self._elements[1] + self._elements[2] + "3"

        with open(fname, 'w') as f:

            f.write("%s %s\n" % (syminfo, formula))
            f.write("1.000\n")
            for i in range(3):
                for j in range(3):
                    f.write("%20.14f" % lattice[i][j])
                f.write('\n')

            atomic_numbers_uniq = list(collections.OrderedDict.fromkeys(numbers))
            num_species = []
            for num in atomic_numbers_uniq:
                f.write("%s " % self._element_list[num])
                nspec = len(np.where(np.array(numbers) == num)[0])
                num_species.append(nspec)
            f.write('\n')
            for elem in num_species:
                f.write("%i " % elem)
            f.write('\n')
            f.write('Direct\n')

            for num in atomic_numbers_uniq:
                for i in range(len(scaled_positions)):
                    if numbers[i] == num:
                        f.write("%20.14f " % scaled_positions[i][0])
                        f.write("%20.14f " % scaled_positions[i][1])
                        f.write("%20.14f " % scaled_positions[i][2])
                        f.write('\n')

    def get_original_structure(self):
        return self._lattice_vector, \
               self._element_nums, \
               self._fractional_coordinate

    def get_distorted_structure(self):
        return self._distorted_supercell, self._element_nums * \
               self._cellsize[0] * self._cellsize[1] * self._cellsize[2]

    def get_supercell(self):
        return self._supercell


def check_supercell222():

    # To pass all checks, the rotation angles must be small but should not be too small
    # because the symmetry finder may think the structure is undistorted with the
    # current value of symprec.

    obj = DistortPerovskite(elements=['La', 'Ni', 'O'], bond_length=1.93, cellsize=[2, 2, 2])
    to_primitive = True
    # a0a0a0 (#221)
    obj.rotate_octahedra(angles=[0, 0, 0], tilt_patterns=['0', '0', '0'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0a0a0.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a0a0c+ (#127)
    obj.rotate_octahedra(angles=[0, 0, 1], tilt_patterns=['0', '0', '+'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0a0c+.POSCAR.vasp", to_primitive=to_primitive)

    print(syminfo)

    # a0b+b+ (#139)
    obj.rotate_octahedra(angles=[0, 1, 1], tilt_patterns=['0', '+', '+'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0b+b+.POSCAR.vasp", to_primitive=to_primitive)

    print(syminfo)

    # a+a+a+ (#204)
    obj.rotate_octahedra(angles=[1, 1, 1], tilt_patterns=['+', '+', '+'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a+a+a+.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a+b+c+ (#71)
    obj.rotate_octahedra(angles=[1, 0.5, 2], tilt_patterns=['+', '+', '+'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a+b+c+.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a0a0c- (#140)
    obj.rotate_octahedra(angles=[0, 0, 1], tilt_patterns=['0', '0', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0a0c-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a0b-b- (#74)
    obj.rotate_octahedra(angles=[0, 1, 1], tilt_patterns=['0', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0b-b-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a-a-a- (#167)
    obj.rotate_octahedra(angles=[1, 1, 1], tilt_patterns=['-', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a-a-a-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a0b-c- (#12)
    obj.rotate_octahedra(angles=[0, 0.5, 1], tilt_patterns=['0', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0b-c-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a-b-b- (#15)
    obj.rotate_octahedra(angles=[1, 0.5, 0.5], tilt_patterns=['-', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a-b-b-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a-b-c- (#2)
    obj.rotate_octahedra(angles=[1, 0.5, 2], tilt_patterns=['-', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a-b-c-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a0b+c- (#63)
    obj.rotate_octahedra(angles=[0, 1, 0.5], tilt_patterns=['0', '+', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a0b+c-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a+b-b- (#62)
    obj.rotate_octahedra(angles=[1, 0.5, 0.5], tilt_patterns=['+', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a+b-b-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a+b-c- (#11)
    obj.rotate_octahedra(angles=[1, 0.5, 2], tilt_patterns=['+', '-', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a+b-c-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

    # a+a+c- (#137)
    obj.rotate_octahedra(angles=[1, 1, 0.5], tilt_patterns=['+', '+', '-'])
    syminfo, _, _, _ = obj.get_symmetrized_structure()
    obj.write_vasp_poscar("a+a+c-.POSCAR.vasp", to_primitive=to_primitive)
    print(syminfo)

if __name__ == '__main__':

    check_supercell222()

