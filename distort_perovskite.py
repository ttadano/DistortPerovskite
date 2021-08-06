# import argparse
import spglib
import numpy as np
import math
import copy


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
        self._kvecs = None

    def _build_primitivecell(self):

        lattice_vector = None
        fractional_coordinate = None
        cartesian_coordinate = None
        element_nums = None

        if self._structure_type == "perovskite":
            lattice_vector = np.array([[self._bond_length * 2.0, 0.0, 0.0],
                                       [0.0, self._bond_length * 2.0, 0.0],
                                       [0.0, 0.0, self._bond_length * 2.0]])

            fractional_coordinate = np.array([[0.5, 0.5, 0.5],  # A cation
                                              [0.0, 0.0, 0.0],  # B cation
                                              [0.5, 0.0, 0.0],  # Anion 1
                                              [0.0, 0.5, 0.0],  # Anion 2
                                              [0.0, 0.0, 0.5]])  # Anion 3

            element_nums = ([self._element_index[0],
                             self._element_index[1],
                             self._element_index[2],
                             self._element_index[2],
                             self._element_index[2]])

            cartesian_coordinate = np.dot(fractional_coordinate, lattice_vector)

        self._lattice_vector = lattice_vector.transpose()
        self._fractional_coordinate = fractional_coordinate
        self._cartesian_coordinate = cartesian_coordinate
        self._element_nums = element_nums

    def _build_supercell(self):

        if len(self._cellsize) == 3:
            transformation_matrix = np.zeros((3, 3))
            transformation_matrix[0, 0] = self._cellsize[0]
            transformation_matrix[1, 1] = self._cellsize[1]
            transformation_matrix[2, 2] = self._cellsize[2]
        else:
            raise RuntimeError("The dimension of cellsize must be 3.")

        nat_primitive = len(self._fractional_coordinate)
        fractional_coordinate = np.zeros((nat_primitive, 3))

        lattice_vector = np.dot(self._lattice_vector, transformation_matrix)

        self._supercell = {'lattice_vector': lattice_vector,
                           'shifts': [],
                           'fractional_coordinates': []}

        for i in range(self._cellsize[0]):
            for j in range(self._cellsize[1]):
                for k in range(self._cellsize[2]):
                    shift = [i, j, k]
                    for iat in range(nat_primitive):
                        x = (self._fractional_coordinate[iat][0] + float(i)) / float(self._cellsize[0])
                        y = (self._fractional_coordinate[iat][1] + float(j)) / float(self._cellsize[1])
                        z = (self._fractional_coordinate[iat][2] + float(k)) / float(self._cellsize[2])
                        fractional_coordinate[iat, 0] = x
                        fractional_coordinate[iat, 1] = y
                        fractional_coordinate[iat, 2] = z

                    self._supercell['shifts'].append(shift)
                    self._supercell['fractional_coordinates'].append(copy.deepcopy(fractional_coordinate))

    def _build_network(self):
        network = {}

        for ishift, structure in enumerate(self._supercell['fractional_coordinates']):
            print(structure)




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

        disp = self._get_displacements(basis='F')

        adjusted_structure = self._adjust_network(disp)



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

        disp_super = np.zeros((len(self._cartesian_coordinate), 3, 8))
        rotation_axes = ['x', 'y', 'z']

        if rigid:
            nat_primitive = len(self._fractional_coordinate)
            if basis == 'F':
                pos_now = copy.deepcopy(self._fractional_coordinate)
            elif basis == 'C':
                pos_now = copy.deepcopy(self._cartesian_coordinate)

            pos_new = np.zeros((nat_primitive, 3))

            if rodrigues:

                disp_now = np.zeros((nat_primitive, 3))

                omegavec = self._distorsion_angles[:]
                norm_omegavec = np.sqrt(np.dot(omegavec, omegavec))
                normalized_omegavec = omegavec / norm_omegavec

                rotmat = self.get_rotation_matrix_around_axis(normalized_omegavec, norm_omegavec).transpose()

                for i in range(2):
                    pos_new[i, :] = pos_now[i, :]
                for i in range(3):
                    pos_new[i + 2, :] = np.dot(pos_new[i + 2, :], rotmat)

                disp_now[:, :] = pos_new - pos_now

                # how to handle this part?
                # for ishift, entry in enumerate(self.get_supercell()['shifts']):
                #     disp_super[:, :, ishift] += disp_now[:, :] \
                #                                 * math.cos(math.pi * np.dot(entry, self._kvecs[iax]))

            else:
                disp_now = np.zeros((nat_primitive, 3, 3))

                for iax, axis in enumerate(rotation_axes):
                    rotmat = self.get_rotation_matrix(axis,
                                                      self._distorsion_angles[iax]).transpose()
                    for i in range(2):
                        pos_new[i, :] = pos_now[i, :]
                    for i in range(3):
                        pos_new[i + 2, :] = np.dot(pos_now[i + 2, :], rotmat)
                    disp_now[:, :, iax] = pos_new - pos_now
                    pos_now[:, :] = pos_new[:, :]

                    for ishift, entry in enumerate(self.get_supercell()['shifts']):
                        disp_super[:, :, ishift] += disp_now[:, :, iax] \
                                                     * math.cos(math.pi * np.dot(entry, self._kvecs[iax]))

        else:
            for iax, axis in enumerate(rotation_axes):
                rotmat = self.get_rotation_matrix(axis, self._distorsion_angles[iax]).transpose()

                if basis == 'F':
                    pos_new = copy.deepcopy(self._fractional_coordinate)
                elif basis == 'C':
                    pos_new = copy.deepcopy(self._cartesian_coordinate)

                for i in range(3):
                    pos_new[i + 2, :] = np.dot(pos_new[i + 2, :], rotmat)

                if basis == 'F':
                    disp_primitive = pos_new - self._fractional_coordinate
                elif basis == 'C':
                    disp_primitive = pos_new - self._cartesian_coordinate

                for ishift, entry in enumerate(self.get_supercell()['shifts']):
                    disp_super[:, :, ishift] += disp_primitive \
                                                * math.cos(math.pi * np.dot(entry, self._kvecs[iax]))
        return disp_super

    def _adjust_network(self, disp_orig):

        for ishift, structure in enumerate(self._supercell['fractional_coordinates']):
            print(structure + 0.5 * disp_orig[:, :, ishift])

        return None

    def get_original_structure(self):
        return self._lattice_vector, \
               self._element_nums, \
               self._fractional_coordinate

    def get_supercell(self):
        return self._supercell


if __name__ == '__main__':
    obj = DistortPerovskite(elements=['La', 'Ni', 'O'], bond_length=2.0)
    # obj.rotate_single_octahedron(angles=[0, 0, 10])
    obj.rotate_octahedra(angles=[5, 5, 5], tilt_patterns=['+', '+', '-'])
    # supercell = obj._build_supercell()
    #print(obj.get_original_structure())
    #print(obj.get_supercell())
    # print(supercell)
