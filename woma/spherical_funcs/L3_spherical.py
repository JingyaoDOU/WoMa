"""
WoMa 3 layer spherical functions 
"""

import numpy as np
from numba import njit
import sys
import warnings

warnings.filterwarnings("ignore")

from woma.misc import glob_vars as gv
from woma.misc import utils
from woma.spherical_funcs import L2_spherical
from woma.eos import eos
from woma.eos.T_rho import T_rho, set_T_rho_args


@njit
def L3_integrate(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
):
    """ Integration of a 2 layer spherical planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            R2 (float):
                Boundary between layers 2 and 3 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.

        Returns:
            A1_r ([float]):
                Array of radii (SI).

            A1_m_enc ([float]):
                Array of cumulative mass (SI).

            A1_P ([float]):
                Array of pressures (SI).

            A1_T ([float]):
                Array of temperatures (SI).

            A1_rho ([float]):
                Array of densities (SI).

            A1_u ([float]):
                Array of internal energy (SI).

            A1_mat_id ([float]):
                Array of material ids (SI).
    """
    A1_r = np.linspace(R, 0, int(num_prof))
    A1_m_enc = np.zeros(A1_r.shape)
    A1_P = np.zeros(A1_r.shape)
    A1_T = np.zeros(A1_r.shape)
    A1_rho = np.zeros(A1_r.shape)
    A1_u = np.zeros(A1_r.shape)
    A1_mat_id = np.zeros(A1_r.shape)

    u_s = eos.u_rho_T(rho_s, T_s, mat_id_L3)
    T_rho_args_L3 = set_T_rho_args(
        T_s, rho_s, T_rho_type_id_L3, T_rho_args_L3, mat_id_L3
    )

    dr = A1_r[0] - A1_r[1]

    A1_m_enc[0] = M
    A1_P[0] = P_s
    A1_T[0] = T_s
    A1_rho[0] = rho_s
    A1_u[0] = u_s
    A1_mat_id[0] = mat_id_L3

    for i in range(1, A1_r.shape[0]):
        # Layer 3
        if A1_r[i] > R2:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L3
            T_rho_type_id = T_rho_type_id_L3
            T_rho_args = T_rho_args_L3
            rho0 = rho
        # Layer 2, 3 boundary
        elif A1_r[i] <= R2 and A1_r[i - 1] > R2:
            rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L2)
            T_rho_args_L2 = set_T_rho_args(
                A1_T[i - 1], rho, T_rho_type_id_L2, T_rho_args_L2, mat_id_L2
            )
            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = A1_rho[i - 1]
        # Layer 2
        elif A1_r[i] > R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L2
            T_rho_type_id = T_rho_type_id_L2
            T_rho_args = T_rho_args_L2
            rho0 = rho
        # Layer 1, 2 boundary
        elif A1_r[i] <= R1 and A1_r[i - 1] > R1:
            rho = eos.rho_P_T(A1_P[i - 1], A1_T[i - 1], mat_id_L1)
            T_rho_args_L1 = set_T_rho_args(
                A1_T[i - 1], rho, T_rho_type_id_L1, T_rho_args_L1, mat_id_L1
            )
            mat_id = mat_id_L1
            T_rho_type_id = T_rho_type_id_L1
            T_rho_args = T_rho_args_L1
            rho0 = A1_rho[i - 1]
        # Layer 1
        elif A1_r[i] <= R1:
            rho = A1_rho[i - 1]
            mat_id = mat_id_L1
            T_rho_type_id = T_rho_type_id_L1
            T_rho_args = T_rho_args_L1
            rho0 = A1_rho[i - 1]

        A1_m_enc[i] = A1_m_enc[i - 1] - 4 * np.pi * A1_r[i - 1] ** 2 * rho * dr
        A1_P[i] = A1_P[i - 1] + gv.G * A1_m_enc[i - 1] * rho / (A1_r[i - 1] ** 2) * dr
        A1_rho[i] = eos.find_rho(
            A1_P[i], mat_id, T_rho_type_id, T_rho_args, rho0, 1.1 * rho
        )
        A1_T[i] = T_rho(A1_rho[i], T_rho_type_id, T_rho_args, mat_id)
        A1_u[i] = eos.u_rho_T(A1_rho[i], A1_T[i], mat_id)
        A1_mat_id[i] = mat_id
        # Update the T-rho parameters
        if mat_id == gv.id_HM80_HHe and T_rho_type_id == gv.type_adb:
            T_rho_args = set_T_rho_args(
                A1_T[i], A1_rho[i], T_rho_type_id, T_rho_args, mat_id
            )

        if A1_m_enc[i] < 0:
            break

    return A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id


def L3_find_M_given_R_R1_R2(
    num_prof,
    R,
    M_max,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """ Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            R2 (float):
                Boundary between layers 2 and 3 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive masses.
                
            verbosity (int):
                Printing options.

        Returns:

            M_max ([float]):
                Mass of the planet (SI).

    """
    min_tol = 1e-7
    if tol > min_tol:
        tol = min_tol

    M_min = 0.0

    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
        num_prof,
        R,
        M_max,
        P_s,
        T_s,
        rho_s,
        R1,
        R2,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
    )

    if A1_m_enc[-1] < 0:
        raise ValueError(
            "M_max is too low, ran out of mass in first iteration.\nPlease increase M_max.\n"
        )

    for i in range(num_attempt):

        M_try = (M_min + M_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M_try,
            P_s,
            T_s,
            rho_s,
            R1,
            R2,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            M_max = M_try
        else:
            M_min = M_try

        tol_reached = np.abs(M_min - M_max) / M_min

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): M=%.5gM_E --> tol=%.2g(%.2g)"
                % (i, num_attempt, M_try / gv.M_earth, tol_reached, tol),
                end="",
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    return M_max


def L3_find_R_given_M_R1_R2(
    num_prof,
    R_max,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """ Finder of the total mass of the planet.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R_max (float):
                Maximum radius of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            R2 (float):
                Boundary between layers 2 and 3 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive masses or radius.
                
            verbosity (int):
                Printing options.


        Returns:

            M_max ([float]):
                Mass of the planet (SI).

    """
    if R1 > R2:
        if verbosity >= 1:
            print("R1 should not be greater than R2")
        return -1  ###is this an error? if so then make it an error!

    R_min = R2

    rho_s_L2 = eos.rho_P_T(P_s, T_s, mat_id_L2)

    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L2_spherical.L2_integrate(
        num_prof,
        R2,
        M,
        P_s,
        T_s,
        rho_s_L2,
        R1,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
    )

    if A1_m_enc[-1] == 0:
        e = (
            "Ran out of mass for a 2 layer planet.\n"
            + "Try increase the mass or reduce R1.\n"
        )
        raise ValueError(e)

    A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
        num_prof,
        R_max,
        M,
        P_s,
        T_s,
        rho_s,
        R1,
        R2,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
    )

    if A1_m_enc[-1] > 0:
        e = (
            "Excess of mass for a 3 layer planet with R = R_max.\n"
            + "Try reduce the mass or increase R_max.\n"
        )
        raise ValueError(e)

    for i in range(num_attempt):
        R_try = (R_min + R_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R_try,
            M,
            P_s,
            T_s,
            rho_s,
            R1,
            R2,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            R_min = R_try
        else:
            R_max = R_try

        tol_reached = np.abs(R_min - R_max) / R_max

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R=%.5gR_E --> tol=%.2g(%.2g)"
                % (i, num_attempt, R_try / gv.R_earth, tol_reached, tol),
                end="",
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    return R_min


def L3_find_R2_given_M_R_R1(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R1,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """ Finder of the boundary between layers 2 and 3 of the planet for
        fixed boundary between layers 1 and 2.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R1 (float):
                Boundary between layers 1 and 2 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive masses or radius.
                
            verbosity (int):
                Printing options.

        Returns:
            R2_max ([float]):
                Boundary between layers 2 and 3 of the planet (SI).
    """
    R2_min = R1
    R2_max = R

    for i in range(num_attempt):

        R2_try = (R2_min + R2_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1,
            R2_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            R2_min = R2_try
        else:
            R2_max = R2_try

        tol_reached = np.abs(R2_min - R2_max) / R2_max

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R2=%.5gR_E --> tol=%.2g(%.2g)"
                % (i, num_attempt, R2_try / gv.R_earth, tol_reached, tol),
                end="",
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    assert (
        R2_max / R > R1 / R + 2 * tol
    ), "Boundary not found, decrease R1 or increase M"
    assert R2_max / R < R / R - 2 * tol, "Boundary not found, increase R1 or decrease M"

    return R2_max


def L3_find_R1_given_M_R_R2(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    R2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    num_attempt=40,
    tol=0.01,
    verbosity=1,
):
    """ Finder of the boundary between layers 2 and 3 of the planet for
        fixed boundary between layers 1 and 2.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            R2 (float):
                Boundary between layers 2 and 3 (SI).

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            num_attempt (float):
                Maximum number of iterations to perform.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive masses or radius.
                
            verbosity (int):
                Printing options.


        Returns:
            R2_max ([float]):
                Boundary between layers 2 and 3 of the planet (SI).
    """
    R1_min = 0.0
    R1_max = R2

    for i in range(num_attempt):

        R1_try = (R1_min + R1_max) * 0.5

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            R2,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        if A1_m_enc[-1] > 0.0:
            R1_min = R1_try
        else:
            R1_max = R1_try

        tol_reached = np.abs(R1_min - R1_max) / R1_max

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R1=%.5gR_E --> tol=%.2g(%.2g)"
                % (i, num_attempt, R1_try / gv.R_earth, tol_reached, tol),
                end="",
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    assert R1_max / R > 2 * tol, "Boundary not found, decrease R2 or increase M"
    assert (
        R1_max / R < R2 / R - 2 * tol
    ), "Boundary not found, increase R2 or decrease M"

    return R1_max


def L3_find_R1_R2_given_M_R_I(
    num_prof,
    R,
    M,
    P_s,
    T_s,
    rho_s,
    I_MR2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    R1_min=None,
    R1_max=None,
    num_attempt=40,
    num_attempt_2=40,
    tol=0.01,
    verbosity=1,
):
    """ Finder of the boundaries of the planet for a
        fixed moment of inertia factor.
        The correct value yields A1_m_enc -> 0 at the center of the planet.

        Args:
            num_prof (int):
                Number of profile integration steps.

            R (float):
                Radii of the planet (SI).

            M (float):
                Mass of the planet (SI).

            P_s (float):
                Pressure at the surface (SI).

            T_s (float):
                Temperature at the surface (SI).

            rho_s (float):
                Density at the surface (SI).

            I_MR2 (float):
                Moment of inertia factor.

            mat_id_L1 (int):
                Material id for layer 1.

            T_rho_type_id_L1 (int)
                Relation between A1_T and A1_rho to be used in layer 1.

            T_rho_args_L1 (list):
                Extra arguments to determine the relation in layer 1.

            mat_id_L2 (int):
                Material id for layer 2.

            T_rho_type_id_L2 (int)
                Relation between A1_T and A1_rho to be used in layer 2.

            T_rho_args_L2 (list):
                Extra arguments to determine the relation in layer 2.

            mat_id_L3 (int):
                Material id for layer 3.

            T_rho_type_id_L3 (int)
                Relation between A1_T and A1_rho to be used in layer 3.

            T_rho_args_L3 (list):
                Extra arguments to determine the relation in layer 3.
                
            R1_min (float):
                Minimum core-mantle boundary to consider (m).
                
            R1_max (float):
                Maximum core-mantle boundary to consider (m).
                
            num_attempt (float):
                Maximum number of iterations to perform. Outer loop.
                
            num_attempt_2 (float):
                Maximum number of iterations to perform. Inner loop.
                
            tol (float):
                Tolerance level. Relative difference between two consecutive masses or radius.
                
            verbosity (int):
                Printing options.

        Returns:
            R1, R2 ([float]):
                Boundaries between layers 1 and 2 and between layers 2 and 3 of
                the planet (SI).
    """
    # Normalisation for moment of inertia factor
    MR2 = M * R ** 2

    try:
        if verbosity >= 1:
            print("Creating a planet with R1_min")
        R2_min = L3_find_R2_given_M_R_R1(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_min,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
            num_attempt=num_attempt,
            tol=tol,
            verbosity=verbosity,
        )

    except:
        raise ValueError("Could not build a planet with R1_min.")

    try:
        if verbosity >= 1:
            print("Creating a planet with R1_max")
        R2_max = L3_find_R2_given_M_R_R1(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_max,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
            num_attempt=num_attempt,
            tol=tol,
            verbosity=verbosity,
        )

    except:
        raise ValueError("Could not build a planet with R1_max.")

    (
        A1_r_min,
        A1_m_enc_min,
        A1_P_min,
        A1_T_min,
        A1_rho_min,
        A1_u_min,
        A1_mat_id_min,
    ) = L3_integrate(
        num_prof,
        R,
        M,
        P_s,
        T_s,
        rho_s,
        R1_min,
        R2_min,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
    )

    I_MR2_R1_min = utils.moi(A1_r_min, A1_rho_min) / MR2

    (
        A1_r_max,
        A1_m_enc_max,
        A1_P_max,
        A1_T_max,
        A1_rho_max,
        A1_u_max,
        A1_mat_id_max,
    ) = L3_integrate(
        num_prof,
        R,
        M,
        P_s,
        T_s,
        rho_s,
        R1_max,
        R2_max,
        mat_id_L1,
        T_rho_type_id_L1,
        T_rho_args_L1,
        mat_id_L2,
        T_rho_type_id_L2,
        T_rho_args_L2,
        mat_id_L3,
        T_rho_type_id_L3,
        T_rho_args_L3,
    )

    I_MR2_R1_max = utils.moi(A1_r_max, A1_rho_max) / MR2

    # compute I_MR2_min and _max
    I_MR2_min = np.min([I_MR2_R1_min, I_MR2_R1_max])
    I_MR2_max = np.max([I_MR2_R1_min, I_MR2_R1_max])

    if verbosity >= 1:
        print("Minimum moment of inertia factor found: {:.3f}".format(I_MR2_min))
        print("Maximum moment of inertia factor found: {:.3f}".format(I_MR2_max))

    if I_MR2 < I_MR2_min or I_MR2_max < I_MR2:
        e = (
            "I_MR2 outside the values found for R1_min and R1_max.\n"
            + "Try modifying R1_min, R1_max or I_MR2."
        )

        raise ValueError(e)

    for i in range(num_attempt):
        R1_try = (R1_min + R1_max) * 0.5

        R2_try = L3_find_R2_given_M_R_R1(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
            num_attempt_2,
            tol,
            verbosity=0,
        )

        A1_r, A1_m_enc, A1_P, A1_T, A1_rho, A1_u, A1_mat_id = L3_integrate(
            num_prof,
            R,
            M,
            P_s,
            T_s,
            rho_s,
            R1_try,
            R2_try,
            mat_id_L1,
            T_rho_type_id_L1,
            T_rho_args_L1,
            mat_id_L2,
            T_rho_type_id_L2,
            T_rho_args_L2,
            mat_id_L3,
            T_rho_type_id_L3,
            T_rho_args_L3,
        )

        I_MR2_iter = utils.moi(A1_r, A1_rho) / MR2

        if I_MR2_R1_min == I_MR2_max:
            if I_MR2_iter < I_MR2:
                R1_max = R1_try
            else:
                R1_min = R1_try

        elif I_MR2_R1_min == I_MR2_min:
            if I_MR2_iter > I_MR2:
                R1_max = R1_try
            else:
                R1_min = R1_try

        else:
            raise ValueError("Something unexpected occured. Please report it!")

        tol_reached = np.abs(I_MR2_iter - I_MR2) / I_MR2

        # Print progress
        if verbosity >= 1:
            print(
                "\rIter %d(%d): R1=%.5gR_E R2=%.5gR_E --> tol=%.2g(%.2g)"
                % (
                    i,
                    num_attempt,
                    R1_try / gv.R_earth,
                    R2_try / gv.R_earth,
                    tol_reached,
                    tol,
                ),
                end="",
            )

        if tol_reached < tol:
            if verbosity >= 1:
                print("")
            break

    return R1_try, R2_try
