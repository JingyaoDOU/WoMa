"""
WoMa 1 layer spinning functions 
"""

import numpy as np
from numba import njit

from woma.spin_funcs import L1_spin
from woma.spin_funcs import utils_spin as us
from woma.eos import eos
from woma.eos.T_rho import T_rho


@njit
def L3_rho_eq_po_from_V(
    A1_r_eq,
    A1_V_eq,
    A1_r_po,
    A1_V_po,
    P_0,
    P_1,
    P_2,
    P_s,
    rho_0,
    rho_s,
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
    """ Compute densities of equatorial and polar profiles given the potential
        for a 3 layer planet.

    Parameters
    ----------
    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_V_eq : [float]
        Equatorial profile of potential (SI).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_V_po : [float]
        Polar profile of potential (SI).

    P_0 : float
        Pressure at the center of the planet (SI).

    P_1 : float
        Pressure at the boundary between layers 1 and 2 of the planet (SI).

    P_2 : float
        Pressure at the boundary between layers 2 and 3 of the planet (SI).

    P_s : float
        Pressure at the surface of the planet (SI).

    rho_0 : float
        Density at the center of the planet (SI).

    rho_s : float
        Density at the surface of the planet (SI).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between T and rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between T and rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between T and rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.


    Returns
    -------
    A1_rho_eq : [float]
        Equatorial profile of densities (SI).

    A1_rho_po : [float]
        Polar profile of densities (SI).
    """

    A1_P_eq = np.zeros(A1_V_eq.shape[0])
    A1_P_po = np.zeros(A1_V_po.shape[0])
    A1_rho_eq = np.zeros(A1_V_eq.shape[0])
    A1_rho_po = np.zeros(A1_V_po.shape[0])

    A1_P_eq[0] = P_0
    A1_P_po[0] = P_0
    A1_rho_eq[0] = rho_0
    A1_rho_po[0] = rho_0

    # equatorial profile
    for i in range(A1_r_eq.shape[0] - 1):
        gradV = A1_V_eq[i + 1] - A1_V_eq[i]
        gradP = -A1_rho_eq[i] * gradV
        A1_P_eq[i + 1] = A1_P_eq[i] + gradP

        # avoid overspin
        if A1_P_eq[i + 1] > A1_P_eq[i]:
            A1_rho_eq[i + 1 :] = A1_rho_eq[i]
            break

        # compute density
        if A1_P_eq[i + 1] >= P_s and A1_P_eq[i + 1] >= P_1:
            A1_rho_eq[i + 1] = eos.find_rho(
                A1_P_eq[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_eq[i],
            )

        elif A1_P_eq[i + 1] >= P_s and A1_P_eq[i + 1] >= P_2:
            A1_rho_eq[i + 1] = eos.find_rho(
                A1_P_eq[i + 1],
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                rho_s * 0.1,
                A1_rho_eq[i],
            )

        elif A1_P_eq[i + 1] >= P_s:
            A1_rho_eq[i + 1] = eos.find_rho(
                A1_P_eq[i + 1],
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                rho_s * 0.1,
                A1_rho_eq[i],
            )

        else:
            A1_rho_eq[i + 1] = 0.0
            break

    # polar profile
    for i in range(A1_r_po.shape[0] - 1):
        gradV = A1_V_po[i + 1] - A1_V_po[i]
        gradP = -A1_rho_po[i] * gradV
        A1_P_po[i + 1] = A1_P_po[i] + gradP

        # avoid overspin
        if A1_P_eq[i + 1] > A1_P_eq[i]:
            A1_rho_eq[i + 1 :] = A1_rho_eq[i]
            break

        # compute density
        if A1_P_po[i + 1] >= P_s and A1_P_po[i + 1] >= P_1:
            A1_rho_po[i + 1] = eos.find_rho(
                A1_P_po[i + 1],
                mat_id_L1,
                T_rho_type_id_L1,
                T_rho_args_L1,
                rho_s * 0.1,
                A1_rho_po[i],
            )

        elif A1_P_po[i + 1] >= P_s and A1_P_po[i + 1] >= P_2:
            A1_rho_po[i + 1] = eos.find_rho(
                A1_P_po[i + 1],
                mat_id_L2,
                T_rho_type_id_L2,
                T_rho_args_L2,
                rho_s * 0.1,
                A1_rho_po[i],
            )

        elif A1_P_po[i + 1] >= P_s:
            A1_rho_po[i + 1] = eos.find_rho(
                A1_P_po[i + 1],
                mat_id_L3,
                T_rho_type_id_L3,
                T_rho_args_L3,
                rho_s * 0.1,
                A1_rho_po[i],
            )

        else:
            A1_rho_po[i + 1] = 0.0
            break

    return A1_rho_eq, A1_rho_po


def L3_spin(
    num_attempt,
    A1_r_eq,
    A1_rho_eq,
    A1_r_po,
    A1_rho_po,
    period,
    P_0,
    P_1,
    P_2,
    P_s,
    rho_0,
    rho_s,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    verbosity=1,
):
    """ Compute spining profile of densities for a 3 layer planet.

    Parameters
    ----------
    num_attempt : int
        Number of num_attempt to run.

    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (SI).

    radii : [float]
        Radii of the spherical profile (SI).

    densities : [float]
        Densities of the spherical profile (SI).

    period : float
        Period of the planet (hours).

    P_0 : float
        Pressure at the center of the planet (SI).

    P_1 : float
        Pressure at the boundary between layers 1 and 2 of the planet (SI).

    P_2 : float
        Pressure at the boundary between layers 2 and 3 of the planet (SI).

    P_s : float
        Pressure at the surface of the planet (SI).

    rho_0 : float
        Density at the center of the planet (SI).

    rho_s : float
        Density at the surface of the planet (SI).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between T and rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.

    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between T and rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.

    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between T and rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    Returns
    -------
    profile_eq ([[float]]):
        List of the num_attempt of the equatorial density profile (SI).

    profile_po ([[float]]):
        List of the num_attempt of the polar density profile (SI).

    """

    profile_eq = []
    profile_po = []

    profile_eq.append(A1_rho_eq)
    profile_po.append(A1_rho_po)

    for i in range(num_attempt):
        A1_V_eq, A1_V_po = L1_spin.V_eq_po_from_rho(
            A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po, period
        )
        A1_rho_eq, A1_rho_po = L3_rho_eq_po_from_V(
            A1_r_eq,
            A1_V_eq,
            A1_r_po,
            A1_V_po,
            P_0,
            P_1,
            P_2,
            P_s,
            rho_0,
            rho_s,
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
        profile_eq.append(A1_rho_eq)
        profile_po.append(A1_rho_po)

    return profile_eq, profile_po


def L3_place_particles(
    A1_r_eq,
    A1_rho_eq,
    A1_r_po,
    A1_rho_po,
    period,
    N,
    rho_1,
    rho_2,
    mat_id_L1,
    T_rho_type_id_L1,
    T_rho_args_L1,
    mat_id_L2,
    T_rho_type_id_L2,
    T_rho_args_L2,
    mat_id_L3,
    T_rho_type_id_L3,
    T_rho_args_L3,
    N_ngb=48,
    verbosity=1,
):
    """ Particle placement for 3 layer spinning planet profile.
    
    Parameters
    ----------
    A1_r_eq : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_eq : [float]
        Equatorial profile of densities (SI).

    A1_r_po : [float]
        Points at equatorial profile where the solution is defined (SI).

    A1_rho_po : [float]
        Polar profile of densities (SI).

    period : float
        Period of the planet (hours).

    N : int
        Number of particles.
        
    rho_1 : float
        Density at the boundary between layers 1 and 2 (SI).

    rho_2 : float
        Density at the boundary between layers 2 and 3 (SI).

    mat_id_L1 : int
        Material id for layer 1.

    T_rho_type_id_L1 : int
        Relation between T and rho to be used in layer 1.

    T_rho_args_L1 : [float]
        Extra arguments to determine the relation in layer 1.
        
    mat_id_L2 : int
        Material id for layer 2.

    T_rho_type_id_L2 : int
        Relation between T and rho to be used in layer 2.

    T_rho_args_L2 : [float]
        Extra arguments to determine the relation in layer 2.
        
    mat_id_L3 : int
        Material id for layer 3.

    T_rho_type_id_L3 : int
        Relation between T and rho to be used in layer 3.

    T_rho_args_L3 : [float]
        Extra arguments to determine the relation in layer 3.

    N_ngb : int
        Number of neighbors in the SPH simulation.
        
    Returns
    -------
    A1_x : [float]
        Position x of each particle (SI).

    A1_y : [float]
        Position y of each particle (SI).

    A1_z : [float]
        Position z of each particle (SI).

    A1_vx : [float]
        Velocity in x of each particle (SI).

    A1_vy : [float]
        Velocity in y of each particle (SI).

    A1_vz : [float]
        Velocity in z of each particle (SI).

    A1_m : [float]
        Mass of every particle (SI).

    A1_rho : [float]
        Density for every particle (SI).
        
    A1_u : [float]
        Internal energy for every particle (SI).

    A1_P : [float]
        Pressure for every particle (SI).
        
    A1_h : [float]
        Smoothing lenght for every particle (SI).

    A1_mat_id : [int]
        Material id for every particle.

    A1_id : [int]
        Identifier for every particle
        
    """
    (
        A1_x,
        A1_y,
        A1_z,
        A1_vx,
        A1_vy,
        A1_vz,
        A1_m,
        A1_rho,
        A1_R,
        A1_Z,
    ) = us.place_particles(A1_r_eq, A1_rho_eq, A1_r_po, A1_rho_po, N, period, verbosity)

    # internal energy
    A1_u = np.zeros((A1_m.shape[0]))

    A1_P = np.zeros((A1_m.shape[0],))

    for k in range(A1_m.shape[0]):
        if A1_rho[k] > rho_1:
            T = T_rho(A1_rho[k], T_rho_type_id_L1, T_rho_args_L1, mat_id_L1)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L1)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L1)

        elif A1_rho[k] > rho_2:
            T = T_rho(A1_rho[k], T_rho_type_id_L2, T_rho_args_L2, mat_id_L2)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L2)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L2)

        else:
            T = T_rho(A1_rho[k], T_rho_type_id_L3, T_rho_args_L3, mat_id_L3)
            A1_u[k] = eos.u_rho_T(A1_rho[k], T, mat_id_L3)
            A1_P[k] = eos.P_u_rho(A1_u[k], A1_rho[k], mat_id_L3)

    # Smoothing lengths, crudely estimated from the densities
    w_edge = 2  # r/h at which the kernel goes to zero
    A1_h = np.cbrt(N_ngb * A1_m / (4 / 3 * np.pi * A1_rho)) / w_edge

    A1_id = np.arange(A1_m.shape[0])
    A1_mat_id = (
        (A1_rho > rho_1) * mat_id_L1
        + np.logical_and(A1_rho <= rho_1, A1_rho > rho_2) * mat_id_L2
        + (A1_rho < rho_2) * mat_id_L3
    )

    return (
        A1_x,
        A1_y,
        A1_z,
        A1_vx,
        A1_vy,
        A1_vz,
        A1_m,
        A1_rho,
        A1_u,
        A1_P,
        A1_h,
        A1_mat_id,
        A1_id,
    )
