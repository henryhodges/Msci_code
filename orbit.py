"""
orbit.py
========
CR3BP-based Distant Retrograde Orbit (DRO) model for the Sun-Earth system.

Calculates the numerically integrated solution to the Circular Restricted
Three-Body Problem (CR3BP) in barycentric inertial coordinates and performs
a frame transformation to Earth-centred rotating coordinates for plotting.


Physical model
--------------
Equations of motion in barycentric inertial (non-dimensional) coordinates:

    q_ddot = -(1 - mu)(q - r_sun) / |q - r_sun|^3
             - mu * (q - r_earth) / |q - r_earth|^3

where mu = M_earth / (M_earth + M_sun), and both primaries orbit the
barycentre at unit angular velocity (one sidereal year = 2*pi nondim time).

Coordinate frames
-----------------
- Barycentric inertial  : origin at Sun-Earth barycentre, non-rotating.
                          Positions in AU, velocities in km/s.
- Earth-centred rotating: origin at Earth, x-axis pointing away from Sun,
                          Hill-scaled by gamma = mu^(1/3).
                          Used for visualisation and DRO identification.

References
----------
Henon, M. (1969). Numerical exploration of the restricted problem

Periozzi et al. (2017). Distant retrograde orbits and the asteroid hazard


Author : Henry Hodges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ------------------------------- Module logger ------------------------------ #

LOGGER = logging.getLogger("dro.orbit")


# ---------------------------- Physical constants ---------------------------- #

@dataclass(frozen=True)
class PhysicalConstants:
    """
    Physical and unit-conversion constants for the Sun-Earth CR3BP.

    All values are fixed at construction time (frozen dataclass) to
    guarantee reproducibility across a run.

    Attributes
    ----------
    AU_M : float
        IAU 2012 exact definition of the astronomical unit in metres.
    DAY_S : float
        Seconds per SI day.
    SIDEREAL_YEAR_DAYS : float
        Length of the sidereal year in days (IAU value).
    M_SUN_KG : float
        Solar mass in kg (IAU 2015 nominal).
    M_EARTH_KG : float
        Earth mass in kg (IAU 2015 nominal).
    """

    AU_M:               float = 149_597_870_700.0   # IAU 2012 exact, metres
    DAY_S:              float = 86_400.0
    SIDEREAL_YEAR_DAYS: float = 365.256_363_004      # IAU value
    M_SUN_KG:           float = 1.988_47e30          # IAU 2015 nominal
    M_EARTH_KG:         float = 5.972_2e24           # IAU 2015 nominal

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def AU_KM(self):
        """Astronomical unit in kilometres."""
        return self.AU_M / 1_000.0

    def sidereal_year_s(self):
        """Sidereal year in seconds."""
        return self.SIDEREAL_YEAR_DAYS * self.DAY_S

    def mu(self):
        """
        CR3BP mass parameter.

        mu = M_earth / (M_earth + M_sun)

        This is the non-dimensional mass of the secondary (Earth).
        For the Sun-Earth system mu ~ 3e-6.
        """
        return self.M_EARTH_KG / (self.M_EARTH_KG + self.M_SUN_KG)

    def gamma(self):
        """
        Hill scaling factor.

        gamma = mu^(1/3)

        Used to convert between Hill (non-dimensional) coordinates and
        AU-scaled Earth-centred rotating coordinates.
        """
        return self.mu() ** (1.0 / 3.0)

    def nondim_time_unit_s(self):
        """
        Non-dimensional time unit in seconds.

        2*pi non-dimensional time corresponds to one sidereal year, so:
            t_unit = T_sidereal / (2*pi)
        """
        return self.sidereal_year_s() / (2.0 * np.pi)

    def nondim_velocity_unit_kmps(self):
        """
        Non-dimensional velocity unit in km/s.

        v_unit = AU / t_unit  (converted to km/s)
        """
        return (self.AU_M / self.nondim_time_unit_s()) / 1_000.0


# ------------------------- Integrator configuration ------------------------- #

@dataclass(frozen=True)
class IntegratorConfig:
    """
    Configuration for the CR3BP numerical integrator.

    Attributes
    ----------
    rtol : float
        Relative tolerance for solve_ivp.
    atol : float
        Absolute tolerance for solve_ivp.
    method : str
        ODE solver method. DOP853 (Dormand-Prince 8th order) is recommended
        for high-accuracy conservative systems.
    n_eval : int
        Number of evenly-spaced evaluation points per period.
    r_min_stop : float
        Non-dimensional distance threshold for early termination if the
        trajectory approaches a primary too closely.
    """

    rtol:        float = 1e-12
    atol:        float = 1e-14
    method:      str   = "DOP853"
    n_eval:      int   = 20_000
    r_min_stop:  float = 1e-6


# ---------- CR3BP equations of motion (barycentric inertial frame) ---------- #

def _primaries_inertial(t: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Positions of Sun and Earth in barycentric inertial coordinates at time t.

    Both primaries orbit the barycentre at unit angular velocity.

    Parameters
    ----------
    t : float
        Non-dimensional time (radians; 2*pi = one sidereal year).
    mu : float
        CR3BP mass parameter (M_earth / (M_earth + M_sun)).

    Returns
    -------
    r_sun : np.ndarray, shape (2,)
        Sun position in AU (non-dimensional).
    r_earth : np.ndarray, shape (2,)
        Earth position in AU (non-dimensional).
    """
    ct, st = np.cos(t), np.sin(t)
    r_sun   = np.array([-mu * ct,        -mu * st       ], dtype=float)
    r_earth = np.array([(1.0 - mu) * ct, (1.0 - mu) * st], dtype=float)
    return r_sun, r_earth


def _potential_and_gradient(
    q: np.ndarray,
    t: float,
    mu: float,
):
    """
    Gravitational potential and its gradient at position q and time t.

    U = -(1-mu)/r_sun - mu/r_earth   (negative sign: attractive)

    Parameters
    ----------
    q : np.ndarray, shape (2,)
        Spacecraft position in barycentric inertial AU.
    t : float
        Non-dimensional time.
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    U : float
        Gravitational potential (non-dimensional).
    grad_U : np.ndarray, shape (2,)
        Gradient of U with respect to q.
    r_sun_dist : float
        Distance from spacecraft to Sun (AU, non-dimensional).
    r_earth_dist : float
        Distance from spacecraft to Earth (AU, non-dimensional).
    """
    r_sun, r_earth = _primaries_inertial(t, mu)
    d_s = q - r_sun
    d_e = q - r_earth

    rs = float(np.hypot(d_s[0], d_s[1]))
    re = float(np.hypot(d_e[0], d_e[1]))

    eps    = 1e-15                      # guard against exact coincidence
    rs_safe = max(rs, eps)
    re_safe = max(re, eps)

    U      = -(1.0 - mu) / rs_safe - mu / re_safe
    grad_U = ((1.0 - mu) * d_s / rs_safe**3
              + mu        * d_e / re_safe**3)

    return float(U), grad_U.astype(float), rs, re


def _rhs_inertial(
    t: float,
    s: np.ndarray,
    mu: float,
):
    """
    Right-hand side of the CR3BP equations of motion in barycentric inertial
    coordinates (Hamiltonian / canonical form):

        dq/dt = p
        dp/dt = -grad_U(q, t)

    Parameters
    ----------
    t : float
        Non-dimensional time.
    s : np.ndarray, shape (4,)
        State vector [q_x, q_y, p_x, p_y].
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    np.ndarray, shape (4,)
        Time derivative of the state vector.
    """
    q = s[0:2]
    p = s[2:4]
    _, grad_U, _, _ = _potential_and_gradient(q, t, mu)
    return np.concatenate([p, -grad_U])


def _event_close_approach(
    t: float,
    s: np.ndarray,
    mu: float,
    r_min: float,
):
    """
    Terminal event: fires when the spacecraft comes within r_min of either
    primary. Used to prevent integrator failure near singularities.
    """
    q = s[0:2]
    _, _, rs, re = _potential_and_gradient(q, t, mu)
    return min(rs, re) - r_min


# --------------------------- Frame transformation --------------------------- #

def _inertial_to_earth_centred_rotating(
    t:   np.ndarray,
    qx:  np.ndarray,
    qy:  np.ndarray,
    mu:  float,
) :
    """
    Transform barycentric inertial coordinates to Earth-centred rotating
    (AU-scaled) coordinates.

    The transformation is:
        [x; y]_Hill = (1/gamma) * R(-t) * [q_x; q_y] - (1/gamma) * [1-mu; 0]
        [x; y]_AU   = gamma * [x; y]_Hill

    where R(-t) is a clockwise rotation by t (undoes the primary rotation),
    and gamma = mu^(1/3) is the Hill scaling factor.

    In the output frame:
        - Earth is at (0, 0)
        - Sun is at (-1, 0) in AU
        - x-axis points from Sun through Earth (anti-sunward)

    Parameters
    ----------
    t : np.ndarray
        Non-dimensional time array.
    qx, qy : np.ndarray
        Barycentric inertial position components (AU, non-dimensional).
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    x_ec_au, y_ec_au : np.ndarray
        Earth-centred rotating position in AU.
    """
    gamma = mu ** (1.0 / 3.0)
    ct, st = np.cos(t), np.sin(t)

    # Rotate back to co-rotating frame
    rq_x =  ct * qx + st * qy
    rq_y = -st * qx + ct * qy

    # Shift origin to Earth, apply Hill scaling, then rescale to AU
    x_ec_au = gamma * ((rq_x - (1.0 - mu)) / gamma)
    y_ec_au = gamma * (rq_y / gamma)

    return x_ec_au, y_ec_au


# ----------------------------- L1 Lagrange point ---------------------------- #

def compute_l1_earth_centred_au(mu: float):
    """
    Compute the L1 Lagrange point position in Earth-centred AU plot
    coordinates (Earth at origin, Sun at x = -1 AU).

    L1 is found by solving the collinear equilibrium condition:
        dOmega/dx = 0,   y = 0

    in the standard CR3BP rotating barycentric frame, then shifting to
    the Earth-centred frame used for plotting.

    Parameters
    ----------
    mu : float
        CR3BP mass parameter.

    Returns
    -------
    x_l1 : float
        L1 x-coordinate in Earth-centred AU (negative, between Sun and Earth).
    y_l1 : float
        L1 y-coordinate (always 0.0 for collinear points).

    Notes
    -----
    In the standard rotating barycentric frame:
        Sun   at x = -mu
        Earth at x = 1 - mu
    L1 lies between the primaries at x in (0.5, 1-mu).
    """
    def dOmega_dx(x: float):
        r1 = abs(x + mu)             # distance to Sun
        r2 = abs(x - (1.0 - mu))    # distance to Earth
        return (x
                - (1.0 - mu) * (x + mu)       / r1**3
                - mu          * (x - (1.0 - mu)) / r2**3)

    x_l1_bary = brentq(
        dOmega_dx,
        a=0.5,
        b=(1.0 - mu) - 1e-10,
        maxiter=200,
        xtol=1e-15,
        rtol=1e-14,
    )

    # Shift to Earth-centred frame (Earth at 0, Sun at -1)
    x_l1 = x_l1_bary - (1.0 - mu)
    return float(x_l1), 0.0


# ------------------------ Core trajectory integrator ------------------------ #

def integrate_dro(
    q0_AU:     Tuple[float, float] = (0.8997, 0.0),
    v0_kmps:   Tuple[float, float] = (0.0, 32.9450),
    n_periods: int                 = 1,
    const:     PhysicalConstants   = None,
    cfg:       IntegratorConfig    = None,
):
    """
    Integrate a DRO trajectory in the Sun-Earth CR3BP.

    Initial conditions are specified in physical units (AU, km/s) and
    converted internally to non-dimensional CR3BP units before integration.

    Parameters
    ----------
    q0_AU : tuple of float
        Initial position (x, y) in AU, barycentric inertial frame.
        Default values are taken from Periozzi et al. (2017).
    v0_kmps : tuple of float
        Initial velocity (vx, vy) in km/s, barycentric inertial frame.
        Default values are taken from Periozzi et al. (2017).
    n_periods : int
        Number of orbital periods to integrate. One period = 2*pi
        non-dimensional time = one sidereal year.
    const : PhysicalConstants, optional
        Physical constants instance. Defaults to PhysicalConstants().
    cfg : IntegratorConfig, optional
        Integrator configuration. Defaults to IntegratorConfig().

    Returns
    -------
    dict with keys:
        't_nondim'    : np.ndarray  -- non-dimensional time
        't_days'      : np.ndarray  -- time in days from t=0
        'qx'          : np.ndarray  -- barycentric inertial x (AU)
        'qy'          : np.ndarray  -- barycentric inertial y (AU)
        'x_ec_au'     : np.ndarray  -- Earth-centred rotating x (AU)
        'y_ec_au'     : np.ndarray  -- Earth-centred rotating y (AU)
        'x_helio_km'  : np.ndarray  -- heliocentric x in km (for CrossoverFinder)
        'y_helio_km'  : np.ndarray  -- heliocentric y in km (for CrossoverFinder)
        'mu'          : float       -- CR3BP mass parameter used
        'gamma'       : float       -- Hill scaling factor used
        'const'       : PhysicalConstants
        'cfg'         : IntegratorConfig

    Raises
    ------
    RuntimeError
        If the ODE integrator fails to complete successfully.

    Notes
    -----
    The default ICs produce a DRO with an approximate radius of ~0.1 AU in
    the Earth-centred rotating frame. The tolerance DRO_TOL_AU in main.py
    should be set accordingly.
    """
    if const is None:
        const = PhysicalConstants()
    if cfg is None:
        cfg = IntegratorConfig()

    mu          = const.mu()
    v_unit_kmps = const.nondim_velocity_unit_kmps()
    AU_KM       = const.AU_KM()

    # Convert ICs to non-dimensional units
    q0 = np.array([q0_AU[0],              q0_AU[1]             ], dtype=float)
    p0 = np.array([v0_kmps[0]/v_unit_kmps, v0_kmps[1]/v_unit_kmps], dtype=float)
    s0 = np.concatenate([q0, p0])

    t0 = 0.0
    tf = 2.0 * np.pi * n_periods
    t_eval = np.linspace(t0, tf, cfg.n_eval * n_periods)

    LOGGER.info(
        "Integrating CR3BP DRO: t in [%.4f, %.4f] nondim, "
        "N=%d points, method=%s, rtol=%.1e",
        t0, tf, len(t_eval), cfg.method, cfg.rtol,
    )
    LOGGER.info(
        "  ICs: q0=(%.6f, %.6f) AU,  v0=(%.6f, %.6f) km/s",
        q0_AU[0], q0_AU[1], v0_kmps[0], v0_kmps[1],
    )

    # Build terminal event with the configured r_min
    def _close_approach_event(t, s):
        return _event_close_approach(t, s, mu, cfg.r_min_stop)

    _close_approach_event.terminal  = True
    _close_approach_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, s: _rhs_inertial(t, s, mu),
        t_span=(t0, tf),
        y0=s0,
        t_eval=t_eval,
        method=cfg.method,
        rtol=cfg.rtol,
        atol=cfg.atol,
        events=_close_approach_event,
    )

    if sol.t_events is not None and len(sol.t_events[0]) > 0:
        LOGGER.warning(
            "Integration stopped early: close approach to primary at "
            "t=%.6e nondim", sol.t_events[0][0],
        )

    if not sol.success:
        raise RuntimeError(f"CR3BP integration failed: {sol.message}")

    t    = sol.t
    qx   = sol.y[0]
    qy   = sol.y[1]

    # Transform to Earth-centred rotating frame (AU)
    x_ec_au, y_ec_au = _inertial_to_earth_centred_rotating(t, qx, qy, mu)

    # Heliocentric positions in km
    # In the barycentric inertial frame the Sun sits at r_sun = [-mu*cos(t), -mu*sin(t)]
    # The heliocentric position of the spacecraft is q - r_sun
    r_sun_x = -mu * np.cos(t)
    r_sun_y = -mu * np.sin(t)
    x_helio_km = (qx - r_sun_x) * AU_KM
    y_helio_km = (qy - r_sun_y) * AU_KM

    # Time in days
    t_days = t * const.nondim_time_unit_s() / const.DAY_S

    LOGGER.info(
        "Integration complete: %d points, "
        "min dist to Earth = %.4e AU, min dist to Sun = %.4e AU",
        len(t),
        float(np.min(np.hypot(x_ec_au, y_ec_au))),
        float(np.min(np.hypot(qx - r_sun_x, qy - r_sun_y))),
    )

    return {
        "t_nondim":   t,
        "t_days":     t_days,
        "qx":         qx,
        "qy":         qy,
        "x_ec_au":    x_ec_au,
        "y_ec_au":    y_ec_au,
        "x_helio_km": x_helio_km,
        "y_helio_km": y_helio_km,
        "mu":         mu,
        "gamma":      const.gamma(),
        "const":      const,
        "cfg":        cfg,
    }


# --------------------------- Constellation builder -------------------------- #

class DROConstellation:
    """
    Three-satellite DRO constellation in the Sun-Earth CR3BP.

    The constellation is generated by integrating a single CR3BP trajectory
    for three full periods and sampling it at t0, t0 + T/3, and t0 + 2T/3,
    where T = 2*pi (one sidereal year in non-dimensional time).

    This is the physically correct approach: in the CR3BP the DRO shape in
    the rotating frame is fixed, so phase offsets correspond to different
    starting times along the same trajectory -- NOT to rotations of a
    Keplerian ellipse.

    The output format of each satellite's "orbit" dict is designed to be a
    drop-in replacement for the old HenonDRO.generate_orbit() output used
    by CrossoverFinder in main.py.

    Attributes
    ----------
    satellites : list of dict
        Each entry has keys:
            "id"          : int   (1, 2, 3)
            "phase_offset": float (0.0, T/3, 2T/3 in nondim time)
            "orbit"       : dict  with keys matching generate_orbit() output:
                "times"      : np.ndarray  days from t=0
                "dro_helio"  : np.ndarray  shape (N, 2), heliocentric km
                "earth_helio": np.ndarray  shape (N, 2), Earth heliocentric km
                "dro_earth"  : np.ndarray  shape (N, 2), Earth-frame AU
                                           (kept for backward compatibility
                                            but units are AU not km here)

    Parameters
    ----------
    q0_AU : tuple of float
        Initial position in AU for the base trajectory.
    v0_kmps : tuple of float
        Initial velocity in km/s for the base trajectory.
    const : PhysicalConstants, optional
    cfg : IntegratorConfig, optional
    """

    def __init__(
        self,
        q0_AU:   Tuple[float, float] = (0.8997, 0.0),
        v0_kmps: Tuple[float, float] = (0.0, 32.9450),
        const:   PhysicalConstants   = None,
        cfg:     IntegratorConfig    = None,
    ):
        if const is None:
            const = PhysicalConstants()
        if cfg is None:
            cfg = IntegratorConfig()

        self.const = const
        self.cfg   = cfg

        LOGGER.info("Building DRO constellation (3 satellites, CR3BP)...")

        # Integrate for 3 periods so we can slice out each satellite's phase
        base = integrate_dro(
            q0_AU=q0_AU,
            v0_kmps=v0_kmps,
            n_periods=3,
            const=const,
            cfg=cfg,
        )

        T_nondim  = 2.0 * np.pi          # one period in nondim time
        AU_KM     = const.AU_KM()
        mu        = base["mu"]

        self.satellites = []

        for sat_id, phase_frac in enumerate([0.0, 1.0/3.0, 2.0/3.0], start=1):
            phase_offset = phase_frac * T_nondim

            # Find the index closest to the phase offset start
            idx_start = int(np.argmin(np.abs(base["t_nondim"] - phase_offset)))
            idx_end   = int(np.argmin(np.abs(base["t_nondim"] - (phase_offset + T_nondim))))

            t_slice        = base["t_nondim"][idx_start:idx_end]
            t_days_slice   = base["t_days"  ][idx_start:idx_end]
            x_hkm_slice    = base["x_helio_km"][idx_start:idx_end]
            y_hkm_slice    = base["y_helio_km"][idx_start:idx_end]
            x_ec_slice     = base["x_ec_au"   ][idx_start:idx_end]
            y_ec_slice     = base["y_ec_au"   ][idx_start:idx_end]
            qx_slice       = base["qx"][idx_start:idx_end]
            qy_slice       = base["qy"][idx_start:idx_end]

            # Earth heliocentric positions at each time step (km)
            # Earth barycentric inertial: r_earth = (1-mu)*[cos t, sin t]
            # Heliocentric: r_earth_helio = r_earth - r_sun
            #             = (1-mu)*[cos t, sin t] - (-mu)*[cos t, sin t]
            #             = [cos t, sin t]   (= 1 AU by construction)
            earth_x_km = np.cos(t_slice) * AU_KM
            earth_y_km = np.sin(t_slice) * AU_KM

            # Pack into the same dict structure CrossoverFinder expects
            orbit_dict = {
                # Primary keys used by CrossoverFinder.find_events()
                "times":       t_days_slice,           # days
                "dro_helio":   np.column_stack([x_hkm_slice, y_hkm_slice]),  # km
                "earth_helio": np.column_stack([earth_x_km,  earth_y_km ]),  # km

                # Earth-frame coords (AU) -- kept for plotting compatibility
                "dro_earth":   np.column_stack([x_ec_slice, y_ec_slice]),     # AU

                # Extra provenance fields
                "t_nondim":    t_slice,
                "qx":          qx_slice,
                "qy":          qy_slice,
            }

            self.satellites.append({
                "id":           sat_id,
                "phase_offset": phase_offset,
                "orbit":        orbit_dict,
            })

            LOGGER.info(
                "  Satellite %d: phase offset = %.4f nondim (%.1f deg equiv), "
                "%d trajectory points",
                sat_id,
                phase_offset,
                phase_frac * 360.0,
                len(t_days_slice),
            )

        LOGGER.info("DROConstellation ready.")