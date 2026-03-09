from scipy.integrate import solve_ivp
from scipy.constants import pi
import numpy as np

# Conversion factor to convert from MeV/fm^3 to M☉/km^3
MEV_PER_FM3_TO_SM_PER_KM_3 = 8.96498313e-7
EPSILON_0_MEV_PER_FM3 = 150
EPSILON_0 = EPSILON_0_MEV_PER_FM3 * MEV_PER_FM3_TO_SM_PER_KM_3  # M☉/km^3
# G_NU = 1.3271244002e11  # km^3 / (M☉ * s^2)
G_NU = 1.47662504  # km / M☉
a = 1 / np.sqrt(G_NU * EPSILON_0)
b = 1 / np.sqrt(G_NU**3 * EPSILON_0)
KAPPA = 4.80307467e-5  # Units to cancel out ε^γ to MeV/fm^3.
# GAMMA = 2.048315956
GAMMA = 2.4
KAPPA_PRIME = KAPPA * (EPSILON_0_MEV_PER_FM3 ** (GAMMA - 1))


def mass_nu(m_prime):
    return b * m_prime


def radius_nu(r_prime):
    return a * r_prime


def eos_epsilon(p_prime):
    return (p_prime / KAPPA_PRIME) ** (1 / GAMMA)


def tov_rhs(r, state):
    p, m = state
    if p <= 0:
        return (0, 0)
    epsilon = eos_epsilon(p)
    f1 = -((m * epsilon) / r**2)
    f2 = 1 + (p / epsilon)
    f3 = 1 + (4 * pi * r**3 * p / m)
    f4 = 1 / (1 - (2 * m / r))
    dp_dr = f1 * f2 * f3 * f4
    dm_dr = (4 * pi) * r**2 * epsilon
    return (dp_dr, dm_dr)


def surface_event(r, state):
    p, _ = state
    return p


surface_event.terminal = True  # type: ignore
surface_event.direction = -1  # type: ignore


def solve_tov(p_c) -> tuple[float, float]:
    r_0 = 1e-5
    epsilon = eos_epsilon(p_c)
    m_0 = (4 * pi / 3) * r_0**3 * epsilon
    solutions = solve_ivp(
        tov_rhs,
        t_span=(r_0, 50),  # Should terminate before reaching endpoint.
        y0=(p_c, m_0),
        events=surface_event,
    )
    r_solutions = solutions.t
    m_solutions = solutions.y[1]
    return (r_solutions[-1], m_solutions[-1])


def main():
    p_c = 100 * MEV_PER_FM3_TO_SM_PER_KM_3
    p_prime = p_c / EPSILON_0
    r, m = solve_tov(p_prime)
    r = radius_nu(r)
    m = mass_nu(m)
    print(f"{r=} {m=}")


if __name__ == "__main__":
    main()
