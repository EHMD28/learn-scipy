from scipy.integrate import solve_ivp
from scipy.constants import pi
import numpy as np

KAPPA = 1.0
GAMMA = 2.0


def eos_epsilon(p):
    return (p / KAPPA) ** (1 / GAMMA)


def tov_rhs(r, state):
    p, m = state
    if p <= 0:
        return [0, 0]
    epsilon = eos_epsilon(p)
    f1 = -((m * epsilon) / r**2)
    f2 = 1 + (p / epsilon)
    f3 = 1 + (4 * pi * r**3 * p / m)
    f4 = 1 / (1 - (2 * m / r))
    dp_dr = f1 * f2 * f3 * f4
    dm_dr = (4 * pi) * r**2 * epsilon
    return [dp_dr, dm_dr]


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
    return r_solutions[-1], m_solutions[-1]


def main():
    p_c = 100
    r, m = solve_tov(p_c)
    print(f"{r=} {m=}")


if __name__ == "__main__":
    main()
