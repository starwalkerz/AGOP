import pytest
import numpy as np
from typing import NamedTuple

# Mocking the AGOP repository structures based on the tree specifications
class Alternative(str):
    pass

class GamePayoff(NamedTuple):
    U_DM: np.ndarray
    U_CP: np.ndarray

class AdmissibilityState(NamedTuple):
    R: list[Alternative]
    D_star: set[Alternative]
    convergence_depth: int
    LCB: dict[Alternative, float]
    Phi: dict[Alternative, float]
    adoption: dict[Alternative, int]
    K_derived: int = 3

# =====================================================================
# PATH 1: The Legacy Engine (agop_core.py)
# =====================================================================
def agop_legacy(problem, K: int) -> AdmissibilityState:
    """
    Simulates the legacy engine behavior. 
    Blindly accepts exogenous K and outputs a ranking even if poisoned.
    """
    V = problem["V"]
    # Simulated calculation that passes back whatever it computes, oblivious to tail-risk
    lcb_dict = {alt: float(problem["lcb_base"][alt]) for alt in V}
    phi_dict = {alt: float(problem["phi_base"][alt]) for alt in V}
    adoption_dict = {alt: 1 for alt in V}
    
    # Sort by phi descending
    ranking = sorted(V, key=lambda x: phi_dict[x], reverse=True)[:K]
    
    return AdmissibilityState(
        R=ranking,
        D_star=set(V),
        convergence_depth=1,
        LCB=lcb_dict,
        Phi=phi_dict,
        adoption=adoption_dict
    )

# =====================================================================
# PATH 2: The DbC Engine (agop_core_design_by_contract.py)
# =====================================================================
def verifies_theorem_5(func):
    """The formal Design-by-Contract Gatekeeper Decorator."""
    def wrapper(*args, **kwargs) -> AdmissibilityState:
        state: AdmissibilityState = func(*args, **kwargs)
        
        # Theorem 2 Invariant Check
        if state.convergence_depth != 1:
            raise AssertionError("Theorem 2 Violation: Fixed-point convergence depth mismatch.")
            
        # Theorem 5 Admissibility Check
        for alt in state.R:
            if state.LCB[alt] <= 0:
                raise AssertionError(f"Theorem 5 Violation: Alternative {alt} breached the non-negative LCB floor!")
            if state.adoption[alt] != 1:
                raise AssertionError(f"Theorem 5 Violation: Alternative {alt} lacks adoption status 1.")
            if state.Phi[alt] <= 0:
                raise AssertionError(f"Theorem 5 Violation: Alternative {alt} has non-positive composite utility.")
                
        return state
    return wrapper

@verifies_theorem_5
def agop_design_by_contract(problem, wl_classes: dict, nash_strategies: dict) -> AdmissibilityState:
    """
    Simulates the endogenous DbC engine behavior.
    Derives K from mathematical primitives and runs guarded validation.
    """
    V = problem["V"]
    
    # Primitive 1: Topological Diversity
    n_wl_classes = len(set(wl_classes.values()))
    
    # Primitive 2: Strategic Matrix Entropy
    sigma_dm = nash_strategies["sigma_DM"]
    sigma_cp = nash_strategies["sigma_CP"]
    
    h_dm = -float(np.sum(sigma_dm * np.log2(sigma_dm + 1e-12)))
    h_cp = -float(np.sum(sigma_cp * np.log2(sigma_cp + 1e-12)))
    
    # Endogenous Capacity Formula Calculation
    systemic_friction = 1.0 + (h_dm * h_cp)
    K_derived = max(1, int(np.floor(n_wl_classes / systemic_friction)))
    
    # Simulated tracking arrays from core execution
    lcb_dict = {alt: float(problem["lcb_base"][alt]) for alt in V}
    phi_dict = {alt: float(problem["phi_base"][alt]) for alt in V}
    adoption_dict = {alt: 1 if lcb_dict[alt] > 0 else 0 for alt in V}
    
    ranking = sorted(V, key=lambda x: phi_dict[x], reverse=True)[:K_derived]
    
    return AdmissibilityState(
        R=ranking,
        D_star=set(ranking),
        convergence_depth=1,
        LCB=lcb_dict,
        Phi=phi_dict,
        adoption=adoption_dict,
        K_derived=K_derived
    )

# =====================================================================
# UNIT TESTS
# =====================================================================

@pytest.fixture
def stable_market_setup():
    """Setup representing a clean, high-edge, low-volatility environment."""
    return {
        "problem": {
            "V": [Alternative("S_1"), Alternative("S_2"), Alternative("S_12")],
            "lcb_base": {"S_1": 23.95, "S_2": 30.86, "S_12": 26.93},
            "phi_base": {"S_1": 18.65, "S_2": 22.58, "S_12": 26.93}
        },
        "wl_classes": {"S_1": "B", "S_2": "A", "S_12": "C"},
        "nash_strategies": {
            "sigma_DM": np.array([1.0, 0.0]),  # Highly predictable pure strategy
            "sigma_CP": np.array([1.0, 0.0])
        }
    }

@pytest.fixture
def toxic_market_setup():
    """Setup representing a chaotic, front-run environment with tail-risk breaches."""
    return {
        "problem": {
            "V": [Alternative("S_1"), Alternative("S_2"), Alternative("S_12")],
            "lcb_base": {"S_1": -5.20, "S_2": -1.10, "S_12": 0.05},  # Severe tail-risk floor breaches
            "phi_base": {"S_1": 12.10, "S_2": 15.40, "S_12": 2.10}
        },
        "wl_classes": {"S_1": "B", "S_2": "B", "S_12": "B"},  # Topological structural collapse
        "nash_strategies": {
            "sigma_DM": np.array([0.5, 0.5]),  # Hyper-chaotic mixed strategic chaos
            "sigma_CP": np.array([0.5, 0.5])
        }
    }


def test_stable_regime_parallel_execution(stable_market_setup):
    """VERIFICATION: Both paths resolve successfully when conditions are safe."""
    prob = stable_market_setup["problem"]
    wl = stable_market_setup["wl_classes"]
    nash = stable_market_setup["nash_strategies"]
    
    # 1. Legacy Run
    legacy_state = agop_legacy(prob, K=3)
    assert len(legacy_state.R) == 3
    assert legacy_state.R[0] == "S_12"  # Highest Phi utility matches repo baseline
    
    # 2. DbC Run
    dbc_state = agop_design_by_contract(prob, wl, nash)
    assert dbc_state.K_derived == 3  # Low strategic entropy preserves full capacity
    assert dbc_state.R == legacy_state.R
    print("\n[PASS] Stable Regime: Both engines successfully execute the ranking.")


def test_toxic_regime_divergence(toxic_market_setup):
    """
    VERIFICATION: Demonstrates the critical paradigm divergence.
    The legacy engine blindly accepts poisoned inputs, exposing capital.
    The DbC engine drops an iron wall via AssertionError to prevent execution.
    """
    prob = toxic_market_setup["problem"]
    wl = toxic_market_setup["wl_classes"]
    nash = toxic_market_setup["nash_strategies"]
    
    # 1. The Legacy Hazard
    legacy_state = agop_legacy(prob, K=3)
    # The legacy bot outputs a ranking oblivious to negative LCB bounds
    assert len(legacy_state.R) == 3
    assert legacy_state.LCB[legacy_state.R[0]] < 0  # CRITICAL EXPOSITION RISK Bypassed!
    
    # 2. The DbC Guardian Circuit Breaker
    # The mathematical engine shrinks capacity and traps the tail risk flight
    with pytest.raises(AssertionError) as exc_info:
        agop_design_by_contract(prob, wl, nash)
        
    assert "Theorem 5 Violation" in str(exc_info.value)
    print("\n[PASS] Toxic Regime: Legacy engine failed silently (exposed capital).")
    print(f"[PASS] Toxic Regime: DbC engine trapped execution safely via: {exc_info.value}")
