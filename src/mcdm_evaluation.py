"""
FBWM-FTOPSIS Hybrid Framework for Sustainable-Resilient Supplier Evaluation
============================================================================

This module implements Phase I of the SUPRA-PPO framework, providing a robust
Multi-Criteria Decision-Making (MCDM) foundation for supplier evaluation under
uncertainty. The hybrid methodology integrates two complementary fuzzy techniques
to derive sustainability-resilience scores that serve as domain knowledge priors
for downstream reinforcement learning.

Methodological Components:
1. Fuzzy Best-Worst Method (FBWM) - Criterion weight elicitation with consistency validation
2. Fuzzy TOPSIS - Supplier ranking via distance-based comparison to ideal solutions

Theoretical Foundations:
- FBWM formulation: Rezaei (2015, 2019) with fuzzy extensions by Guo & Zhao (2017)
- Linear programming decomposition: Three-LP approach for triangular fuzzy number weights
- FTOPSIS distance metrics: Chen (2000) vertex distance method for fuzzy sets
- Defuzzification: Graded Mean Integration Representation (GMIR) for crisp conversion
- Aggregation: Geometric mean operator for multi-expert consensus

Application Domain:
Lithium-ion battery cell supply chain for utility-scale grid energy storage systems,
evaluating suppliers across Economic, Environmental, Social, and Resilience dimensions.

References:
- Rezaei, J. (2015). Best-worst multi-criteria decision-making method. Omega, 53, 49-57.
- Rezaei, J. (2019). Piecewise linear value functions for multi-criteria decision-making. 
  Expert Systems with Applications, 98, 43-56.
- Guo, S., & Zhao, H. (2017). Fuzzy best-worst multi-criteria decision-making method and
  its applications. Knowledge-Based Systems, 121, 23-31.
- Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy
  environment. Fuzzy Sets and Systems, 114(1), 1-9.

Author: Ali Vaezi, Erfan Rabbani, Giulia Bruno
Date: October 2025
Version: 2.0
"""

import math
import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Tuple

# =============================================================================
# 1. TRIANGULAR FUZZY NUMBER (TFN) CLASS AND ARITHMETIC OPERATIONS
# =============================================================================

class TFN:
    """
    Represents a Triangular Fuzzy Number (TFN) as a triplet (l, m, u).
    
    A TFN characterizes uncertain or imprecise information through three parameters:
    - l (lower bound): Minimum plausible value
    - m (modal value): Most likely value (peak of membership function)
    - u (upper bound): Maximum plausible value
    
    Constraint: l ≤ m ≤ u for mathematical validity
    
    This implementation provides overloaded arithmetic operators for fuzzy
    arithmetic operations following Zadeh's extension principle, enabling
    intuitive fuzzy computations in MCDM applications.
    
    Attributes:
        l (float): Lower bound of the triangular fuzzy number
        m (float): Modal (most likely) value
        u (float): Upper bound of the triangular fuzzy number
        
    Example:
        >>> tfn1 = TFN(1, 2, 3)  # Represents "approximately 2"
        >>> tfn2 = TFN(2, 3, 4)  # Represents "approximately 3"
        >>> result = tfn1 + tfn2  # Fuzzy addition
        >>> print(result)  # TFN(3.0000, 5.0000, 7.0000)
    """
    __slots__ = ("l", "m", "u")

    def __init__(self, l: float, m: float, u: float):
        self.l, self.m, self.u = float(l), float(m), float(u)

    def __repr__(self) -> str:
        return f"TFN({self.l:.4f}, {self.m:.4f}, {self.u:.4f})"

    def __add__(self, other: 'TFN') -> 'TFN':
        return TFN(self.l + other.l, self.m + other.m, self.u + other.u)

    def __sub__(self, other: 'TFN') -> 'TFN':
        return TFN(self.l - other.u, self.m - other.m, self.u - other.l)

    def __mul__(self, other) -> 'TFN':
        if isinstance(other, TFN):
            return TFN(self.l * other.l, self.m * other.m, self.u * other.u)
        return TFN(self.l * other, self.m * other, self.u * other)

    def __rmul__(self, other) -> 'TFN':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'TFN':
        EPS = 1e-12
        if isinstance(other, TFN):
            return TFN(self.l / max(other.u, EPS), self.m / max(other.m, EPS), self.u / max(other.l, EPS))
        return TFN(self.l / max(other, EPS), self.m / max(other, EPS), self.u / max(other, EPS))

    def gmir(self) -> float:
        """
        Calculates the Graded Mean Integration Representation (GMIR) for defuzzification.
        
        GMIR provides a weighted average defuzzification method that emphasizes the modal
        value while incorporating boundary uncertainty. The formula weights the modal value
        four times more than the extreme values, ensuring stability in crisp conversion.
        
        Formula: GMIR(A) = (l + 4m + u) / 6
        
        This method is widely used in fuzzy MCDM due to its computational simplicity
        and proven effectiveness in preserving fuzzy information during defuzzification.
        
        Returns:
            float: Crisp (defuzzified) value representing the fuzzy number
            
        Reference:
            Chen, S.-H., & Hsieh, C.-H. (2000). Representation, ranking, distance, and
            similarity of L-R type fuzzy number and application. Australian Journal of
            Intelligent Processing Systems, 6(4), 217-229.
        """
        return (self.l + 4 * self.m + self.u) / 6.0

def tfn_geom_mean(tfn_list: List[TFN]) -> TFN:
    """
    Calculates the component-wise geometric mean of a list of triangular fuzzy numbers.
    
    The geometric mean is preferred over arithmetic mean in MCDM for aggregating
    expert judgments because it:
    1. Maintains the multiplicative consistency of pairwise comparisons
    2. Reduces the influence of extreme outlier judgments
    3. Preserves the reciprocal property in comparison matrices
    
    For a set of n TFNs, the geometric mean is computed separately for each component:
    - L_geo = (∏ l_i)^(1/n)
    - M_geo = (∏ m_i)^(1/n)
    - U_geo = (∏ u_i)^(1/n)
    
    Args:
        tfn_list (List[TFN]): List of triangular fuzzy numbers to aggregate
        
    Returns:
        TFN: Geometric mean triangular fuzzy number preserving triangular property
        
    Note:
        Returns TFN(0,0,0) for empty input list as a safe default.
    """
    k = len(tfn_list)
    if k == 0: return TFN(0, 0, 0)
    L = math.prod(t.l for t in tfn_list) ** (1.0 / k)
    M = math.prod(t.m for t in tfn_list) ** (1.0 / k)
    U = math.prod(t.u for t in tfn_list) ** (1.0 / k)
    return TFN(L, M, U)

# =============================================================================
# 2. FUZZY BEST-WORST METHOD (FBWM) IMPLEMENTATION
# =============================================================================

def _bwm_lp(br_idx: int, wr_idx: int, a_Bj: np.ndarray, a_jW: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solves the core Best-Worst Method optimization problem as a Linear Program.
    
    This function implements the BWM mathematical model (Rezaei, 2015) which minimizes
    the maximum absolute deviation between criterion weights and pairwise comparison
    judgments. The formulation ensures consistency between Best-to-Others (BO) and
    Others-to-Worst (OW) preference vectors.
    
    Mathematical Formulation:
        Minimize: ξ (consistency indicator)
        Subject to:
            |w_B - a_Bj * w_j| ≤ ξ  for all criteria j
            |w_j - a_jW * w_W| ≤ ξ  for all criteria j
            Σ w_j = 1
            w_j ≥ 0  for all j
    
    Where:
        - w_j: Weight of criterion j
        - a_Bj: Importance of Best criterion over criterion j
        - a_jW: Importance of criterion j over Worst criterion
        - ξ: Maximum deviation (to be minimized)
    
    Args:
        br_idx (int): Index of the Best (most important) criterion
        wr_idx (int): Index of the Worst (least important) criterion
        a_Bj (np.ndarray): Best-to-Others comparison vector (n elements)
        a_jW (np.ndarray): Others-to-Worst comparison vector (n elements)
        
    Returns:
        Tuple[np.ndarray, float]: 
            - Normalized criterion weights (sum = 1)
            - Consistency indicator ξ* (optimal objective value)
            
    Raises:
        RuntimeError: If linear program fails to converge
        
    Reference:
        Rezaei, J. (2015). Best-worst multi-criteria decision-making method.
        Omega, 53, 49-57. DOI: 10.1016/j.omega.2014.11.009
    """
    n = len(a_Bj)
    c = np.zeros(n + 1); c[-1] = 1.0
    A, b = [], []
    for j in range(n):
        row1 = np.zeros(n+1); row1[br_idx]=1.0; row1[j]-=a_Bj[j]; row1[-1]=-1.0
        A.append(row1); b.append(0.0)
        row2 = np.zeros(n+1); row2[br_idx]=-1.0; row2[j]+=a_Bj[j]; row2[-1]=-1.0
        A.append(row2); b.append(0.0)
    for j in range(n):
        row1 = np.zeros(n+1); row1[j]=1.0; row1[wr_idx]-=a_jW[j]; row1[-1]=-1.0
        A.append(row1); b.append(0.0)
        row2 = np.zeros(n+1); row2[j]=-1.0; row2[wr_idx]+=a_jW[j]; row2[-1]=-1.0
        A.append(row2); b.append(0.0)

    Aeq = np.zeros((1, n + 1)); Aeq[0, :n] = 1.0
    beq = np.array([1.0])
    bounds = [(0.0, None)] * (n + 1)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success: raise RuntimeError(f"BWM LP failed: {res.message}")
    w, xi = res.x[:n], res.x[-1]
    return w / np.sum(w), xi

def _extract_vectors(criteria: List[str], best: str, worst: str, BO: Dict[str, TFN], OW: Dict[str, TFN], which: str = "m") -> Tuple:
    """
    Extracts a specific component (l, m, or u) from fuzzy comparison vectors for LP solving.
    
    This helper function decomposes triangular fuzzy numbers into their constituent
    components, enabling the three-LP decomposition approach for fuzzy BWM. Each LP
    solves for weights using one component (lower, modal, or upper) of the fuzzy
    comparison judgments.
    
    Args:
        criteria (List[str]): List of all criterion names
        best (str): Name of the Best (most important) criterion
        worst (str): Name of the Worst (least important) criterion
        BO (Dict[str, TFN]): Best-to-Others fuzzy comparison dictionary
        OW (Dict[str, TFN]): Others-to-Worst fuzzy comparison dictionary
        which (str): Component to extract - 'l' (lower), 'm' (modal), or 'u' (upper)
        
    Returns:
        Tuple: (best_idx, worst_idx, a_Bj_vector, a_jW_vector)
            - best_idx: Integer index of best criterion
            - worst_idx: Integer index of worst criterion
            - a_Bj_vector: Crisp comparison values (Best to Others)
            - a_jW_vector: Crisp comparison values (Others to Worst)
    """
    idx = {c: i for i, c in enumerate(criteria)}
    a_Bj, a_jW = np.zeros(len(criteria)), np.zeros(len(criteria))
    for i, c in enumerate(criteria):
        tB, tW = BO[c], OW[c]
        if which == "l": a_Bj[i], a_jW[i] = tB.l, tW.l
        elif which == "u": a_Bj[i], a_jW[i] = tB.u, tW.u
        else: a_Bj[i], a_jW[i] = tB.m, tW.m
    return idx[best], idx[worst], a_Bj, a_jW

def fbwm_ci_from_scale(a_bw_gmir: float) -> float:
    """
    Provides the Consistency Index (CI) based on the Best-to-Worst comparison value.
    
    The Consistency Index represents the maximum possible inconsistency (ξ*) for a
    given Best-to-Worst preference intensity. This table is derived theoretically
    from the BWM model structure and serves as a normalization baseline for
    calculating the Consistency Ratio.
    
    CR = ξ* / CI
    
    Where ξ* ≤ CI indicates acceptable consistency (CR ≤ 1.0), with CR ≤ 0.10
    representing excellent consistency analogous to AHP's threshold.
    
    Args:
        a_bw_gmir (float): Defuzzified Best-to-Worst comparison intensity
        
    Returns:
        float: Consistency Index corresponding to the comparison scale
        
    Consistency Index Lookup Table:
        a_BW ∈ [1, 1.5] → CI = 0.44
        a_BW ∈ (1.5, 2.5] → CI = 0.99
        a_BW ∈ (2.5, 3.5] → CI = 1.51
        a_BW ∈ (3.5, 4.5] → CI = 2.00
        a_BW > 4.5 → CI = 2.46
        
    Reference:
        Rezaei, J. (2015). Best-worst multi-criteria decision-making method.
        Omega, 53, 49-57. (Table 4, p. 54)
    """
    cuts = [(1.5, 0.44), (2.5, 0.99), (3.5, 1.51), (4.5, 2.00)]
    for threshold, ci_value in cuts:
        if a_bw_gmir <= threshold: return ci_value
    return 2.46

def fbwm_weights_fuzzy(criteria: List[str], best: str, worst: str, BO: Dict[str, TFN], OW: Dict[str, TFN], ci_func) -> Tuple:
    """
    Calculates fuzzy criterion weights using the three-LP decomposition method.
    
    This function implements the Fuzzy Best-Worst Method (Guo & Zhao, 2017) by
    solving three separate linear programs for the lower (l), modal (m), and upper (u)
    components of the fuzzy comparison judgments. This approach preserves the
    triangular structure while ensuring mathematical consistency.
    
    The method guarantees:
    1. Each LP produces valid normalized weights (sum = 1)
    2. Sorting ensures triangular property: l ≤ m ≤ u for each weight
    3. Consistency is validated via the modal solution's ξ* value
    
    Process:
        Step 1: Extract l, m, u components from fuzzy comparisons
        Step 2: Solve three independent BWM-LPs
        Step 3: Sort weight components to ensure l ≤ m ≤ u
        Step 4: Calculate Consistency Ratio from modal solution
    
    Args:
        criteria (List[str]): List of all evaluation criterion names
        best (str): Name of the Best (most important) criterion
        worst (str): Name of the Worst (least important) criterion
        BO (Dict[str, TFN]): Best-to-Others fuzzy pairwise comparisons
        OW (Dict[str, TFN]): Others-to-Worst fuzzy pairwise comparisons
        ci_func (Callable): Function mapping a_BW value to Consistency Index
        
    Returns:
        Tuple containing:
            - fuzzy_w (Dict[str, TFN]): Fuzzy weights for each criterion
            - crisp_modal (Dict[str, float]): Crisp modal weights (sum = 1)
            - CR (float): Consistency Ratio (should be ≤ 0.10 for acceptable consistency)
            
    Note:
        The sorting step corrects potential numerical issues where LP solutions
        might produce non-triangular results due to floating-point arithmetic.
        This ensures mathematical validity without compromising solution quality.
        
    Reference:
        Guo, S., & Zhao, H. (2017). Fuzzy best-worst multi-criteria decision-making
        method and its applications. Knowledge-Based Systems, 121, 23-31.
    """
    # Extract crisp comparison vectors for each TFN component
    br_l, wr_l, aB_l, aW_l = _extract_vectors(criteria, best, worst, BO, OW, "l")
    br_m, wr_m, aB_m, aW_m = _extract_vectors(criteria, best, worst, BO, OW, "m")
    br_u, wr_u, aB_u, aW_u = _extract_vectors(criteria, best, worst, BO, OW, "u")
    
    wL, _ = _bwm_lp(br_l, wr_l, aB_l, aW_l)
    wM, xiM = _bwm_lp(br_m, wr_m, aB_m, aW_m)
    wU, _ = _bwm_lp(br_u, wr_u, aB_u, aW_u)
    
    # Fuzzy weights are created by sorting the results to ensure l <= m <= u.
    fuzzy_w = {c: TFN(*sorted([wL[i], wM[i], wU[i]])) for i, c in enumerate(criteria)}
    
    # Crisp modal weights are taken directly from the wM vector, which is guaranteed to sum to 1.
    crisp_modal = {c: wM[i] for i, c in enumerate(criteria)}

    a_bw_gmir = BO[worst].gmir()
    CI = ci_func(a_bw_gmir)
    CR = float(xiM) / CI if CI > 0 else np.nan
    return fuzzy_w, crisp_modal, CR

def aggregate_bo_ow(expert_bo_list: List[Dict[str, TFN]], expert_ow_list: List[Dict[str, TFN]]) -> Tuple:
    """
    Aggregates multiple experts' Best-Worst pairwise comparison judgments.
    
    In group decision-making scenarios, different experts may provide varying
    assessments of criterion importance. This function synthesizes diverse expert
    opinions into a single consolidated fuzzy comparison matrix using the geometric
    mean aggregation operator.
    
    The geometric mean is theoretically superior to arithmetic mean for aggregating
    pairwise comparisons because it:
    1. Preserves reciprocal properties (if a_ij = 1/a_ji, this holds after aggregation)
    2. Minimizes bias from extreme outlier judgments
    3. Maintains mathematical consistency in comparison matrices
    
    Args:
        expert_bo_list (List[Dict[str, TFN]]): List of Best-to-Others comparison
            dictionaries, one per expert
        expert_ow_list (List[Dict[str, TFN]]): List of Others-to-Worst comparison
            dictionaries, one per expert
            
    Returns:
        Tuple[Dict[str, TFN], Dict[str, TFN]]:
            - BO: Aggregated Best-to-Others fuzzy comparisons
            - OW: Aggregated Others-to-Worst fuzzy comparisons
            
    Example:
        Expert 1: Quality over Cost = (2, 3, 4)
        Expert 2: Quality over Cost = (1, 2, 3)
        Aggregated: Quality over Cost ≈ (1.41, 2.45, 3.46) [geometric mean]
    """
    crits = list(expert_bo_list[0].keys())
    BO = {c: tfn_geom_mean([bo[c] for bo in expert_bo_list]) for c in crits}
    OW = {c: tfn_geom_mean([ow[c] for ow in expert_ow_list]) for c in crits}
    return BO, OW

# =============================================================================
# 3. FUZZY TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
# =============================================================================

def aggregate_supplier_ratings(expert_ratings: List[Dict[str, Dict[str, TFN]]]) -> Dict[str, Dict[str, TFN]]:
    """
    Aggregates multiple experts' fuzzy supplier performance ratings into group consensus.
    
    When multiple domain experts evaluate supplier performance on various criteria,
    their individual assessments must be synthesized into a single decision matrix.
    This function applies component-wise geometric mean aggregation to preserve
    the mathematical properties required for robust TOPSIS ranking.
    
    Args:
        expert_ratings (List[Dict[str, Dict[str, TFN]]]): List of rating dictionaries
            where each expert provides: {supplier_id: {criterion: TFN_rating}}
            
    Returns:
        Dict[str, Dict[str, TFN]]: Aggregated ratings with structure
            {supplier_id: {criterion: aggregated_TFN}}
            
    Example:
        Expert 1 rates S1 on Quality: (7, 8, 9)
        Expert 2 rates S1 on Quality: (6, 7, 8)
        Aggregated: S1 Quality ≈ (6.48, 7.48, 8.49)
    """
    suppliers = list(expert_ratings[0].keys())
    criteria = list(expert_ratings[0][suppliers[0]].keys())
    return {s: {c: tfn_geom_mean([E[s][c] for E in expert_ratings]) for c in criteria} for s in suppliers}

def ftopsis_rank_suppliers(supplier_data: Dict[str, Dict[str, TFN]], criteria_types: Dict[str, str], fuzzy_weights: Dict[str, TFN]) -> Dict[str, float]:
    """
    Ranks suppliers using Fuzzy TOPSIS methodology based on distance to ideal solutions.
    
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) operates
    on the principle that the best alternative should have the shortest distance to
    the Fuzzy Positive Ideal Solution (FPIS) and the longest distance to the Fuzzy
    Negative Ideal Solution (FNIS). This dual-distance approach provides robust
    rankings even with conflicting criteria.
    
    Algorithm Steps:
    1. Normalize the fuzzy decision matrix via vector normalization
    2. Apply criterion weights to construct weighted normalized matrix
    3. Identify FPIS (best values) and FNIS (worst values) for each criterion
    4. Calculate Euclidean distances from each supplier to FPIS and FNIS
    5. Compute Closeness Coefficient: CC_i = d⁻_i / (d⁺_i + d⁻_i)
    
    The Closeness Coefficient ranges from 0 to 1, where values closer to 1 indicate
    superior overall performance across all weighted criteria.
    
    Args:
        supplier_data (Dict[str, Dict[str, TFN]]): Fuzzy performance ratings
            {supplier: {criterion: TFN_rating}}
        criteria_types (Dict[str, str]): Criterion optimization directions
            {'criterion_name': 'benefit'} for maximization criteria
            {'criterion_name': 'cost'} for minimization criteria
        fuzzy_weights (Dict[str, TFN]): Fuzzy importance weights from FBWM
            {criterion: TFN_weight}
            
    Returns:
        Dict[str, float]: Closeness coefficients (CC) for ranking
            {supplier_id: CC_score} where higher scores indicate better performance
            
    Mathematical Notes:
        - Distance metric: Euclidean distance in 3D fuzzy space √[(l₁-l₂)² + (m₁-m₂)² + (u₁-u₂)²]/3
        - Normalization handles heterogeneous criterion scales
        - FPIS/FNIS determination accounts for benefit vs. cost criteria
        
    References:
        Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under
        fuzzy environment. Fuzzy Sets and Systems, 114(1), 1-9.
    """
    suppliers, criteria = list(supplier_data.keys()), list(criteria_types.keys())
    EPS = 1e-12

    first_supplier_criteria = set(supplier_data[suppliers[0]].keys())
    assert not (first_supplier_criteria - set(fuzzy_weights)), f"Missing weights for: {first_supplier_criteria - set(fuzzy_weights)}"
    assert not (first_supplier_criteria - set(criteria_types)), f"Missing direction for: {first_supplier_criteria - set(criteria_types)}"

    r = {s: {} for s in suppliers}
    c_star = {c: max(supplier_data[s][c].u for s in suppliers) for c in criteria}
    a_minus = {c: min(supplier_data[s][c].l for s in suppliers) for c in criteria}
    
    for s in suppliers:
        for c in criteria:
            x = supplier_data[s][c]
            if criteria_types[c] == 'benefit': r[s][c] = x / max(c_star[c], EPS)
            else: r[s][c] = TFN(a_minus[c] / max(x.u, EPS), a_minus[c] / max(x.m, EPS), a_minus[c] / max(x.l, EPS))

    v = {s: {c: r[s][c] * fuzzy_weights[c] for c in criteria} for s in suppliers}
    FPIS, FNIS = {}, {}
    for c in criteria:
        col = [v[s][c] for s in suppliers]
        if criteria_types[c] == 'benefit':
            FPIS[c], FNIS[c] = TFN(max(t.l for t in col), max(t.m for t in col), max(t.u for t in col)), TFN(min(t.l for t in col), min(t.m for t in col), min(t.u for t in col))
        else:
            FPIS[c], FNIS[c] = TFN(min(t.l for t in col), min(t.m for t in col), min(t.u for t in col)), TFN(max(t.l for t in col), max(t.m for t in col), max(t.u for t in col))
    
    def d(a, b): return math.sqrt(((a.l - b.l)**2 + (a.m - b.m)**2 + (a.u - b.u)**2) / 3.0)
    d_plus = {s: sum(d(v[s][c], FPIS[c]) for c in criteria) for s in suppliers}
    d_minus = {s: sum(d(v[s][c], FNIS[c]) for c in criteria) for s in suppliers}
    return {s: d_minus[s] / max(d_plus[s] + d_minus[s], EPS) for s in suppliers}

# =============================================================================
# 4. UTILITY FUNCTIONS FOR VALIDATION AND DATA TRANSFORMATION
# =============================================================================

def _validate_fuzzy_weights(fw: Dict[str, TFN], crisp_modal_weights: Dict[str, float]):
    """
    Validates the mathematical integrity of fuzzy weights derived from FBWM.
    
    This function performs two critical validation checks:
    1. Triangular Property: Ensures l ≤ m ≤ u for each fuzzy weight
    2. Normalization Constraint: Verifies modal weights sum to exactly 1.0
    
    These validations are essential for ensuring the fuzzy weights can be
    meaningfully used in downstream TOPSIS calculations and for detecting
    potential numerical instabilities in the LP solutions.
    
    Args:
        fw (Dict[str, TFN]): Fuzzy criterion weights to validate
        crisp_modal_weights (Dict[str, float]): Corresponding modal (m-component) weights
        
    Raises:
        AssertionError: If triangular property is violated for any criterion
        AssertionError: If modal weights sum deviates from 1.0 beyond tolerance (1e-6)
        
    Side Effects:
        Prints validation results including sum of L, M, U components
        
    Note:
        The L and U component sums may deviate from 1.0 due to the independent
        LP solutions, but the M component (modal weights) must sum to 1.0 exactly.
    """
    for k, t in fw.items():
        assert t.l <= t.m <= t.u, f"Non-triangular weight for {k}: {t}"
    
    modal_sum = sum(crisp_modal_weights.values())
    assert abs(modal_sum - 1) < 1e-6, f"Sum of crisp modal weights is not 1: {modal_sum}"
    
    l_sum, u_sum = sum(t.l for t in fw.values()), sum(t.u for t in fw.values())
    print("Fuzzy weights are valid (triangular property holds).")
    print(f"Sum of L-bounds: {l_sum:.4f}, M-bounds (from crisp): {modal_sum:.4f}, U-bounds: {u_sum:.4f}")

def fuzzify(score: float, uncertainty_percent: int = 10) -> TFN:
    """
    Converts crisp numerical scores into triangular fuzzy numbers with symmetric spread.
    
    In real-world supplier evaluation, crisp scores often mask inherent uncertainty
    in assessment. This function models epistemic uncertainty by creating symmetric
    fuzzy numbers around the modal value, representing measurement imprecision
    and subjective judgment variability.
    
    The uncertainty percentage defines the relative spread of the fuzzy number:
    - l (lower) = m × (1 - α)
    - m (modal) = original score
    - u (upper) = m × (1 + α)
    
    where α = uncertainty_percent / 100
    
    Boundary constraints ensure the resulting TFN remains within valid [0, 10] scale.
    
    Args:
        score (float): Crisp performance score (typically on 0-10 scale)
        uncertainty_percent (int): Symmetric uncertainty range as percentage (default: 10%)
        
    Returns:
        TFN: Triangular fuzzy number representing the score with uncertainty
        
    Example:
        >>> fuzzify(7.0, uncertainty_percent=10)
        TFN(6.3, 7.0, 7.7)  # 10% spread around modal value
        >>> fuzzify(9.5, uncertainty_percent=10)
        TFN(8.55, 9.5, 10.0)  # Upper bound capped at 10
        
    Note:
        10% uncertainty is standard in MCDM literature for modeling expert judgment
        variability. Higher percentages may be appropriate for highly subjective criteria.
    """
    m = float(score)
    alpha = float(uncertainty_percent) / 100.0
    l = m * (1 - alpha)
    u = m * (1 + alpha)
    return TFN(max(0, l), m, min(10, u))

def calculate_dimensional_scores(supplier_data: Dict[str, Dict[str, TFN]], 
                                criteria_types: Dict[str, str], 
                                fuzzy_weights: Dict[str, TFN]) -> Dict[str, Dict[str, float]]:
    """
    Calculates TOPSIS scores for each dimension separately (Economic, Environmental, Social, Resilience).
    
    This function groups the 16 criteria into their respective strategic dimensions and
    computes partial TOPSIS scores for each dimension. This provides granular insight
    into supplier performance across different aspects of sustainability and resilience.
    
    Dimension Mappings (based on the four-pillar framework):
        Economic: Cost Comp., Quality, Delivery, Financial (4 criteria)
        Environmental: Critical-minerals, Eco Cert., Recycling, Waste Mgmt (4 criteria)
        Social: Health Safety, Social compliance, Labor Contract, Employment (4 criteria)
        Resilience: Agility, Flexibility, Robustness, Visibility (4 criteria)
    
    Args:
        supplier_data: Fuzzy performance ratings {supplier: {criterion: TFN}}
        criteria_types: Optimization directions {criterion: 'benefit'/'cost'}
        fuzzy_weights: Fuzzy importance weights {criterion: TFN}
        
    Returns:
        Dict[str, Dict[str, float]]: Dimensional scores scaled to 0-10 range
            {supplier: {'Economic': score, 'Environmental': score, 'Social': score, 'Resilience': score}}
    """
    # Define dimension groupings
    dimensions = {
        'Economic': ['Cost Comp.', 'Quality', 'Delivery', 'Financial'],
        'Environmental': ['Critical-minerals', 'Eco Cert.', 'Recycling', 'Waste Mgmt'],
        'Social': ['Health Safety', 'Social compliance', 'Labor Contract', 'Employment'],
        'Resilience': ['Agility', 'Flexibility', 'Robustness', 'Visibility']
    }
    
    dimensional_scores = {s: {} for s in supplier_data.keys()}
    
    for dim_name, dim_criteria in dimensions.items():
        # Filter data for this dimension only
        dim_supplier_data = {
            s: {c: supplier_data[s][c] for c in dim_criteria if c in supplier_data[s]}
            for s in supplier_data.keys()
        }
        
        dim_criteria_types = {c: criteria_types[c] for c in dim_criteria if c in criteria_types}
        dim_fuzzy_weights = {c: fuzzy_weights[c] for c in dim_criteria if c in fuzzy_weights}
        
        # Calculate TOPSIS scores for this dimension
        dim_rankings = ftopsis_rank_suppliers(
            supplier_data=dim_supplier_data,
            criteria_types=dim_criteria_types,
            fuzzy_weights=dim_fuzzy_weights
        )
        
        # Scale to 0-10 range for interpretability (0 = worst, 10 = best)
        for supplier, cc_score in dim_rankings.items():
            dimensional_scores[supplier][dim_name] = cc_score * 10.0
    
    return dimensional_scores

def get_ftopsis_scores_for_rl(final_rankings: Dict[str, float]) -> np.ndarray:
    """
    Exports FTOPSIS closeness coefficients in the standardized format for RL integration.
    
    This function serves as the critical interface between Phase I (MCDM evaluation)
    and Phase II (RL-based dynamic allocation). The FTOPSIS scores represent
    domain knowledge that can be incorporated as Bayesian priors in the reinforcement
    learning agent's policy initialization or state augmentation.
    
    The output array order matches the supplier indexing convention used in
    run_experiment.py, ensuring seamless integration without manual reordering.
    
    Args:
        final_rankings (Dict[str, float]): FTOPSIS closeness coefficients
            {supplier_id: CC_score} where CC ∈ [0, 1]
            
    Returns:
        np.ndarray: 1D array of shape (5,) containing [S1, S2, S3, S4, S5] scores
        
    Example:
        >>> rankings = {'S1': 0.15, 'S2': 0.62, 'S3': 0.79, 'S4': 0.52, 'S5': 0.78}
        >>> get_ftopsis_scores_for_rl(rankings)
        array([0.15, 0.62, 0.79, 0.52, 0.78])
        
    Integration Note:
        In run_experiment.py, use as:
        "ftopsis_scores": np.array([0.1523, 0.6237, 0.7905, 0.5198, 0.7837])
        
    Reference:
        See Section 4.2, "Integration of MCDM Priors in RL State Space"
    """
    return np.array([
        final_rankings['S1'],
        final_rankings['S2'],
        final_rankings['S3'],
        final_rankings['S4'],
        final_rankings['S5']
    ])

def print_supplier_archetypes():
    """
    Displays supplier archetype definitions for interpretability and validation.
    
    These archetypes represent distinct strategic positions in the sustainable
    supplier landscape, derived from real-world lithium-ion battery supply chain
    dynamics. Each archetype embodies specific trade-offs between cost, quality,
    sustainability, and operational resilience.
    
    Archetype Characterization:
        S1 (Low-Cost Leader): Optimizes for cost efficiency, weak on ESG metrics
        S2 (Green Specialist): Environmental/social excellence, premium pricing
        S3 (Resilient Incumbent): Operational reliability, agility, and robustness
        S4 (Balanced Generalist): Average across all dimensions, no extremes
        S5 (Premium Innovator): Top-tier quality and sustainability, highest cost
        
    Purpose:
        - Aids in results interpretation and validation
        - Provides context for stakeholder communication
        - Enables sanity checking of FTOPSIS rankings against archetype expectations
        
    Side Effects:
        Prints formatted archetype descriptions to console
        
    Reference:
        See Section 4.1, "Supplier Archetype Design and Performance Profiles"
    """
    print("\n" + "="*80)
    print("SUPPLIER ARCHETYPES (Battery Supply Chain)")
    print("="*80)
    archetypes = {
        'S1': 'Low-Cost Leader: Cost minimization but poor sustainability/stability',
        'S2': 'Green Specialist: Environmental excellence, premium pricing',
        'S3': 'Resilient Incumbent: Operational reliability and robustness',
        'S4': 'Balanced Generalist: Average capabilities across all dimensions',
        'S5': 'Premium Innovator: Top quality and sustainability, very high cost'
    }
    for supplier, desc in archetypes.items():
        print(f"  {supplier}: {desc}")
    print("="*80 + "\n")

# =============================================================================
# MAIN EXECUTION BLOCK - PHASE I SUPPLIER EVALUATION PIPELINE
# =============================================================================

if __name__ == "__main__":
    import os
    import pandas as pd

    # Create output directory for MCDM results
    output_dir = "../results/mcdm/"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("PHASE I: SUSTAINABLE-RESILIENT SUPPLIER EVALUATION (FBWM-FTOPSIS)")
    print("="*80)
    print("Application: Lithium-ion Battery Supply Chain for Grid Storage Systems")
    print("Framework: Hybrid FBWM-FTOPSIS with 16 Criteria Evaluation")
    print("="*80)
    
    print_supplier_archetypes()

    # -------------------------------------------------------------------------
    # SECTION 1: INPUT DATA CONFIGURATION
    # -------------------------------------------------------------------------
    # All model inputs are centralized here for maintainability and transparency.
    # Modify this dictionary to adapt the evaluation framework to different:
    #   - Criteria sets (add/remove evaluation dimensions)
    #   - Fuzzy linguistic scales (adjust comparison intensities)
    #   - Expert judgments (incorporate multiple decision-makers)
    #   - Supplier performance data (update with empirical measurements)
    
    INPUT_DATA = {
        "criteria": [
            "Cost Comp.", "Quality", "Delivery", "Financial", "Critical-minerals", 
            "Eco Cert.", "Recycling", "Waste Mgmt", "Health Safety", "Social compliance", 
            "Labor Contract", "Employment", "Agility", "Flexibility", "Robustness", "Visibility"
        ],
        "criteria_types": {
            "Cost Comp.": "cost", "Quality": "benefit", "Delivery": "benefit", "Financial": "benefit", 
            "Critical-minerals": "benefit", "Eco Cert.": "benefit", "Recycling": "benefit", 
            "Waste Mgmt": "cost", "Health Safety": "benefit", "Social compliance": "benefit", 
            "Labor Contract": "benefit", "Employment": "benefit", "Agility": "benefit", 
            "Flexibility": "benefit", "Robustness": "benefit", "Visibility": "benefit"
        },
        "scale": {
            "Equal": TFN(1, 1, 1), "Moderate": TFN(1, 1.5, 2), "Moderate+": TFN(1.5, 2, 2.5), 
            "Strong": TFN(2, 2.5, 3), "Strong+": TFN(2.5, 3, 3.5), "Very": TFN(3, 3.5, 4), 
            "Extreme": TFN(3.5, 4, 4.5)
        },
        "fbwm_judgments": {
            # Best criterion: Robustness (aligns with resilience-oriented evaluation)
            # Worst criterion: Cost Comp. (allows cost–sustainability trade-off analysis)
            "best_criterion": "Robustness",
            "worst_criterion": "Cost Comp.",
            "experts": [
                { # Expert 1 pairwise comparisons (Best-to-Others and Others-to-Worst)
                    # CR = 0.0442 — well below the 0.10 consistency threshold
                    "bo": {"Cost Comp.": TFN(1.5,2,2.5), "Quality": TFN(1,1,1), "Delivery": TFN(1,1.5,2), "Financial": TFN(1,1.5,2), "Critical-minerals": TFN(2,2.5,3), "Eco Cert.": TFN(2.5,3,3.5), "Recycling": TFN(3,3.5,4), "Waste Mgmt": TFN(2.5,3,3.5), "Health Safety": TFN(2,2.5,3), "Social compliance": TFN(2.5,3,3.5), "Labor Contract": TFN(2.5,3,3.5), "Employment": TFN(2.5,3,3.5),"Agility": TFN(1,1.5,2), "Flexibility": TFN(1.5,2,2.5), "Robustness": TFN(1,1,1), "Visibility": TFN(2,2.5,3)},
                    "ow": {"Cost Comp.": TFN(1,1,1), "Quality": TFN(3,3.5,4), "Delivery": TFN(2.5,3,3.5), "Financial": TFN(2.5,3,3.5), "Critical-minerals": TFN(1.5,2,2.5), "Eco Cert.": TFN(1.5,2,2.5), "Recycling": TFN(3,3.5,4), "Waste Mgmt": TFN(1.5,2,2.5), "Health Safety": TFN(2,2.5,3), "Social compliance": TFN(1.5,2,2.5), "Labor Contract": TFN(1.5,2,2.5), "Employment": TFN(1.5,2,2.5),"Agility": TFN(2.5,3,3.5), "Flexibility": TFN(2,2.5,3), "Robustness": TFN(3,3.5,4), "Visibility": TFN(2,2.5,3)}
                }
            ]
        },
        "supplier_crisp_scores": {
            # Supplier performance scores (Table 1, Section 4.1)
            # Order: Cost Comp., Quality, Delivery, Financial, Critical-minerals, Eco Cert., 
            #        Recycling, Waste Mgmt, Health Safety, Social Compliance, Labor Contract, 
            #        Employment, Agility, Flexibility, Robustness, Visibility
            'S1': [9.5, 6.8, 6.4, 7.9, 2.3, 1.8, 2.9, 3.1, 3.5, 2.7, 3.8, 2.4, 5.2, 6.1, 4.9, 4.6],
            'S2': [4.4, 8.2, 8.1, 7.1, 9.4, 9.1, 8.7, 9.2, 8.6, 9.0, 8.4, 8.3, 6.2, 6.8, 5.9, 7.1],
            'S3': [6.8, 8.5, 8.8, 8.7, 6.8, 7.0, 6.7, 6.1, 7.2, 7.4, 6.7, 6.9, 9.2, 8.8, 9.5, 8.9],
            'S4': [7.1, 7.4, 7.6, 7.3, 6.6, 6.5, 6.3, 6.2, 5.7, 6.2, 5.8, 6.1, 7.1, 7.0, 6.6, 6.8],
            'S5': [4.0, 9.4, 9.3, 8.9, 8.2, 8.3, 7.9, 8.0, 7.8, 7.6, 7.5, 7.3, 8.8, 9.1, 9.4, 8.7]
        }
    }

    print("--- Phase 1: Sustainable-Resilient Supplier Evaluation using FBWM-FTOPSIS ---\n")
    
    # -------------------------------------------------------------------------
    # SECTION 2: FUZZY BEST-WORST METHOD (FBWM) - CRITERION WEIGHT DERIVATION
    # -------------------------------------------------------------------------
    # Extract and aggregate expert pairwise comparison judgments
    expert_bo_list = [exp["bo"] for exp in INPUT_DATA["fbwm_judgments"]["experts"]]
    expert_ow_list = [exp["ow"] for exp in INPUT_DATA["fbwm_judgments"]["experts"]]
    
    # Aggregate multi-expert judgments using geometric mean operator
    BO, OW = aggregate_bo_ow(expert_bo_list, expert_ow_list)
    
    # Solve the fuzzy BWM model via three-LP decomposition
    fuzzy_weights, modal_weights, cr = fbwm_weights_fuzzy(
        criteria=INPUT_DATA["criteria"],
        best=INPUT_DATA["fbwm_judgments"]["best_criterion"],
        worst=INPUT_DATA["fbwm_judgments"]["worst_criterion"],
        BO=BO, OW=OW,
        ci_func=fbwm_ci_from_scale
    )
    
    # Display FBWM results with consistency validation
    print("--- FBWM Results (16 Criteria) ---")
    _validate_fuzzy_weights(fuzzy_weights, modal_weights)
    print("\nCalculated Crisp Modal Weights:")
    for crit, weight in sorted(modal_weights.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {crit:<20}: {weight:.4f}")
    print(f"\nConsistency Ratio (CR): {cr:.4f} {'(Consistent)' if cr <= 0.1 else '(Inconsistent)'}\n")

    # -------------------------------------------------------------------------
    # SECTION 3: FUZZY TOPSIS - SUPPLIER RANKING VIA DISTANCE TO IDEAL SOLUTIONS
    # -------------------------------------------------------------------------
    # Convert crisp supplier scores to fuzzy numbers with 10% uncertainty
    fuzzified_ratings = {}
    for supplier, scores in INPUT_DATA["supplier_crisp_scores"].items():
        fuzzified_ratings[supplier] = {
            crit: fuzzify(score, uncertainty_percent=10) 
            for crit, score in zip(INPUT_DATA["criteria"], scores)
        }
    
    # Aggregate ratings (single expert scenario - could extend to multi-expert)
    agg_ratings = aggregate_supplier_ratings([fuzzified_ratings])
    
    # Execute FTOPSIS ranking algorithm
    final_rankings = ftopsis_rank_suppliers(
        supplier_data=agg_ratings,
        criteria_types=INPUT_DATA["criteria_types"],
        fuzzy_weights=fuzzy_weights
    )
    
    # -------------------------------------------------------------------------
    # SECTION 4: DIMENSIONAL ANALYSIS AND RESULTS PRESENTATION
    # -------------------------------------------------------------------------
    
    # Calculate dimensional scores (Economic, Environmental, Social, Resilience)
    dimensional_scores = calculate_dimensional_scores(
        supplier_data=agg_ratings,
        criteria_types=INPUT_DATA["criteria_types"],
        fuzzy_weights=fuzzy_weights
    )
    
    # Sort suppliers by closeness coefficient (descending order = best to worst)
    sorted_suppliers = sorted(final_rankings.items(), key=lambda item: item[1], reverse=True)
    
    print("\n" + "="*80)
    print("FTOPSIS FINAL RANKINGS (16 CRITERIA)")
    print("="*80)
    print(f"{'Rank':<6} | {'Supplier':<12} | {'Archetype':<35} | {'CC Score':<10}")
    print("-" * 80)
    
    archetype_names = {
        'S1': 'Low-Cost Leader',
        'S2': 'Green Specialist',
        'S3': 'Resilient Incumbent',
        'S4': 'Balanced Generalist',
        'S5': 'Premium Innovator'
    }
    
    for i, (supplier, cc) in enumerate(sorted_suppliers, 1):
        print(f"{i:<6} | {supplier:<12} | {archetype_names[supplier]:<35} | {cc:.4f}")
    
    print("="*80)
    
    # Calculate aggregated dimensional weights for reporting
    dimension_groups = {
        'Economic': ['Cost Comp.', 'Quality', 'Delivery', 'Financial'],
        'Environmental': ['Critical-minerals', 'Eco Cert.', 'Recycling', 'Waste Mgmt'],
        'Social': ['Health Safety', 'Social compliance', 'Labor Contract', 'Employment'],
        'Resilience': ['Agility', 'Flexibility', 'Robustness', 'Visibility']
    }
    
    aggregated_dim_weights = {}
    for dim_name, dim_criteria in dimension_groups.items():
        dim_weight_sum = sum(modal_weights[c] for c in dim_criteria if c in modal_weights)
        aggregated_dim_weights[dim_name] = dim_weight_sum
    
    print("\n" + "="*80)
    print("AGGREGATED DIMENSIONAL WEIGHTS (from FBWM)")
    print("="*80)
    total_weight = sum(aggregated_dim_weights.values())
    for dim_name, dim_weight in sorted(aggregated_dim_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dim_name:<15}: {dim_weight:.4f} ({dim_weight*100:.1f}%)")
    print(f"  {'Total':<15}: {total_weight:.4f} (should be 1.0000)")
    print("="*80)
    
    # Display dimensional scores table with TOPSIS scores
    print("\n" + "="*80)
    print("DIMENSIONAL PERFORMANCE BREAKDOWN")
    print("="*80)
    print("Note: Scores are RELATIVE TOPSIS rankings (0=worst among 5 suppliers, 10=best)")
    print("-" * 80)
    print(f"{'Supplier':<10} | {'Economic':<10} | {'Environmental':<14} | {'Social':<10} | {'Resilience':<12} | {'Composite':<10} | {'Rank':<6}")
    print("-" * 80)
    
    for i, (supplier, cc) in enumerate(sorted_suppliers, 1):
        econ = dimensional_scores[supplier]['Economic']
        env = dimensional_scores[supplier]['Environmental']
        soc = dimensional_scores[supplier]['Social']
        res = dimensional_scores[supplier]['Resilience']
        comp = cc * 10.0  # Scale composite to 0-10 for consistency
        print(f"{supplier:<10} | {econ:<10.2f} | {env:<14.2f} | {soc:<10.2f} | {res:<12.2f} | {comp:<10.2f} | {i:<6}")
    
    print("="*80)
    
    # Calculate and display RAW average scores for absolute performance context
    print("\n" + "="*80)
    print("RAW AVERAGE PERFORMANCE (Absolute Scores 0-10)")
    print("="*80)
    print("Note: These are ACTUAL performance levels, not relative rankings")
    print("-" * 80)
    print(f"{'Supplier':<10} | {'Economic':<10} | {'Environmental':<14} | {'Social':<10} | {'Resilience':<12}")
    print("-" * 80)
    
    criteria_list = INPUT_DATA["criteria"]
    for supplier in ['S3', 'S5', 'S2', 'S4', 'S1']:  # Sorted by FTOPSIS rank
        raw_scores = INPUT_DATA["supplier_crisp_scores"][supplier]
        econ_raw = np.mean([raw_scores[criteria_list.index(c)] for c in dimension_groups['Economic']])
        env_raw = np.mean([raw_scores[criteria_list.index(c)] for c in dimension_groups['Environmental']])
        soc_raw = np.mean([raw_scores[criteria_list.index(c)] for c in dimension_groups['Social']])
        res_raw = np.mean([raw_scores[criteria_list.index(c)] for c in dimension_groups['Resilience']])
        print(f"{supplier:<10} | {econ_raw:<10.2f} | {env_raw:<14.2f} | {soc_raw:<10.2f} | {res_raw:<12.2f}")
    
    print("="*80)
    print("\nINTERPRETATION GUIDE:")
    print("  - TOPSIS Scores (Table 1): Relative ranking among suppliers (competitive position)")
    print("  - Raw Averages (Table 2): Absolute performance capability (intrinsic quality)")
    print("  - S1's 0.00 scores mean 'worst relative position', NOT 'zero performance'")
    print("  - S2's 10.00 scores mean 'best relative position', NOT 'perfect performance'")
    print("="*80)
    
    # Export closeness coefficients for Phase II RL integration
    rl_scores = get_ftopsis_scores_for_rl(final_rankings)
    
    print("\n" + "="*80)
    print("EXPORT FOR PHASE II (RL INTEGRATION)")
    print("="*80)
    print("Copy these scores to run_experiment.py line 69:")
    scores_list = [float(f"{x:.4f}") for x in rl_scores]
    print(f'"ftopsis_scores": np.array({scores_list}),')
    print("\nOrdered as: [S1, S2, S3, S4, S5]")
    print(f"Values:      [{rl_scores[0]:.4f}, {rl_scores[1]:.4f}, {rl_scores[2]:.4f}, {rl_scores[3]:.4f}, {rl_scores[4]:.4f}]")
    print("="*80)
    
    # -------------------------------------------------------------------------
    # SECTION 5: SUMMARY STATISTICS AND VALIDATION
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PHASE I SUMMARY")
    print("="*80)
    print(f"FBWM Consistency Ratio: {cr:.4f} (Target: <= 0.10)")
    print(f"Criteria Evaluated: {len(INPUT_DATA['criteria'])}")
    print(f"Suppliers Ranked: {len(sorted_suppliers)}")
    print(f"Top Performer: {sorted_suppliers[0][0]} ({archetype_names[sorted_suppliers[0][0]]}) - Score: {sorted_suppliers[0][1]:.4f}")
    print(f"Worst Performer: {sorted_suppliers[-1][0]} ({archetype_names[sorted_suppliers[-1][0]]}) - Score: {sorted_suppliers[-1][1]:.4f}")
    print(f"Score Range: {sorted_suppliers[-1][1]:.4f} - {sorted_suppliers[0][1]:.4f}")
    print("="*80)
    
    # =========================================================================
    # SAVE OUTPUTS TO FILES FOR PIPELINE INTEGRATION
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING OUTPUTS TO results/mcdm/")
    print("="*80)
    
    # Export 1: FBWM Weights
    weights_df = pd.DataFrame([
        {"Criterion": k, "Weight": v, "Rank": i+1}
        for i, (k, v) in enumerate(sorted(modal_weights.items(), key=lambda x: x[1], reverse=True))
    ])
    weights_df.to_csv(os.path.join(output_dir, "fbwm_weights.csv"), index=False)
    print("fbwm_weights.csv - Criterion importance weights")
    
    # Export 2: FTOPSIS Rankings
    rankings_df = pd.DataFrame([
        {
            "Rank": sorted_suppliers.index((supplier, final_rankings[supplier])) + 1,
            "Supplier": supplier,
            "Archetype": archetype_names[supplier],
            "CC_Score": final_rankings[supplier],
            "Allocation_Priority": "High" if final_rankings[supplier] > 0.6 else "Medium" if final_rankings[supplier] > 0.4 else "Low"
        }
        for supplier in ['S1', 'S2', 'S3', 'S4', 'S5']
    ])
    rankings_df = rankings_df.sort_values("CC_Score", ascending=False).reset_index(drop=True)
    rankings_df["Rank"] = range(1, len(rankings_df) + 1)
    rankings_df.to_csv(os.path.join(output_dir, "ftopsis_rankings.csv"), index=False)
    print("ftopsis_rankings.csv - Final supplier rankings")
    
    # Export 3: Dimensional Scores with CC_Score (scaled 0-10)
    dimensional_data = []
    for supplier in ['S1', 'S2', 'S3', 'S4', 'S5']:
        dimensional_data.append({
            "Supplier": supplier,
            "Archetype": archetype_names[supplier],
            "Economic": dimensional_scores[supplier]['Economic'],
            "Environmental": dimensional_scores[supplier]['Environmental'],
            "Social": dimensional_scores[supplier]['Social'],
            "Resilience": dimensional_scores[supplier]['Resilience'],
            "CC_Score": final_rankings[supplier] * 10.0  # Scale composite CC score to 0-10 for consistency
        })
    dimensional_df = pd.DataFrame(dimensional_data)
    # Sort by CC_Score descending (best to worst)
    dimensional_df = dimensional_df.sort_values("CC_Score", ascending=False).reset_index(drop=True)
    dimensional_df.to_csv(os.path.join(output_dir, "dimensional_scores.csv"), index=False)
    print("dimensional_scores.csv - Performance by dimension")
    
    # Export 4: Normalized scores for RL integration (NumPy format)
    np.save(os.path.join(output_dir, "supplier_scores_for_rl.npy"), rl_scores)
    print("supplier_scores_for_rl.npy - Normalised scores for RL integration")
    
    print("="*80)
    print(f"All outputs saved to: {os.path.abspath(output_dir)}")
    print("=" * 80)
    print("-> Next Step: Run 'python src/run_experiment.py' to use these scores")
    print("=" * 80)
    print("\nPhase I Complete. Ready for Phase II (SUPRA-PPO Training).\n")