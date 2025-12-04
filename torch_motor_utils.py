from kingdon import Algebra, MultiVector
import torch
from data_loader import JOINT_TRANSLATORS
from typing import Tuple
import json

alg = Algebra(3, 0, 1)
locals().update(alg.blades)
with open('serialized/kinematic_child_parent.json') as f:
    kinematic_child_parent = json.load(f)

# exp/log
def null_exp(B: MultiVector)-> MultiVector:
    return torch.ones_like(B.e01) + B

def rot_exp(B: MultiVector)-> MultiVector:
    signature = (B ** 2).e
    normsq = abs(signature)
    t = torch.sqrt(normsq)
    return torch.cos(t) + torch.sinc(t/torch.pi)*B

def se3_split(B: MultiVector, tol=1e-8):
    inner = (B|~B).e
    inner_safe = inner.clamp_min(tol)
    r = 0.5*(B^~B)/inner_safe
    return B*(1-r), B*r

def se3_exp(B: MultiVector, tol=1e-8)-> MultiVector:
    B1, B0 = se3_split(B, tol)
    R1 = rot_exp(B1)
    R0 = null_exp(B0)
    return R1*R0

def origin_split(B: MultiVector):
    keys = B.keys()
    vals = B.values()
    # Clone slices to avoid aliasing underlying storage that could be modified later
    B0 = alg.multivector(keys=keys[:3], values=[v.clone() for v in vals[:3]])
    B1 = alg.multivector(keys=keys[3:], values=[v.clone() for v in vals[3:]])
    return B0, B1

def trans_rot_exp(B: MultiVector, tol=1e-8)-> MultiVector:
    B0, B1 = origin_split(B)
    R0 = null_exp(B0)
    R1 = rot_exp(B1)
    return R0 * R1

# log
def normsq(A: MultiVector):
    return A.sp(A.reverse()).e.abs()

def norm(A):
    return torch.sqrt(normsq(A) + 1e-12)

def inv_norm(A: MultiVector, eps: float = 1e-12):
    """Compute 1 / ||A|| using rsqrt to avoid saving sqrt outputs in autograd."""
    return torch.rsqrt(normsq(A) + eps)


def translator_log(R: MultiVector)-> MultiVector:
    return R.grade(2)

def rotor_log(R: MultiVector, tol=1e-8)-> MultiVector:
    blade = R.grade(2)
    phi = torch.atan2(norm(blade), R.e.clamp(-1.0, 1.0))
    phi_over_sin_phi = torch.where(phi.abs() > 1e-2, phi / torch.sin(phi), 1 + phi**2 / 6 + 7 * phi**4 / 360)
    return phi_over_sin_phi * blade

def motor_split_log(R: MultiVector, tol=1e-8)-> Tuple[MultiVector, MultiVector]:
    # FIXME: when the tangent not well defined
    # assert (normsq(R.grade(2)) > tol).all(), "zero bivector"
    T1, T0 = se3_split(R.grade(2))
    phi = torch.atan2(norm(T1), R.e.clamp(-1.0, 1.0))
    phi_over_sin_phi = torch.where(phi.abs() > 1e-2, phi / torch.sin(phi), 1 + phi**2 / 6 + 7 * phi**4 / 360)
    return phi_over_sin_phi * T1 + T0 / (R.e + 1e-6)

def motor_origin_split_log(R: MultiVector, tol=1e-8):
    # FIXME: when the tangent not well defined
    # assert (normsq(R.grade(2)) > tol).all(), "zero bivector"
    T0, T1 = origin_split(R.grade(2))
    phi = torch.atan2(norm(T1), R.e.clamp(-1.0, 1.0))
    phi_over_sin_phi = torch.where(phi.abs() > 1e-2, phi / torch.sin(phi), 1 + phi**2 / 6 + 7 * phi**4 / 360)
    return phi_over_sin_phi * T1, T0 / (R.e + 1e-6)

# rotor_accumulation
# along time
# FIXME prefix sum for parallelization
def cumprod(initial, mvs):
    result = initial
    out = []
    for mv in mvs:
        result = result * mv
        out.append(torch.stack(result.values()))
    return alg.multivector(values=torch.cat(out, dim=1), keys=mvs.keys())

# along kinematic chain
def chain_accumulate(rotors):
    """
    Functionally accumulate along the kinematic tree without in-place tensor writes
    to keep autograd happy.

    Given per-joint rotors, propagate from parent to child so that each child's
    motor is left-multiplied by its parent's motor.
    """
    full_motors = JOINT_TRANSLATORS * rotors
    # Rebuild values tensor at each step instead of in-place indexing
    keys = full_motors.keys()
    vals = torch.stack(full_motors.values())  # [n_blades, T, K, B]
    for child, parent in kinematic_child_parent:
        if child >= 22:
            break
        if parent > 0:
            # Slice parent and child joints as MultiVectors of width 1 along joint dim
            parent_vals = vals[:, :, parent-1:parent, :]
            child_vals = vals[:, :, child-1:child, :]
            parent_mv = alg.multivector(keys=keys, values=parent_vals)
            child_mv = alg.multivector(keys=keys, values=child_vals)
            prod_vals = torch.stack((parent_mv * child_mv).values())  # [n_blades, T, 1, B]
            # Replace child's slice by concatenation (no in-place write)
            vals = torch.cat([vals[:, :, :child-1, :], prod_vals, vals[:, :, child:, :]], dim=2)
    return alg.multivector(keys=keys, values=vals)


# inspectation
def max_diff(mv1, mv2):
    return torch.cat((mv1 - mv2).values()).abs().max()

def mean_diff(mv1, mv2):
    return torch.cat((mv1 - mv2).values()).abs().mean()