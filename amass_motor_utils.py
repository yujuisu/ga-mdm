import sys
sys.path.append('../')
from kingdon import Algebra, MultiVector
import numpy as np
alg = Algebra(3, 0, 1)
locals().update(alg.blades)


smpl = np.load('serialized/model.npz')
kinematic_child_parent = list(enumerate(smpl['kintree_table'][0][1:22], start=1))

def xyz_to_point(v):
    x,y,z = v
    return (e0 + x * e1 + y * e2 + z * e3).dual()

def point_to_xyz(p):
    # for normalized points
    a,b,c,_ = p.values()
    return np.stack((-c, b, -a), axis=-1)

def trp(p0, p1):
    """ Translate from p0 to p1. """
    line = p0 & p1
    return ((0.5 * e0) ^ (line | e123)).exp()

bone_vertices = smpl['J'][:22]
bone_points = [xyz_to_point(v) for v in bone_vertices]
# station_joint_translation = [1 + 0*e01 + 0*e02 + 0*e03]
JOINT_TRANSLATORS = []
for child, parent in kinematic_child_parent:
    if child >= 22:
        break
    JOINT_TRANSLATORS.append(trp(bone_points[parent], bone_points[child]))

arr = np.array([t.values() for t in JOINT_TRANSLATORS])
JOINT_TRANSLATORS = alg.multivector(e=arr[...,0], e01=arr[...,1], e02=arr[...,2], e03=arr[...,3])

def chain_accumulate(rotors, translators):
    full_motors = [t * r for r, t in zip(rotors, translators)]
    for child, parent in kinematic_child_parent:
        if child >= 22:
            break
        if parent > 0:
            full_motors[child] = full_motors[parent] * full_motors[child]

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    # assert len(q.shape) == 3
    assert q.shape[-1] == 4
    dot_products = np.sum(q[1:] * q[:-1], axis=-1)
    mask = dot_products < 0
    num_flip = np.sum(mask)
    if num_flip:
        print(f'with {num_flip} sign flips')
        result = q.copy()
        mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
        result[1:][mask] *= -1
        return result
    return q


def root_to_translator(trans):
    t = -0.5 * trans
    return alg.multivector(e=np.ones_like(t[...,0]), e01=t[...,0], e02=t[...,1],e03=t[...,2])


# e, -e23, e13, -e12
def axis_angle_to_rotor(axis_angles):
    quats = expmap_to_quaternion(axis_angles)
    quats = qfix(quats)
    # cont_6d = quaternion_to_cont6d(quats)
    rotors = quat_to_rotor(quats)
    return rotors

def quat_to_rotor(quats):
    return alg.multivector(e=quats[...,0], e23=-quats[...,1], e13=quats[...,2], e12=-quats[...,3])


def rotor_to_quat(rotor):
    x, y, z, w = rotor.values()
    return np.stack([x, -w, z, -y], axis=-1)


def rotor_to_cont6d(rotor):
    quats = rotor_to_quat(rotor)
    return quaternion_to_cont6d(quats)


# body frame increments
# world frame increments is R2 * ~R1
def rotor_to_increment(rotors):
    return (~rotors[:-1]) * rotors[1:]


def chain_accumulate(rotors):
    full_motors = JOINT_TRANSLATORS * rotors
    for child, parent in kinematic_child_parent:
        if child >= 22:
            break
        if parent > 0:
            full_motors[:, child-1] = full_motors[:, parent-1] * full_motors[:, child-1]
    return full_motors


def quaternion_to_cont6d(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: np.ndarray of shape (..., 4) with real part first.

    Returns:
        np.ndarray of shape (..., 3, 3) containing rotation matrices.
    """
    r, i, j, k = np.moveaxis(quaternions, -1, 0)  # unpack last axis
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    two_s = np.where(np.isfinite(two_s), two_s, 0.0)  # handle zero norm

    return np.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            # two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            # two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            # 1 - two_s * (i * i + j * j),
        ]
    )
    # return o.reshape(quaternions.shape[:-1] + (6,))

# def load_to_rotor(src_path):
#     root_orient, root_trans, pose_body = load_amass(src_path)
#     body_rotors, body_6d = axis_angle_to_rotor(pose_body)
#     root_rotor, root_6d = axis_angle_to_rotor(root_orient)
#     root_translator = root_to_translator(root_trans)
#     return body_rotors, root_translator, root_rotor, body_6d, root_6d


def blade_exp(B, tol=1e-12):
    signature = (B ** 2).e
    normsq = abs(signature)
    if abs(signature) < tol:
        return 1 + B
    
    t = np.sqrt(normsq)
    b = B / t
    if signature < -tol:
        return np.cos(t) + np.sin(t) * b
    
    if signature > tol:
        return np.cosh(t) + np.sinh(t) * b
    

def simple_rotor_log(R, tol=1e-8):
    blade = R.grade(2)
    if abs(R.e - 1) < tol:
        return blade
    signature = (blade**2).e
    norm = np.sqrt(abs(signature))
    if signature < -tol:
        if np.any(R.e > 1 + 1e-12) or np.any(R.e < -1 - 1e-12):
            raise ValueError("Input to arccos is outside [-1,1] by more than tolerance.")
        return np.arccos(np.clip(R.e, -1.0, 1.0)) * blade/norm
    if signature > tol:
        return np.arccosh(R.e) * blade/norm
    

def translator_log(R):
    return R.grade(2)


def rotor_log(R, tol=1e-8):
    blade = R.grade(2)
    signature = (blade**2).e
    norm = np.sqrt(abs(signature))
    norm[norm < tol] = 1.0
    if np.any(R.e > 1 + 1e-12) or np.any(R.e < -1 - 1e-12):
        raise ValueError("Input to arccos is outside [-1,1] by more than tolerance.")
    return np.arccos(np.clip(R.e, -1.0, 1.0)) * blade/norm


def se3_split(B: MultiVector, tol=1e-8):
    inner = (B|~B).e
    inner[inner < tol] = 1.0
    r = 0.5*(B^~B)/inner
    return B*(1-r), B*r

def normsq(A: MultiVector):
    return abs(A.sp(A.reverse()).e)

def norm(A):
    return np.sqrt(normsq(A))

def normalize(A, tol=1e-8):
    n = norm(A)
    assert (n > tol).all(), f"zero norm {n[n <= tol]}"
    return A / n


def motor_split_log(R: MultiVector, tol=1e-8):
    # FIXME: when the tangent not well defined
    # assert (normsq(R.grade(2)) > tol).all(), "zero bivector"
    tangent = R.grade(2) / R.e
    T1, T0 = se3_split(tangent)
    norms = norm(T1)
    norms[norms < tol] = 1.0
    if np.any(R.e > 1 + 1e-12) or np.any(R.e < -1 - 1e-12):
        raise ValueError("Input to arccos is outside [-1,1] by more than tolerance.")
    return np.arccos(np.clip(R.e, -1.0, 1.0))*(T1/norms) + T0


# Welford's method for online mean and variance
# see Parallel algorithm of https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
def update_mean_var(mean, M2, count, new_data, axis=1):
    # new_data: shape (N, D)
    n = new_data.shape[axis]
    new_mean = np.nanmean(new_data, axis=axis)
    new_M2 = np.nansum((new_data - np.expand_dims(new_mean, axis=axis)) ** 2, axis=axis)
    delta = new_mean - mean
    total_count = count + n

    if count == 0:
        mean = new_mean
        M2 = new_M2
    else:
        mean = mean + delta * n / total_count
        M2 = M2 + new_M2 + (delta ** 2) * count * n / total_count

    return mean, M2, total_count


# for future use
def update_mean_var_cov(mean, M2, cov, count, new_data, axis=1):
    """
    Online update of mean, variance along `axis=1` and covariance along axis 0 using Welford's method.
    
    Parameters
    ----------
    mean : np.ndarray
        Running mean along axis 1 (shape: (D0,))
    M2 : np.ndarray
        Running sum of squares of deviations along axis 1 (shape: (D0,))
    cov : np.ndarray
        Running covariance matrix along axis 0 (shape: (D0, D0))
    count : int
        Number of samples seen so far
    new_data : np.ndarray
        New batch of data, shape (D0, D1)
    axis : int
        Axis along which to compute mean/variance (default=1)
        
    Returns
    -------
    mean, M2, cov, total_count
    """
    # Ensure axis=1 for row-wise mean/variance
    if axis != 1:
        new_data = np.moveaxis(new_data, axis, 1)
    
    n = new_data.shape[1]  # number of columns (for mean/var update)
    D0 = new_data.shape[0] # number of rows (for covariance update)
    
    # --- Update mean and M2 along axis=1 ---
    new_mean = np.nanmean(new_data, axis=axis)
    new_M2 = np.nansum((new_data - np.expand_dims(new_mean, axis=axis))**2, axis=axis)
    delta = new_mean - mean
    total_count = count + n
    
    if count == 0:
        mean = new_mean
        M2 = new_M2
        # Initialize covariance
        cov = np.zeros((D0, D0))
        for i in range(n):
            x = new_data[:, i:i+1]  # column vector
            cov += (x - new_data.mean(axis=1, keepdims=True)) @ (x - new_data.mean(axis=1, keepdims=True)).T
    else:
        mean = mean + delta * n / total_count
        M2 = M2 + new_M2 + (delta ** 2) * count * n / total_count
        
        # Update covariance along axis 0 using Welford-like update
        new_data_mean0 = np.nanmean(new_data, axis=1, keepdims=True)  # mean along axis 1
        for i in range(n):
            x = new_data[:, i:i+1]
            cov += (x - new_data_mean0) @ (x - new_data_mean0).T

        cov = cov / total_count  # normalize to get covariance
    
    return mean, M2, cov, total_count
