from amass_motor_utils import *


EXFPS = 20
def load_amass(src_path, start_frame=0, end_frame=-1):
    motion = np.load(src_path, allow_pickle=True)
    fps = 0
    try:
        fps = motion['mocap_framerate']
        # frame_number = bdata['trans'].shape[0]
    except:
#         print(list(bdata.keys()))
        return None, fps
    down_sample = int(fps / EXFPS)
#     print(frame_number)
#     print(fps)
    root_orient = motion['poses'][::down_sample, :3].reshape(-1, 1, 3) # 1 root joint
    pose_body = motion['poses'][::down_sample, 3:66].reshape(-1, 21, 3) # 21 child joints
    root_trans = motion['trans'][::down_sample,...].reshape(-1, 1, 3)
    body_rotors = axis_angle_to_rotor(pose_body[start_frame:end_frame])
    root_rotor0 = axis_angle_to_rotor(root_orient[start_frame:end_frame])
    root_translator0 = root_to_translator(root_trans[start_frame:end_frame])
    return (body_rotors, root_rotor0, root_translator0), fps


def process_rotors(rotors, stat_map=None):
    body_rotors, root_rotor0, root_translator0 = rotors
    root_motor0 = root_translator0 * root_rotor0

    # floor as a moving plane relative to root's frame
    floor = (~root_motor0) >> e3
    # height = floor.e0

    # initiate to root's frame
    root_motor = ~root_motor0[0] * root_motor0
    root_translator = ~root_rotor0[0] >> (~root_translator0[0] * root_translator0)
    root_rotor = ~root_rotor0[0] * root_rotor0

    # velocitiy
    # Maybe ignore root[0]
    velocity_body_rotors = rotor_to_increment(body_rotors)
    velocity_root_motor = rotor_to_increment(root_motor)
    velocity_root_translator = (~root_rotor[:-1]) >> rotor_to_increment(root_translator)
    velocity_root_rotor = rotor_to_increment(root_rotor)
    # velocity_root_rotor = (root_translator[:-1] * ~root_translator[0]) >> rotor_to_increment(root_rotor)

    # acceleration
    acceleration_body_rotors = rotor_to_increment(velocity_body_rotors)
    acceleration_root_motor = rotor_to_increment(velocity_root_motor)
    acceleration_root_translator = (~velocity_root_rotor[:-1]) >> rotor_to_increment(velocity_root_translator)
    acceleration_root_rotor = rotor_to_increment(velocity_root_rotor)
    # acceleration_root_rotor = velocity_root_translator[:-1] >> rotor_to_increment(velocity_root_rotor)


    # chained for interaction with geometric objects
    chained_body_rotors = chain_accumulate(body_rotors)
    
    # a map to track all motors
    motor_map = dict(
        root_motor=root_motor,
        velocity_root_motor=velocity_root_motor,
        acceleration_root_motor=acceleration_root_motor,
        chained_body_rotors=chained_body_rotors,
    )

    rotor_map = dict(
        body_rotors=body_rotors,
        root_rotor=root_rotor,
        velocity_body_rotors=velocity_body_rotors,
        velocity_root_rotor=velocity_root_rotor,
        acceleration_body_rotors=acceleration_body_rotors,
        acceleration_root_rotor=acceleration_root_rotor,
    )

    rotor_map_6d = dict(
        body_rotors=body_rotors,
        root_rotor=root_rotor,
        # chained_body_rotors=chained_body_rotors,
        velocity_body_rotors=velocity_body_rotors,
        velocity_root_rotor=velocity_root_rotor,
        acceleration_body_rotors=acceleration_body_rotors,
        acceleration_root_rotor=acceleration_root_rotor,
    )

    translator_map = dict(
        root_translator=root_translator,
        velocity_root_translator=velocity_root_translator,
        acceleration_root_translator=acceleration_root_translator,
    )

    geo_object_map = dict(
        floor=floor,
    )

    # logarithms
    motor_lie_map = {"log_" + k: motor_split_log(v) for k, v in motor_map.items()}
    rotor_lie_map = {"log_" + k: rotor_log(v) for k, v in rotor_map.items()}
    translator_lie_map = {"log_" + k: translator_log(v) for k, v in translator_map.items()}
    rot_trans_lie_map = {k.replace("_translator", ""): v + rotor_lie_map[k.replace("translator", "rotor")] for k, v in translator_lie_map.items()}
    
    # all values
    all_map = {**motor_map, **rotor_map, **translator_map, **motor_lie_map, **rotor_lie_map, **translator_lie_map, **geo_object_map, **rot_trans_lie_map}

    key_map = {k: v.keys() for k, v in all_map.items()}
    value_map = {k: np.stack(v.values()) for k, v in all_map.items()}
    # FIXME swapping left/right, should enforce an equivariance shortcut of NN
    value_map_m = {k: np.stack((e1 >> v).values()) for k, v in all_map.items()}

    # cont_6d
    cont_6d_map = {k+"_6d": rotor_to_cont6d(v) for k, v in rotor_map_6d.items()}
    cont_6d_map_m = {k+"_6d": rotor_to_cont6d(e1 >> v) for k, v in rotor_map_6d.items()}
    value_map = {**value_map, **cont_6d_map}
    value_map_m = {**value_map_m, **cont_6d_map_m}

    # first map
    first_map = {k+"_0": v[0:1] for k, v in value_map.items()}
    first_map_m = {k+"_0": v[0:1] for k, v in value_map_m.items()}
    value_map = {**value_map, **first_map}
    value_map_m = {**value_map_m, **first_map_m}
    
    # statistics
    if stat_map is None:
        stat_map = {k: (np.zeros((v.shape[0], v.shape[2])), np.zeros((v.shape[0], v.shape[2])), 0) for k, v in value_map.items()}
    stat_map = {k: update_mean_var(*stat_map[k], v) for k, v in value_map.items()}
    return value_map, value_map_m, stat_map, key_map

