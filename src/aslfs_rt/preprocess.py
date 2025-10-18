
import numpy as np, cv2, mediapipe as mp
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose

POSE_KEEP = [11, 12, 13, 14, 15, 16]
LH_IDX = list(range(21))
RH_IDX = list(range(21))
EXPECTED_IN_FEAT = (len(POSE_KEEP) + len(LH_IDX) + len(RH_IDX)) * 3

_pose_ctx = None
_hands_ctx = None

def init_mediapipe(max_hands=2, min_det=0.5, min_track=0.5):
    global _pose_ctx, _hands_ctx
    if _pose_ctx is None:
        _pose_ctx = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
    if _hands_ctx is None:
        _hands_ctx = mp_hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                    min_detection_confidence=min_det, min_tracking_confidence=min_track)

def _safe_arr3(v):
    try:
        return np.array([float(v.x), float(v.y), float(getattr(v, 'z', 0.0))], dtype=np.float32)
    except Exception:
        return None

def extract_landmarks(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    pose_results = _pose_ctx.process(img)
    hands_results = _hands_ctx.process(img)
    img.flags.writeable = True

    out = {'pose': {}, 'left': {}, 'right': {}}
    if pose_results.pose_landmarks:
        plms = pose_results.pose_landmarks.landmark
        for i in POSE_KEEP:
            v = _safe_arr3(plms[i])
            if v is not None: out['pose'][i] = v

    if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hand_lms, handed in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
            label = handed.classification[0].label.lower()
            pts = hand_lms.landmark
            for j in range(21):
                v = _safe_arr3(pts[j])
                if v is not None:
                    out[label][j] = v
    return out

def normalize_frame(det):
    have_pose = (11 in det['pose']) and (12 in det['pose'])
    if have_pose:
        lsh, rsh = det['pose'][11], det['pose'][12]
        center = (lsh + rsh) / 2.0
        scale = np.linalg.norm(lsh - rsh)
    else:
        lw = det['left'].get(0, None)
        rw = det['right'].get(0, None)
        if lw is None and rw is None:
            center = np.zeros(3, dtype=np.float32); scale = 1e-6
        else:
            if lw is None: lw = rw
            if rw is None: rw = lw
            center = (lw + rw) / 2.0
            scale  = np.linalg.norm(lw - rw)
    scale = max(scale, 1e-6)

    feats = []
    for i in POSE_KEEP:
        xyz = det['pose'].get(i, np.zeros(3, dtype=np.float32))
        n = (xyz - center) / scale; feats.extend(n.tolist())
    for j in LH_IDX:
        xyz = det['left'].get(j, np.zeros(3, dtype=np.float32))
        n = (xyz - center) / scale; feats.extend(n.tolist())
    for j in RH_IDX:
        xyz = det['right'].get(j, np.zeros(3, dtype=np.float32))
        n = (xyz - center) / scale; feats.extend(n.tolist())

    x = np.asarray(feats, dtype=np.float32)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(x, -1e6, 1e6, out=x)
    return x
