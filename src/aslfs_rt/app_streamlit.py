
import streamlit as st, cv2
from .config import RTConfig
from .model import load_bundle
from .preprocess import init_mediapipe, extract_landmarks, normalize_frame, EXPECTED_IN_FEAT
from .decoder import RealTimeDecoder

def run(bundle="."):
    st.set_page_config(page_title="ASL Fingerspelling (Phase 3)", layout="wide")
    st.title("ASL Fingerspelling — Real-Time Demo (Streamlit)")
    source = st.sidebar.selectbox("Source", ["Webcam 0"], index=0)
    flip = st.sidebar.checkbox("Flip horizontally", True)
    cfg = RTConfig()

    model, char2idx, idx2char, cfg_dict, device, in_feat, vocab_size = load_bundle(bundle)
    assert in_feat == EXPECTED_IN_FEAT, f"Feature mismatch: model expects {in_feat}, pipeline provides {EXPECTED_IN_FEAT}"

    init_mediapipe(max_hands=cfg.mp_max_hands, min_det=cfg.mp_min_det_conf, min_track=cfg.mp_min_track_conf)
    dec = RealTimeDecoder(model, char2idx, idx2char, device, in_feat, vocab_size,
                          window=cfg.window, smooth_k=cfg.smooth_k,
                          commit_thresh=cfg.commit_thresh, commit_patience=cfg.commit_patience)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam not available.")
        return

    ph = st.empty(); dec.reset_text()
    while True:
        ok, frame = cap.read()
        if not ok: break
        if flip: frame = cv2.flip(frame, 1)
        det = extract_landmarks(frame)
        x = normalize_frame(det)
        dec.push_frame(x)
        _, text = dec.step()
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
        ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if st.button("Stop"): break
    cap.release()

if __name__ == "__main__":
    run(".")
