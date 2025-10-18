
import argparse, time, cv2
from .config import RTConfig
from .model import load_bundle
from .preprocess import init_mediapipe, extract_landmarks, normalize_frame, EXPECTED_IN_FEAT
from .decoder import RealTimeDecoder

def main():
    ap = argparse.ArgumentParser(description="ASL Fingerspelling Real-Time (CLI)")
    ap.add_argument("--bundle", type=str, default=".", help="Path to extracted bundle root")
    ap.add_argument("--source", type=str, default="0", help="0=webcam, or path to video file")
    ap.add_argument("--flip", action="store_true", help="Flip frame horizontally")
    ap.add_argument("--save", type=str, default="", help="Optional path to save output video")
    args = ap.parse_args()

    model, char2idx, idx2char, cfg_dict, device, in_feat, vocab_size = load_bundle(args.bundle)
    cfg = RTConfig()

    assert in_feat == EXPECTED_IN_FEAT, f"Feature mismatch: model expects {in_feat}, pipeline provides {EXPECTED_IN_FEAT}"

    init_mediapipe(max_hands=cfg.mp_max_hands, min_det=cfg.mp_min_det_conf, min_track=cfg.mp_min_track_conf)
    dec = RealTimeDecoder(model, char2idx, idx2char, device, in_feat, vocab_size,
                          window=cfg.window, smooth_k=cfg.smooth_k,
                          commit_thresh=cfg.commit_thresh, commit_patience=cfg.commit_patience)

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Could not open source. Try --source=/path/to/video.mp4")

    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)) or cfg.target_fps)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w,h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    last = time.time()
    frame_interval = 1.0 / max(1, cfg.target_fps)
    dec.reset_text()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if args.flip: frame = cv2.flip(frame, 1)

            det = extract_landmarks(frame)
            x = normalize_frame(det)
            dec.push_frame(x)
            _, text = dec.step()

            cv2.putText(frame, text, (20, 40), font, 1.0, (0,255,0), 2, cv2.LINE_AA)

            now = time.time()
            if now - last >= frame_interval:
                last = now
                cv2.imshow("ASL Fingerspelling (CLI)", frame)
            if writer: writer.write(frame)
            if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
