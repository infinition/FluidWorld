"""
FluidWorld Inspector -- Interactive visualization server.

Loads a FluidWorld checkpoint and serves a web dashboard to explore
latent representations, BeliefField dynamics, and prediction quality.

Supports both FluidWorldModelV2 (UCF-101, 3-channel) and legacy
FluidWorldModel (Moving MNIST, 1-channel).

Usage:
    python tools/inspector/server.py \
        --checkpoint checkpoints/phase1_pixel/model_step_8000.pt

    # Legacy MNIST mode
    python tools/inspector/server.py \
        --checkpoint checkpoints/old/model.pt --model v1 --in-channels 1
"""

import argparse
import base64
import io
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
_project = _here.parent.parent
sys.path.insert(0, str(_project))


def _load_model_class(version: str):
    """Import the right model class."""
    if version == "v2":
        from fluidworld.core.world_model_v2 import FluidWorldModelV2
        return FluidWorldModelV2
    else:
        from fluidworld.core.world_model import FluidWorldModel
        return FluidWorldModel


# ---------------------------------------------------------------------------
#  Model manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Thread-safe model loader and inference engine."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = None
        self.model_version = "v2"
        self.ckpt_info: dict = {}
        self.lock = threading.Lock()
        self.video_data = None       # Video sequences (T, C, H, W) or MNIST (T, N, H, W)
        self.labeled_data = None     # Labeled data for PCA scatter
        self._belief_state = None
        self.in_channels = 3

    def load_checkpoint(self, path: str, model_version: str = "v2",
                        in_channels: int = 3, d_model: int = 128,
                        stimulus_dim: int = 1) -> dict:
        with self.lock:
            self.model_version = model_version
            self.in_channels = in_channels
            ModelClass = _load_model_class(model_version)

            if model_version == "v2":
                model = ModelClass(
                    in_channels=in_channels,
                    d_model=d_model,
                ).to(self.device)
            else:
                model = ModelClass(
                    in_channels=in_channels,
                    d_model=d_model,
                    stimulus_dim=stimulus_dim,
                ).to(self.device)

            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state_dict, strict=False)
            model.requires_grad_(False)
            model.eval()

            self.model = model
            self._belief_state = None
            self.ckpt_info = {
                "path": str(path),
                "epoch": ckpt.get("epoch", "?"),
                "global_step": ckpt.get("global_step", "?"),
                "params_total": sum(p.numel() for p in model.parameters()),
                "params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "d_model": d_model,
                "in_channels": in_channels,
                "model_version": model_version,
            }
            return self.ckpt_info

    def load_video_data(self, path: str):
        """Load video data (.npy) for test sequences."""
        p = Path(path)
        if not p.exists():
            return
        raw = np.load(str(p))
        self.video_data = raw.astype(np.float32)
        if self.video_data.max() > 1.0:
            self.video_data = self.video_data / 255.0
        print(f"  Video data: {self.video_data.shape}")

    def load_labeled_data(self, path: str):
        """Load labeled data (.npz) for PCA scatter."""
        p = Path(path)
        if not p.exists():
            return
        data = np.load(str(p))
        images = data["images"].astype(np.float32)
        if images.max() > 1.0:
            images = images / 255.0
        if images.ndim == 3:
            images = images[:, np.newaxis, :, :]
        self.labeled_data = {
            "images": images,
            "labels": data["labels"].astype(int),
        }
        print(f"  Labeled data: {images.shape[0]} images, "
              f"{len(set(data['labels'].tolist()))} classes")

    def _prepare_input(self, image_np: np.ndarray) -> torch.Tensor:
        """Normalize input to (1, C, H, W) tensor."""
        if image_np.ndim == 2:
            # Grayscale HxW -> 1x1xHxW or 1x3xHxW
            image_np = image_np[np.newaxis, :, :]
            if self.in_channels == 3:
                image_np = np.repeat(image_np, 3, axis=0)
        elif image_np.ndim == 3:
            if image_np.shape[0] not in (1, 3):
                image_np = image_np.transpose(2, 0, 1)  # HWC -> CHW
            if image_np.shape[0] == 1 and self.in_channels == 3:
                image_np = np.repeat(image_np, 3, axis=0)
            elif image_np.shape[0] == 3 and self.in_channels == 1:
                image_np = image_np.mean(axis=0, keepdims=True)
        return torch.from_numpy(image_np).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_image(self, image_np: np.ndarray) -> dict:
        """Encode a single image, return features + pooled + heatmaps."""
        with self.lock:
            if self.model is None:
                return {"error": "No model loaded"}

            x = self._prepare_input(image_np)

            # Encode (v2 and v1 both have .encode())
            enc_out = self.model.encode(x)
            features = enc_out["features"]       # (1, d, H, W)
            pooled = features.mean(dim=(-2, -1)) # (1, d)

            steps_used = [info.get("steps_used", 0) for info in enc_out.get("info", [])]

            feat_np = features[0].cpu().numpy()
            pooled_np = pooled[0].cpu().numpy()

            C, Hf, Wf = feat_np.shape
            feat_flat = feat_np.reshape(C, -1)

            # PCA -> RGB
            mean = feat_flat.mean(axis=1, keepdims=True)
            centered = feat_flat - mean
            cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            top3 = eigvecs[:, -3:][:, ::-1]
            projected = (top3.T @ feat_flat).reshape(3, Hf, Wf)
            for c in range(3):
                pmin, pmax = projected[c].min(), projected[c].max()
                if pmax > pmin:
                    projected[c] = (projected[c] - pmin) / (pmax - pmin) * 255
                else:
                    projected[c] = 128
            pca_rgb = projected.astype(np.uint8).tolist()

            # Top 8 channel heatmaps
            channel_norms = np.linalg.norm(feat_np.reshape(C, -1), axis=1)
            top_channels = np.argsort(channel_norms)[-8:][::-1].tolist()
            channel_maps = {}
            for ch in top_channels:
                hmap = feat_np[ch]
                hmin, hmax = hmap.min(), hmap.max()
                if hmax > hmin:
                    hmap = (hmap - hmin) / (hmax - hmin)
                channel_maps[int(ch)] = hmap.tolist()

            per_dim_std = feat_np.reshape(C, -1).std(axis=1)

            return {
                "pooled": pooled_np.tolist(),
                "pca_rgb": pca_rgb,
                "channel_maps": channel_maps,
                "top_channels": top_channels,
                "feature_std": float(per_dim_std.mean()),
                "feature_std_min": float(per_dim_std.min()),
                "feature_std_max": float(per_dim_std.max()),
                "pde_steps": steps_used,
                "spatial_shape": [int(Hf), int(Wf)],
                "n_channels": int(C),
            }

    @torch.no_grad()
    def encode_sequence(self, frames: np.ndarray) -> dict:
        """Encode a sequence, return BeliefField evolution and predictions."""
        with self.lock:
            if self.model is None:
                return {"error": "No model loaded"}

            states = []
            predictions = []
            features_list = []
            self._belief_state = None

            for t in range(len(frames)):
                x = self._prepare_input(frames[t])
                enc_out = self.model.encode(x)
                z_t = enc_out["features"]
                z_pooled = z_t.mean(dim=(-2, -1))

                if self.model_version == "v2":
                    # V2: belief_field API
                    if self._belief_state is None:
                        self._belief_state = self.model.belief_field.init_state(
                            1, self.device, x.dtype)
                    else:
                        self._belief_state = self._belief_state.detach()

                    state_updated = self.model.belief_field.write(self._belief_state, z_t)
                    next_state = self.model.belief_field.evolve(state_updated)
                    z_pred = self.model.belief_field.read(next_state)
                    z_pred_proj = self.model.predictor(z_pred)
                    self._belief_state = next_state.detach()
                    bf_np = next_state[0].cpu().numpy()
                else:
                    # V1: stimulus-based API
                    stimulus = torch.zeros(1, self.model.stimulus_dim, device=self.device)
                    if self._belief_state is None:
                        self._belief_state = self.model.belief_field.init_state(
                            1, self.device, x.dtype)
                    else:
                        self._belief_state = self._belief_state.detach()

                    state_updated = self.model.belief_field.write(self._belief_state, z_t)
                    next_state = self.model.belief_field.evolve(state_updated, stimulus=stimulus)
                    z_pred = self.model.belief_field.read(next_state)
                    z_pred_proj = self.model.predictor(z_pred)
                    self._belief_state = next_state.detach()
                    bf_np = next_state[0].cpu().numpy()

                bf_energy = np.linalg.norm(bf_np, axis=0).tolist()

                states.append({
                    "energy_map": bf_energy,
                    "mean_energy": float(np.mean(np.abs(bf_np))),
                })
                features_list.append(z_pooled[0].cpu().numpy().tolist())

                if t > 0:
                    cos_sim = float(F.cosine_similarity(
                        z_pred_proj.mean(dim=(-2, -1)) if z_pred_proj.dim() == 4 else z_pred_proj,
                        z_pooled, dim=1).item())
                    mse_val = float(F.mse_loss(
                        z_pred_proj.mean(dim=(-2, -1)) if z_pred_proj.dim() == 4 else z_pred_proj,
                        z_pooled).item())
                    predictions.append({"cosine_sim": cos_sim, "mse": mse_val})

            return {
                "n_frames": len(frames),
                "states": states,
                "predictions": predictions,
                "features": features_list,
            }

    @torch.no_grad()
    def compute_pca(self, n_samples: int = 500) -> dict:
        """Compute PCA on labeled data."""
        with self.lock:
            if self.model is None:
                return {"error": "No model loaded"}
            if self.labeled_data is None:
                return {"error": "No labeled data loaded"}

            images = self.labeled_data["images"]
            labels = self.labeled_data["labels"]

            idx = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
            imgs = images[idx]
            lbls = labels[idx]

            all_z = []
            bs = 64
            for i in range(0, len(imgs), bs):
                batch_np = imgs[i:i+bs]
                # Handle channel mismatch
                if batch_np.shape[1] == 1 and self.in_channels == 3:
                    batch_np = np.repeat(batch_np, 3, axis=1)
                batch = torch.from_numpy(batch_np).float().to(self.device)
                # Resize if needed
                if batch.shape[-1] != 64:
                    batch = F.interpolate(batch, size=64, mode='bilinear', align_corners=False)
                z = self.model.encode(batch)["features"].mean(dim=(-2, -1))
                all_z.append(z.cpu().numpy())
            Z = np.concatenate(all_z, axis=0)

            Z_centered = Z - Z.mean(axis=0, keepdims=True)
            cov = Z_centered.T @ Z_centered / max(len(Z) - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            top2 = eigvecs[:, -2:][:, ::-1]
            coords = Z_centered @ top2

            for d in range(2):
                cmin, cmax = coords[:, d].min(), coords[:, d].max()
                if cmax > cmin:
                    coords[:, d] = (coords[:, d] - cmin) / (cmax - cmin)

            S = np.linalg.svd(Z_centered, compute_uv=False)
            S_norm = S / (S.sum() + 1e-8)
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
            eff_rank = float(np.exp(entropy))

            return {
                "points": [
                    {"x": float(coords[i, 0]), "y": float(coords[i, 1]),
                     "label": int(lbls[i])}
                    for i in range(len(coords))
                ],
                "effective_rank": eff_rank,
                "explained_var": [float(eigvals[-1]), float(eigvals[-2])],
                "n_samples": len(coords),
                "method": "PCA",
            }

    def get_test_frames(self, seq_idx: int = 0, n_frames: int = 10) -> dict:
        """Get test frames from loaded video data."""
        if self.video_data is None:
            return {"error": "No video data loaded"}

        data = self.video_data
        if data.ndim == 4:
            # MNIST format: (T, N, H, W)
            T, N, H, W = data.shape
            seq_idx = seq_idx % N
            frames = data[:min(n_frames, T), seq_idx]
            is_color = False
        elif data.ndim == 5:
            # Video format: (N, T, H, W, C) or (N, T, C, H, W)
            frames = data[seq_idx % len(data)][:n_frames]
            is_color = True
        else:
            return {"error": f"Unexpected data shape: {data.shape}"}

        encoded = []
        for f in frames:
            img_bytes = _array_to_png_bytes(f)
            encoded.append(base64.b64encode(img_bytes).decode("ascii"))

        return {
            "frames_b64": encoded,
            "n_frames": len(encoded),
            "seq_idx": seq_idx,
            "raw": frames.tolist(),
            "is_color": is_color,
        }

    def list_checkpoints(self) -> list:
        """List available checkpoints across all training runs."""
        ckpt_dir = _project / "checkpoints"
        if not ckpt_dir.exists():
            return []
        results = []
        for subdir in sorted(ckpt_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for p in sorted(subdir.glob("*.pt")):
                results.append({
                    "name": f"{subdir.name}/{p.name}",
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / 1e6, 1),
                    "run": subdir.name,
                })
        return results


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert array to PNG bytes. Handles grayscale (H,W) and color (H,W,3) or (C,H,W)."""
    import zlib, struct

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    pixels = (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    if pixels.ndim == 2:
        h, w = pixels.shape
        color_type = 0  # grayscale
        raw = b""
        for y in range(h):
            raw += b'\x00' + pixels[y].tobytes()
    else:
        h, w, _ = pixels.shape
        color_type = 2  # RGB
        raw = b""
        for y in range(h):
            raw += b'\x00' + pixels[y].tobytes()

    def _chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return sig + _chunk(b'IHDR', ihdr) + _chunk(b'IDAT', zlib.compress(raw)) + _chunk(b'IEND', b'')


# ---------------------------------------------------------------------------
#  HTTP Handler
# ---------------------------------------------------------------------------

class InspectorHandler(BaseHTTPRequestHandler):
    manager: ModelManager = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            html_path = _here / "index.html"
            body = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/checkpoint/info":
            self._json(self.manager.ckpt_info)

        elif self.path == "/api/checkpoints":
            self._json({"checkpoints": self.manager.list_checkpoints()})

        elif self.path.startswith("/api/test-images"):
            import urllib.parse
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            seq = int(params.get("seq", ["0"])[0])
            n = int(params.get("n", ["10"])[0])
            self._json(self.manager.get_test_frames(seq, n))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        body = self._read_body()

        if self.path == "/api/encode":
            data = json.loads(body)
            if "image" in data:
                arr = np.array(data["image"], dtype=np.float32)
            elif "frame_index" in data and "seq_index" in data:
                si = data["seq_index"]
                fi = data["frame_index"]
                if self.manager.video_data is not None:
                    vd = self.manager.video_data
                    if vd.ndim == 4:
                        T, N, H, W = vd.shape
                        arr = vd[fi % T, si % N]
                    else:
                        arr = vd[si % len(vd)][fi]
                else:
                    self._json({"error": "No video data loaded"}, 400)
                    return
            else:
                self._json({"error": "Provide 'image' or 'frame_index'+'seq_index'"}, 400)
                return
            result = self.manager.encode_image(arr)
            self._json(result)

        elif self.path == "/api/sequence":
            data = json.loads(body)
            seq_idx = data.get("seq_index", 0)
            n_frames = data.get("n_frames", 10)
            if self.manager.video_data is not None:
                vd = self.manager.video_data
                if vd.ndim == 4:
                    T, N, H, W = vd.shape
                    si = seq_idx % N
                    frames = vd[:min(n_frames, T), si]
                else:
                    frames = vd[seq_idx % len(vd)][:n_frames]
                result = self.manager.encode_sequence(frames)
                self._json(result)
            else:
                self._json({"error": "No video data loaded"}, 400)

        elif self.path == "/api/pca":
            data = json.loads(body) if body else {}
            n = data.get("n_samples", 500)
            result = self.manager.compute_pca(n)
            self._json(result)

        elif self.path == "/api/checkpoint/load":
            data = json.loads(body)
            path = data.get("path", "")
            version = data.get("model_version", "v2")
            try:
                info = self.manager.load_checkpoint(
                    path,
                    model_version=version,
                    in_channels=data.get("in_channels", 3),
                    d_model=data.get("d_model", 128),
                    stimulus_dim=data.get("stimulus_dim", 1),
                )
                self._json(info)
            except Exception as e:
                self._json({"error": str(e)}, 500)

        elif self.path == "/api/predict":
            data = json.loads(body)
            seq_idx = data.get("seq_index", 0)
            frame_a = data.get("frame_a", 0)
            frame_b = data.get("frame_b", 1)
            if self.manager.video_data is not None:
                vd = self.manager.video_data
                if vd.ndim == 4:
                    T, N, H, W = vd.shape
                    si = seq_idx % N
                    fA = vd[frame_a % T, si]
                    fB = vd[frame_b % T, si]
                else:
                    fA = vd[seq_idx % len(vd)][frame_a]
                    fB = vd[seq_idx % len(vd)][frame_b]
                res_a = self.manager.encode_image(fA)
                res_b = self.manager.encode_image(fB)
                zA = np.array(res_a["pooled"])
                zB = np.array(res_b["pooled"])
                cos = float(np.dot(zA, zB) / (np.linalg.norm(zA) * np.linalg.norm(zB) + 1e-8))
                mse = float(np.mean((zA - zB) ** 2))
                self._json({
                    "frame_a": res_a,
                    "frame_b": res_b,
                    "cosine_similarity": cos,
                    "mse": mse,
                })
            else:
                self._json({"error": "No video data loaded"}, 400)

        else:
            self.send_response(404)
            self.end_headers()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FluidWorld Inspector")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Checkpoint .pt file to load at startup")
    parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2"],
                        help="Model version: v2 (default, UCF-101) or v1 (legacy MNIST)")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--in-channels", type=int, default=3,
                        help="Input channels: 3 for RGB (UCF-101), 1 for grayscale (MNIST)")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--stimulus-dim", type=int, default=1,
                        help="Stimulus dim (v1 only)")
    parser.add_argument("--video-data", type=str, default=None,
                        help="Path to .npy video data for test sequences")
    parser.add_argument("--labeled-data", type=str, default=None,
                        help="Path to .npz labeled data for PCA scatter")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  FluidWorld Inspector")
    print("=" * 60)
    print(f"  Model: {args.model} | Channels: {args.in_channels} | Device: {args.device}")

    manager = ModelManager(device=args.device)

    # Auto-detect and load data
    data_dir = _project / "data"

    if args.video_data:
        manager.load_video_data(args.video_data)
    else:
        # Try common paths
        for p in ["mnist_test_seq.npy", "mnist_moving.npy"]:
            if (data_dir / p).exists():
                manager.load_video_data(str(data_dir / p))
                break

    if args.labeled_data:
        manager.load_labeled_data(args.labeled_data)
    elif (data_dir / "mnist_labeled.npz").exists():
        manager.load_labeled_data(str(data_dir / "mnist_labeled.npz"))

    # Load checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _project / ckpt_path
        info = manager.load_checkpoint(
            str(ckpt_path),
            model_version=args.model,
            in_channels=args.in_channels,
            d_model=args.d_model,
            stimulus_dim=args.stimulus_dim,
        )
        print(f"  Checkpoint: {info['path']}")
        print(f"  Step: {info['global_step']}, Epoch: {info['epoch']}")
        print(f"  Params: {info['params_total']:,}")
    else:
        ckpts = manager.list_checkpoints()
        if ckpts:
            # Pick latest from phase1_pixel by default
            pixel_ckpts = [c for c in ckpts if "phase1_pixel" in c["run"]]
            latest = pixel_ckpts[-1] if pixel_ckpts else ckpts[-1]
            try:
                info = manager.load_checkpoint(
                    latest["path"],
                    model_version=args.model,
                    in_channels=args.in_channels,
                    d_model=args.d_model,
                    stimulus_dim=args.stimulus_dim,
                )
                print(f"  Auto-loaded: {latest['name']}")
            except Exception as e:
                print(f"  WARNING: Failed to auto-load {latest['name']}: {e}")
        else:
            print("  WARNING: No checkpoint found. Use the dashboard to load one.")

    InspectorHandler.manager = manager

    server = HTTPServer(("0.0.0.0", args.port), InspectorHandler)
    print(f"\n  Dashboard: http://localhost:{args.port}")
    print(f"  API:       http://localhost:{args.port}/api/checkpoint/info")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
