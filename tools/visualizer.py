import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "phase1", "model_step_2500.pt")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mnist_test_seq.npy")

# Add project root to path for src imports
sys.path.append(PROJECT_ROOT)

from fluidworld.core.world_model import FluidWorldModel


def resolve_project_path(path):
    raw_path = os.path.expandvars(os.path.expanduser(path.strip()))
    normalized = os.path.normpath(raw_path)

    if os.path.isabs(normalized) and os.path.exists(normalized):
        return normalized

    drive, _ = os.path.splitdrive(normalized)
    if (raw_path.startswith("/") or raw_path.startswith("\\")) and not drive:
        return os.path.normpath(os.path.join(PROJECT_ROOT, raw_path.lstrip("/\\")))

    if not os.path.isabs(normalized):
        return os.path.normpath(os.path.join(PROJECT_ROOT, normalized))

    return normalized

class FluidInspector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset = None
        self.current_ckpt = None
        self.num_sequences = 0
        self.num_frames = 0

    def get_index_bounds(self):
        if self.dataset is None:
            return 0, 0
        return max(0, self.dataset.shape[0] - 1), max(0, self.dataset.shape[1] - 1)

    def load_model(self, ckpt_path):
        resolved_ckpt = resolve_project_path(ckpt_path)
        if self.current_ckpt == resolved_ckpt and self.model is not None:
            return "Model already loaded."
        
        try:
            # Default parameters
            self.model = FluidWorldModel(in_channels=1, d_model=128, stimulus_dim=1).to(self.device)
            state_dict = torch.load(resolved_ckpt, map_location=self.device)
            
            # Handle old/new checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.current_ckpt = resolved_ckpt
            return f"Checkpoint loaded: {os.path.basename(resolved_ckpt)} ({sum(p.numel() for p in self.model.parameters())} params)"
        except Exception as e:
            return f"Loading error: {str(e)}"

    def load_data(self, data_path):
        try:
            # Expected shape: (N, T, H, W).
            resolved_data = resolve_project_path(data_path)
            data = np.load(resolved_data, allow_pickle=False)

            if isinstance(data, np.lib.npyio.NpzFile):
                if len(data.files) == 0:
                    return "Data loading error: empty NPZ archive."
                data = data[data.files[0]]

            if data.ndim != 4:
                return f"Unexpected format: {data.shape}. Expected: 4D (N,T,H,W) or (T,N,H,W)."

            transposed = False
            # Moving-MNIST is often stored as (T, N, H, W), e.g. (20, 10000, 64, 64).
            if data.shape[0] <= 64 and data.shape[1] > data.shape[0]:
                data = np.transpose(data, (1, 0, 2, 3))
                transposed = True

            data = data.astype(np.float32, copy=False)
            if data.max() > 1.0:
                data = data / 255.0

            self.dataset = torch.from_numpy(data)
            self.num_sequences = int(self.dataset.shape[0])
            self.num_frames = int(self.dataset.shape[1])

            note = " (T,N,H,W format detected and transposed)" if transposed else ""
            return f"Dataset loaded: {self.num_sequences} sequences of {self.num_frames} frames.{note}"
        except Exception as e:
            return f"Data loading error: {str(e)}"

    def inspect_frame(self, seq_idx, frame_idx):
        if self.model is None or self.dataset is None:
            return None, "Please load the model and data first."

        seq_max, frame_max = self.get_index_bounds()
        seq_raw = int(seq_idx)
        frame_raw = int(frame_idx)
        seq_idx = min(max(seq_raw, 0), seq_max)
        frame_idx = min(max(frame_raw, 0), frame_max)
        was_clamped = (seq_raw != seq_idx) or (frame_raw != frame_idx)

        # Prepare image (B, C, H, W)
        img = self.dataset[seq_idx, frame_idx].unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. Encoding and feature extraction
            encode_out = self.model.encode(img)
            features = encode_out["features"] # (B, d_model, H', W')
            
            # 2. If BeliefField is accessible, attempt to read its state
            # Simulate a null stimulus for inspection
            stimulus = torch.zeros(1, 1).to(self.device) 
            # Note: depending on exact implementation, internal belief_field state could be extracted here
            
        # --- VISUALIZATION ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original Image
        axes[0].imshow(img[0, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title(f"Input: Sequence {seq_idx}, Frame {frame_idx}")
        axes[0].axis('off')
        
        # 2. Latent Features (Channel-averaged global activation)
        # Assumes features shape is (1, C, H, W) or (1, N, C)
        if len(features.shape) == 4:
            feat_map = features[0].mean(dim=0).cpu().numpy()
        else:
            # If flattened (B, N, C), attempt to reshape (e.g. 8x8)
            seq_len = features.shape[1]
            h = int(np.sqrt(seq_len))
            feat_map = features[0].mean(dim=-1).view(h, h).cpu().numpy()
            
        im = axes[1].imshow(feat_map, cmap='magma')
        axes[1].set_title("Latent Features (Activity Map)")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Energy / Variance per dimension (to detect collapse)
        # Compute feature variance for this image
        if len(features.shape) == 4:
            feat_vars = features[0].view(features.shape[1], -1).var(dim=1).cpu().numpy()
        else:
            feat_vars = features[0].var(dim=0).cpu().numpy()
            
        axes[2].bar(range(len(feat_vars)), feat_vars, color='royalblue', alpha=0.7)
        axes[2].set_title("Variance per Dimension (Anti-Collapse)")
        axes[2].set_xlabel("Dimension (d_model)")
        axes[2].set_ylabel("Variance")

        plt.tight_layout()
        
        # Convert plot to image for Gradio
        fig.canvas.draw()
        if hasattr(fig.canvas, "buffer_rgba"):
            rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image_from_plot = rgba[..., :3].copy()
        else:
            width, height = fig.canvas.get_width_height()
            argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
            image_from_plot = argb[..., 1:4].copy()
        plt.close(fig)

        clamp_msg = ""
        if was_clamped:
            clamp_msg = (
                f" (indices clamped to seq={seq_idx}/{seq_max}, frame={frame_idx}/{frame_max})"
            )
        return image_from_plot, f"Generation successful.{clamp_msg}"

inspector = FluidInspector()

# Gradio Interface
with gr.Blocks(title="FluidWorld - PDE World Model Inspector", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# FluidWorld - Scientific Latent Space Visualizer")
    gr.Markdown("Inspect internal representations (PDE, Belief Field, Features) of your model in real time.")

    with gr.Row():
        with gr.Column(scale=1):
            ckpt_input = gr.Textbox(label="Checkpoint Path (.pt)", value=DEFAULT_CKPT_PATH)
            data_input = gr.Textbox(label="Data Path (.npy/.npz)", value=DEFAULT_DATA_PATH)
            load_btn = gr.Button("Load Model & Data", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Environment Controls")
            seq_slider = gr.Slider(minimum=0, maximum=19, step=1, label="Sequence Index", value=0)
            frame_slider = gr.Slider(minimum=0, maximum=19, step=1, label="Frame Index", value=0)
            inspect_btn = gr.Button("Inspect Dynamics", variant="secondary")

        with gr.Column(scale=3):
            output_img = gr.Image(label="Inspection Dashboard", type="numpy")
            log_output = gr.Textbox(label="Analysis", interactive=False)

    def on_load(ckpt, data):
        msg1 = inspector.load_model(ckpt)
        msg2 = inspector.load_data(data)
        seq_max, frame_max = inspector.get_index_bounds()
        return (
            f"{msg1}\n{msg2}",
            gr.update(maximum=seq_max, value=0),
            gr.update(maximum=frame_max, value=0),
        )

    load_btn.click(fn=on_load, inputs=[ckpt_input, data_input], outputs=[status_text, seq_slider, frame_slider])
    
    # Update image on button click or slider change
    inputs_inspect = [seq_slider, frame_slider]
    inspect_btn.click(fn=inspector.inspect_frame, inputs=inputs_inspect, outputs=[output_img, log_output])
    seq_slider.change(fn=inspector.inspect_frame, inputs=inputs_inspect, outputs=[output_img, log_output])
    frame_slider.change(fn=inspector.inspect_frame, inputs=inputs_inspect, outputs=[output_img, log_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
