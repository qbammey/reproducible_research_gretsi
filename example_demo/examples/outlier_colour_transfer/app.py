# app.py
import io
import numpy as np
import streamlit as st
from PIL import Image
import run  # your run.py in the same folder

st.set_page_config(page_title="OT Color Transfer (tutorial)", layout="centered")
st.title("🎨 OT Color Transfer (tutorial)")

# uploads
base_file = st.file_uploader("Base image (to stylize)", type=["jpg", "jpeg", "png"])
target_file = st.file_uploader("Target image (style source)", type=["jpg", "jpeg", "png"])

# parameters (defaults match run.py)
col1, col2 = st.columns(2)
with col1:
    downscale = st.number_input("Downscale size", min_value=8, max_value=256, value=50, step=1)
    metric = st.selectbox("Metric", ["l2", "l1"], index=0)
with col2:
    reg = st.number_input("Entropic reg (0 = EMD)", min_value=0.0, value=0.0, step=0.01, format="%.4f")
    sinkhorn_max_iter = st.number_input("Sinkhorn max iters", min_value=100, max_value=20000, value=1000, step=100)

run_btn = st.button("Run")

if run_btn:
    if base_file is None or target_file is None:
        st.warning("Please upload both images.")
    else:
        # load to [0,1] float RGB
        base_img = np.asarray(Image.open(base_file).convert("RGB"), dtype=np.float64) / 255.0
        target_img = np.asarray(Image.open(target_file).convert("RGB"), dtype=np.float64) / 255.0

        styl = run.color_transfer_ot(
            base_img=base_img,
            target_img=target_img,
            downscale=int(downscale),
            metric=metric,
            reg=float(reg),
            sinkhorn_max_iter=int(sinkhorn_max_iter),
        )

        st.subheader("Result")
        st.image(styl, use_container_width=True)

        # quick download
        buf = io.BytesIO()
        Image.fromarray((np.clip(styl, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
        st.download_button("Download PNG", data=buf.getvalue(), file_name="stylized.png", mime="image/png")

