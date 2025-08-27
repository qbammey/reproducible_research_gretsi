import io
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path

# --- Import your research function ---
import run

# --- Define the directory of the current script to find example images ---
# This is the key to making file paths robust. `Path(__file__)` gets the path
# to this .py file, and `.parent` gets the directory it's in.
SCRIPT_DIR = Path(__file__).parent

# --- Page Configuration ---
st.set_page_config(
    page_title="OT Color Transfer (Advanced)",
    layout="wide"
)

# --- Caching the Core Function ---
@st.cache_data
def cached_color_transfer(base_img_bytes, target_img_bytes, downscale, metric, reg, sinkhorn_max_iter):
    """
    A wrapper around the core function to make it cacheable.
    We pass image bytes instead of numpy arrays because arrays are not hashable.
    """
    base_img = np.asarray(Image.open(io.BytesIO(base_img_bytes)).convert("RGB"), dtype=np.float64) / 255.0
    target_img = np.asarray(Image.open(io.BytesIO(target_img_bytes)).convert("RGB"), dtype=np.float64) / 255.0
    
    return run.color_transfer_ot(
        base_img=base_img,
        target_img=target_img,
        downscale=int(downscale),
        metric=metric,
        reg=float(reg),
        sinkhorn_max_iter=int(sinkhorn_max_iter),
    )

# --- Helper function to load example images using pathlib ---
def load_example_image(image_filename):
    """Loads an image from a path relative to the script's directory."""
    # Construct the full, absolute path to the image
    image_path = SCRIPT_DIR / image_filename
    with open(image_path, "rb") as f:
        return f.read()

# --- Main App UI ---
st.title("🎨 Advanced OT Color Transfer")
st.write("This demo showcases a more robust Streamlit app with caching, state management, and error handling.")

# --- Sidebar for Inputs and Parameters ---
with st.sidebar:
    st.header("Inputs")
    
    # 2. Update the example options with your filenames and names
    example_options = {
        "Example 1": ("B2.jpg", "A1.jpg"),
        "Example 2": ("B2.jpg", "C1.jpg"),
    }
    
    source_type = st.radio("Choose image source:", ["Upload your own", "Use an example"], index=1) # Default to example

    if source_type == "Upload your own":
        base_file = st.file_uploader("Base image (to stylize)", type=["jpg", "jpeg", "png"])
        target_file = st.file_uploader("Target image (style source)", type=["jpg", "jpeg", "png"])
    else:
        selected_example = st.selectbox("Select an example:", list(example_options.keys()))
        base_filename, target_filename = example_options[selected_example]
        # Use our robust helper function to load the images
        base_file = load_example_image(base_filename)
        target_file = load_example_image(target_filename)

    st.header("Parameters")
    with st.expander("Algorithm Controls", expanded=True):
        downscale = st.number_input("Downscale size", 8, 256, 50, 1)
        metric = st.selectbox("Metric", ["l2", "l1"], 0)
        reg = st.number_input("Entropic reg (0 = EMD)", 0.0, 1.0, 0.0, 0.01, format="%.4f")
        if reg > 0:
            sinkhorn_max_iter = st.number_input("Sinkhorn max iters", 100, 20000, 1000, 100)
        else:
            sinkhorn_max_iter = 1000

    run_btn = st.button("Run Style Transfer", type="primary")

# --- Main Area for Displaying Images ---
col1, col2 = st.columns(2)
if base_file is not None:
    col1.image(base_file, caption="Base Image", use_container_width=True)
if target_file is not None:
    col2.image(target_file, caption="Target Style", use_container_width=True)
    
if run_btn:
    if base_file is None or target_file is None:
        st.warning("Please provide both a base and a target image.")
    else:
        base_bytes = base_file.getvalue() if hasattr(base_file, 'getvalue') else base_file
        target_bytes = target_file.getvalue() if hasattr(target_file, 'getvalue') else target_file
        
        with st.spinner('Stylizing image... this may take a moment.'):
            try:
                stylized_image = cached_color_transfer(
                    base_bytes, target_bytes, downscale, metric, reg, sinkhorn_max_iter
                )
                st.session_state['last_result'] = stylized_image
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                if 'last_result' in st.session_state:
                    del st.session_state['last_result']

if 'last_result' in st.session_state:
    st.divider()
    st.header("Result")
    result_image = st.session_state['last_result']
    st.image(result_image, caption="Stylized Result", use_column_width=True)

    buf = io.BytesIO()
    Image.fromarray((np.clip(result_image, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
    st.download_button(
        "Download Result",
        data=buf.getvalue(),
        file_name="stylized_result.png",
        mime="image/png"
    )
