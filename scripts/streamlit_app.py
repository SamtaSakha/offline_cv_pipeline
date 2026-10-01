import sys
import json
import tempfile
from pathlib import Path
from PIL import Image

import streamlit as st

# Ensure we can import from src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bhiv_cv import CVPipeline
from bhiv_cv.gallery import Gallery

st.set_page_config(page_title="BHIV Face Recognition", page_icon="🕵️", layout="centered")

# Initialize pipeline once
@st.cache_resource
def get_pipeline():
    pipeline = CVPipeline()
    
    # Try to load existing gallery or auto-enroll the sample data so it just works
    gallery_path = ROOT / "data" / "gallery_store.json"
    if gallery_path.exists():
        pipeline.gallery = Gallery.load(str(gallery_path))
    else:
        # Pre-enroll from data/gallery if they exist
        sample_gallery_dir = ROOT / "data" / "gallery"
        if sample_gallery_dir.exists():
            for identity_dir in sample_gallery_dir.iterdir():
                if identity_dir.is_dir():
                    identity = identity_dir.name
                    for img_file in identity_dir.glob("*.png"):
                        try:
                            pipeline.enroll(str(img_file), identity)
                            break # Just one per identity is enough for demo
                        except Exception:
                            pass
    return pipeline

pipeline = get_pipeline()

# UI Layout
st.title("┌──────────────────────────────┐")
st.title("│ &nbsp; &nbsp; &nbsp; &nbsp; BHIV Face Recognition  &nbsp; &nbsp; &nbsp; │")
st.title("├──────────────────────────────┤")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image and display it
    image = Image.open(uploaded_file)
    image = Image.open(uploaded_file)

    # Save to temp file because pipeline.identify expects a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Run identification
        result = pipeline.identify(tmp_path)
        
        # Create a beautiful two-column layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Probe Image", use_container_width=True)
            
        with col2:
            st.markdown("### Analysis Results")
            if result.error:
                st.error("❌ Face Detection Failed")
                st.warning(result.error)
            else:
                if result.match:
                    st.success("✅ Known Identity Verified")
                else:
                    st.error("⚠️ Unknown Identity")
                
                # Use Streamlit metrics for a beautiful dashboard look
                st.metric(label="Identity Match", value=result.identity or "Unknown")
                
                mcol1, mcol2 = st.columns(2)
                mcol1.metric(label="Similarity Score", value=f"{result.similarity:.2f}")
                mcol2.metric(label="Confidence", value=f"{result.confidence:.2f}")
                
                # Show full details exactly as requested
                st.markdown("**Full System Record:**")
                st.json(result.to_json())
                
    except Exception as e:
        st.error(f"Failed to process image: {e}")
