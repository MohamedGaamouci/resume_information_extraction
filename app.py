import streamlit as st
from PIL import Image
import base64
import json
from cv_information_extraction import cv_information_extraction as cv_ext
from streamlit_option_menu import option_menu

cv_detection = cv_ext()

# ---------------------- Run Extraction Method ----------------------


def Run_Extraction(uploaded_files, show_boxes=True, filter=False, target_classes=None, search_query=None):
    image_inputs = []
    valid_files = []

    # Step 1: Prepare image inputs
    for file in uploaded_files:
        if file.type.startswith("image"):
            try:
                image = Image.open(file).convert("RGB")
                image_inputs.append(image)
                valid_files.append(file)
            except Exception as e:
                st.warning(f"Failed to read {file.name}: {e}")
        else:
            st.warning(f"{file.name} is not an image. Skipping.")

    if not image_inputs:
        st.info("No valid images uploaded.")
        return

    # Step 2: Batch detection and OCR
    results = cv_detection.detect_and_ocr_batch(image_inputs)

    # Step 3: Display results
    for idx, (json_info, image_det, personal_image) in enumerate(results):
        file = valid_files[idx]

        # Step 4: Apply filtering if enabled
        if filter:
            matched = False
            if target_classes and search_query and search_query.strip() != "":
                for target_class in target_classes:
                    if target_class in json_info:
                        value = json_info[target_class]
                        if search_query.lower() in value.lower():
                            matched = True
                            break
                if not matched:
                    continue  # Skip if not matched

        # Step 5: Display layout
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"📌 Document {idx+1}")
            st.write(f"Filename: `{file.name}`")

            st.image(
                image_inputs[idx], caption="Original Document", use_container_width=True)

            if show_boxes:
                st.image(image_det, caption="📦 Layout Detection Preview",
                         use_container_width=True)

            if personal_image is not None:
                st.image(personal_image, caption="🖼️ Personal Image",
                         use_container_width=True)

        with col2:
            st.subheader("🧠 Extracted Content")
            extracted_text = json_info.get('Name', 'No name extracted')
            st.markdown(extracted_text, unsafe_allow_html=True)

            if enable_download:
                with st.expander("🧾 Preview Extracted CV (Structured Format)", expanded=True):
                    st.json(json_info)

                def create_download_link(data_dict, filename):
                    json_str = json.dumps(data_dict, indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="{filename}_extracted.json">📥 Download Extracted JSON</a>'
                    return href

                st.markdown(create_download_link(
                    json_info, file.name), unsafe_allow_html=True)

# -----------------------------------------------------------


# ---------------------- APP CONFIG ----------------------
st.set_page_config(page_title="Document Information Extractor", layout="wide")
st.title("📄 Information Extraction from Scanned Documents 'CV'")
st.caption(
    "Built with layout analysis (YOLOv11), OCR (Tesseract), and NLP — UI by Streamlit")

# ---------------------- SIDEBAR ----------------------
# st.sidebar.header("⚙️ Settings")
# page = st.sidebar.radio(
#     "Navigate", ["Upload & Extract", "View Uploaded Images"])
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",  # Optional
        options=["Upload & Extract", "View Uploaded Images"],
        icons=["cloud-upload", "image"],  # Font Awesome icon names
        menu_icon="cast",  # Main icon at the top
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "#4285f4", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px",
                "--hover-color": "#eee",
            },
            # "nav-link-selected": {"background-color": "#4285f4", "color": "white"},
        },
    )

# ---------------------- SESSION STATE ----------------------
# Initialize session state to hold uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ---------------------- PAGE 1: Upload & Extract ----------------------
if page == "Upload & Extract":
    # Show detected layout
    show_boxes = st.sidebar.checkbox("Show Detected Layout Boxes", value=True)
    # Sidebar settings
    filter_cv = st.sidebar.checkbox("Filter Resumes (CV)", value=False)

    # Define your 14 CV sections (adjust if needed)
    default_cv_sections = [
        "Name", "Profile", "Contact", "Education", "Experience", "Skills",
        "Certifications", "Languages", "Projects", "Achievements",
        "Interests", "Summary", "References", "Personal Info"
    ]

    # Initialize session state
    if "selected_sections" not in st.session_state:
        st.session_state.selected_sections = default_cv_sections[4:6]

    if filter_cv:
        st.sidebar.markdown("### 🎯 Resume Filter Settings")

        with st.sidebar.expander("🔎 Custom Resume Filter", expanded=True):
            st.caption(
                "Type a keyword and select the CV sections you want to filter by:")

            # Keyword input
            search_query = st.text_input("Keyword", "")

            # Buttons for quick toggle
            col1, col2, col3 = st.columns(3)
            if col1.button("✅ Select All"):
                st.session_state.selected_sections = default_cv_sections.copy()
            if col2.button("❌ Deselect All"):
                st.session_state.selected_sections = []
            if col3.button("🚀 Run filter"):
                filter = True
            else:
                filter = False

            # Checkboxes
            selected_sections = []
            for section in default_cv_sections:
                checked = section in st.session_state.selected_sections
                new_checked = st.checkbox(section, value=checked, key=section)
                if new_checked:
                    selected_sections.append(section)

            # Update session state
            st.session_state.selected_sections = selected_sections
    else:
        search_query = ""
        selected_sections = default_cv_sections.copy()

    #

    enable_download = st.sidebar.checkbox("Enable Export", value=True)

    # File upload
    uploaded_files = st.file_uploader("📤 Upload one or more scanned documents (Image or PDF)",
                                      type=["jpg", "jpeg", "png", "pdf"],
                                      accept_multiple_files=True)
    # Merge new uploads with existing ones, avoiding duplicates
    existing_files = {f.name: f for f in st.session_state.uploaded_files}
    new_files = {f.name: f for f in uploaded_files}
    # new_files overwrites duplicates
    combined_files = {**existing_files, **new_files}

    # Update session state with unique files
    st.session_state.uploaded_files = list(combined_files.values())
    uploaded_files = st.session_state.uploaded_files

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files  # Store in session state
        st.success(f"{len(uploaded_files)} file(s) uploaded.")

        if filter_cv:
            if filter:
                Run_Extraction(uploaded_files, show_boxes, filter=filter,
                               target_classes=st.session_state.selected_sections, search_query=search_query)
        # Analysis button
        else:
            if st.button("🚀 Run Extraction"):
                Run_Extraction(uploaded_files, show_boxes)

    else:
        st.info("⬆️ Please upload one or more scanned documents to begin.")

# ---------------------- PAGE 2: View Uploaded Images ----------------------
elif page == "View Uploaded Images":
    st.title("🖼️ Uploaded Document Previews")

    if st.session_state.uploaded_files:
        uploaded_files = st.session_state.uploaded_files
        # Calculate number of rows needed
        rows = (len(uploaded_files) + 3) // 4

        # Display images in a 4-column grid
        for i in range(rows):
            cols = st.columns(4)
            for j in range(4):
                idx = i * 4 + j
                if idx < len(uploaded_files):
                    file = uploaded_files[idx]
                    with cols[j]:
                        if file.type.startswith("image"):
                            image = Image.open(file)
                            st.image(image, use_container_width=True)
                        else:
                            st.info("📄 PDF file")
    else:
        st.warning("No files uploaded yet. Go to the Upload page first.")
