import io
import zipfile
import fitz  # PyMuPDF
import docx2txt
from PIL import Image
from pdf2image import convert_from_bytes
import streamlit as st
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# Load model and processor once
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    return processor, model

processor, model = load_model()

def extract_images_from_pdf(pdf_bytes):
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                images.append(img_pil)
            except Exception:
                continue
    return images

def extract_images_from_docx(docx_bytes):
    # docx2txt requires a filename, so we write bytes to temp file then extract images from folder
    import tempfile
    import os

    images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(docx_bytes)
        tmp_path = tmp_file.name

    # Extract images to temp folder
    temp_img_dir = tempfile.mkdtemp()
    try:
        docx2txt.process(tmp_path, temp_img_dir)
        # Load images from temp_img_dir
        for filename in os.listdir(temp_img_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(f"{temp_img_dir}/{filename}").convert("RGB")
                    images.append(img)
                except Exception:
                    continue
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return images

def extract_images_from_visual_detection(image):
    sub_images = []
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    width, height = image.size
    results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]

    for box in results["boxes"]:
        xmin, ymin, xmax, ymax = map(int, box.tolist())
        # Safety crop boundaries
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, width)
        ymax = min(ymax, height)
        cropped = image.crop((xmin, ymin, xmax, ymax))
        sub_images.append(cropped)
    return sub_images

# Streamlit UI
st.write("Just a small effort to make things easier — no more old-school screenshots! 😊 This app might not be fancy, but it's made to help students like you.")
st.title("Tables and Images Extractor")

uploaded_file = st.file_uploader("Upload a PDF, DOCX or Image file", type=["pdf", "docx", "jpg", "jpeg", "png"])

option = st.radio("Choose what to extract:", ["Tables", "Images", "Both"])

submit = st.button("Submit")

# Placeholders for progress and download button
progress = st.empty()
progress_bar = st.empty()
download_btn_placeholder = st.empty()

# Variables to hold results and state
table_images = []
non_table_images = []
download_ready = False

if submit:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a file before submitting.")
    else:
        st.info("⏳ Please wait while we process your file...")  # show once before extraction

        file_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()

        table_images = []
        non_table_images = []
        page_images = []

        # Load pages/images depending on file type
        if filename.endswith(".pdf"):
            try:
                page_images = convert_from_bytes(file_bytes, poppler_path=r"D:\\poppler\\Library\\bin")
            except Exception as e:
                st.error(f"Error converting PDF to images: {e}")
                page_images = []

            if option in ["Images", "Both"]:
                non_table_images = extract_images_from_pdf(file_bytes)

        elif filename.endswith(".docx"):
            page_images = []
            if option in ["Images", "Both"]:
                non_table_images = extract_images_from_docx(file_bytes)

        else:  # Image file
            try:
                img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                page_images = [img]
            except Exception as e:
                st.error(f"Error loading image: {e}")
                page_images = []

            if option in ["Images", "Both"]:
                detected_images = extract_images_from_visual_detection(img) if page_images else []
                non_table_images = detected_images if detected_images else ([img] if page_images else [])

        # Extract tables if needed
        if option in ["Tables", "Both"] and page_images:
            total = len(page_images)
            for idx, img in enumerate(page_images):
                inputs = processor(images=img, return_tensors="pt")
                outputs = model(**inputs)
                width, height = img.size
                results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]

                for box in results["boxes"]:
                    xmin, ymin, xmax, ymax = map(int, box.tolist())
                    # Add a small margin for better cropping
                    xmin = max(xmin - 15, 0)
                    ymin = max(ymin - 15, 0)
                    xmax = min(xmax + 15, width)
                    ymax = min(ymax + 15, height)
                    cropped = img.crop((xmin, ymin, xmax, ymax))
                    table_images.append(cropped)

                progress_val = int(((idx + 1) / total) * 100)
                progress_bar.progress(progress_val)
                progress.markdown(f"**Progress: {progress_val}%**")
        else:
            if option in ["Tables", "Both"]:
                page_images = []

        progress_bar.progress(100)
        progress.markdown("Extraction complete!")

        # Show appropriate no data found warnings based on option
        if option == "Tables":
            if len(table_images) == 0:
                st.warning("No tables found in the provided document.")
        elif option == "Images":
            if len(non_table_images) == 0:
                st.warning("No images found in the provided document.")
        else:  # Both
            if len(table_images) == 0 and len(non_table_images) == 0:
                st.warning("No tables or images found in the provided document.")

        download_ready = True

if download_ready:
    # Show download button on top (replace progress)
    download_btn_placeholder.empty()
    progress.empty()
    progress_bar.empty()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        if option in ["Tables", "Both"]:
            for i, img in enumerate(table_images):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                zipf.writestr(f"table_{i+1}.png", buf.getvalue())
        if option in ["Images", "Both"]:
            for j, img in enumerate(non_table_images):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                zipf.writestr(f"image_{j+1}.png", buf.getvalue())

    st.download_button(
        label="⬇️ Download Selected Extracted Images as ZIP",
        data=zip_buffer.getvalue(),
        file_name="extracted_output.zip",
        mime="application/zip",
        key="download_button_top",
    )

    # Show extracted images below download button
    if option in ["Tables", "Both"] and table_images:
        st.subheader("Extracted Tables")
        for i, img in enumerate(table_images):
            st.image(img, caption=f"Table {i+1}")

    if option in ["Images", "Both"] and non_table_images:
        st.subheader("Extracted Images")
        for j, img in enumerate(non_table_images):
            st.image(img, caption=f"Image {j+1}")
