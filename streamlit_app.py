import streamlit as st
import google.generativeai as genai
import logging
import os
from PIL import Image
import io
import numpy as np
from google.cloud import translate_v2 as translate
import requests
from deep_translator import GoogleTranslator
import time

# Set up logging with more detail for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set page configuration for Streamlit
st.set_page_config(
    page_title="Family Medicine Helper",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💊"
)

# Configure the Gemini API
GOOGLE_API_KEY = 'AIzaSyC3DFZnqCKGwFkocf-Hd5U6hkjEICwI1_g'  # Replace with your actual API key
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Custom CSS with improved Telugu support
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;700&display=swap');
    
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #ff6b6b !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #ff5252 !important;
        transition: all 0.3s ease;
    }
    .upload-text {
        text-align: center;
        padding: 1rem;
        background: #f0f2f6;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .medicine-info {
        background: #f8f9fa !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
        margin: 1rem 0 !important;
        color: #000000 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        white-space: pre-line !important;
    }
    .telugu-text {
        font-family: 'Noto Sans Telugu', sans-serif !important;
        font-size: 1.2em !important;
        line-height: 1.8 !important;
        background: #f8f9fa !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        margin: 1rem 0 !important;
        white-space: pre-line !important;
        color: #000000 !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }
    .language-selector {
        background: #fff;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .help-text {
        color: #666;
        font-size: 0.9rem;
        font-style: italic;
    }
    .stTab {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def process_image(image):
    """
    Process the uploaded or captured image for use with the Gemini API.
    """
    try:
        if not isinstance(image, Image.Image):
            st.error("Invalid image format. Please try again.")
            return None
            
        # Validate image size
        max_size = (1024, 1024)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
        # Convert PIL Image to bytes
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Create a byte stream with optimized settings
        byte_stream = io.BytesIO()
        image.save(byte_stream, format='JPEG', quality=85, optimize=True)
        image_bytes = byte_stream.getvalue()
        
        return image_bytes
        
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        st.error("There was an error processing your image. Please try again with a different image.")
        return None

def safe_translate(text, dest_language):
    """Improved translation function using deep_translator"""
    if not text:
        return None
        
    try:
        # Initialize the Google Translator with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source='auto', target=dest_language)
                
                # Split text into smaller chunks
                def split_into_chunks(text, max_size=4500):
                    paragraphs = text.split('\n')
                    chunks = []
                    current_chunk = []
                    current_size = 0
                    
                    for paragraph in paragraphs:
                        if current_size + len(paragraph) > max_size:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = [paragraph]
                            current_size = len(paragraph)
                        else:
                            current_chunk.append(paragraph)
                            current_size += len(paragraph)
                    
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    
                    return chunks

                # Translate chunks with delay between requests
                chunks = split_into_chunks(text)
                translated_chunks = []
                
                for chunk in chunks:
                    translated_text = translator.translate(chunk)
                    if translated_text:
                        translated_chunks.append(translated_text)
                    time.sleep(1)  # Add delay between translations
                
                final_translation = '\n'.join(translated_chunks)
                
                if final_translation and not final_translation.isspace():
                    return final_translation
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)  # Wait before retrying
                
        return text  # Return original text if all retries fail
        
    except Exception as e:
        logging.error(f"Translation error: {e}")
        st.error(f"Translation failed. Using backup translation method...")
        
        # Backup translation method using requests
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": "auto",
                "tl": dest_language,
                "dt": "t",
                "q": text
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                try:
                    translation = ""
                    for item in response.json()[0]:
                        if item[0]:
                            translation += item[0]
                    return translation
                except Exception as e:
                    logging.error(f"Backup translation parsing error: {e}")
                    return text
            else:
                return text
                
        except Exception as e:
            logging.error(f"Backup translation error: {e}")
            return text

def query_tablet_assistant(tablet_name, image=None):
    """Query the Gemini API with better error handling"""
    try:
        model = genai.GenerativeModel('gemini-pro-vision' if image is not None else 'gemini-pro')
        
        if image is not None:
            prompt = f"""Please analyze this medicine and tell me in simple terms:
            1. What is this medicine? (if you can identify it)
            2. How does it look? (shape, color, markings)
            3. What is it commonly used for?
            4. Important things to remember when taking it
            5. What should I watch out for? (common side effects)
            
            Please use very simple language that parents can easily understand.
            If you're not completely sure about the medicine, please say so clearly."""
            
            response = model.generate_content([prompt, image])
        else:
            prompt = f"""Please tell me about the medicine '{tablet_name}' in simple terms:
            1. What is this medicine used for?
            2. When should it be taken?
            3. What are the basic do's and don'ts?
            4. What side effects should I watch out for?
            5. When should I call the doctor?
            
            Please explain everything in simple words that parents can easily understand.
            Avoid medical terms where possible, and when you must use them, explain what they mean."""
            
            response = model.generate_content(prompt)
        
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Unable to get medicine information. Please try again."
            
    except Exception as e:
        logging.error(f"Error in query_tablet_assistant: {e}")
        return "Sorry, there was an error getting the medicine information. Please try again."

def main():
    st.title("💊 Family Medicine Helper")
    st.markdown("##### A simple tool to help you understand your medicines better")
    
    # Initialize session state for storing results
    if 'tablet_details' not in st.session_state:
        st.session_state.tablet_details = None
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = None
    
    # Language selection at the top
    languages = {
        "English": "en",
        "తెలుగు (Telugu)": "te",
        "தமிழ் (Tamil)": "ta",
        "हिंदी (Hindi)": "hi",
        "ಕನ್ನಡ (Kannada)": "kn",
        "മലയാളം (Malayalam)": "ml"
    }
    
    selected_lang = st.selectbox(
        "Choose your preferred language:",
        options=list(languages.keys()),
        key="lang_selector"
    )
    
    input_method = st.radio(
        "How would you like to check your medicine?",
        ("✍️ Type Medicine Name", "📱 Take Picture", "🖼️ Choose Picture from Gallery"),
        key="input_method"
    )
    
    tablet_name = None
    image_data = None
    
    # Input method handling with improved error checking
    if input_method == "✍️ Type Medicine Name":
        tablet_name = st.text_input(
            "What's the name of your medicine?", 
            placeholder="Example: Paracetamol or Crocin"
        )
    
    elif input_method == "📱 Take Picture":
        st.info("📸 Hold your medicine in front of the camera where we can see it clearly")
        image_data = st.camera_input("Take a picture of your medicine")
        if image_data:
            try:
                image = Image.open(image_data)
                processed_image = process_image(image)
                if processed_image:
                    image_data = processed_image
                    st.success("✅ Picture taken successfully!")
                else:
                    st.error("Failed to process the image. Please try again.")
            except Exception as e:
                st.error(f"Error capturing image: {str(e)}")
                logging.error(f"Camera capture error: {e}")
    
    else:  # Choose Picture from Gallery
        image_file = st.file_uploader(
            "Choose a clear picture of your medicine",
            type=["jpg", "jpeg", "png"],
            help="Make sure the medicine is clearly visible in the picture"
        )
        if image_file:
            try:
                image = Image.open(image_file)
                processed_image = process_image(image)
                if processed_image:
                    image_data = processed_image
                    st.image(image, caption="Your Medicine Picture", use_column_width=True)
                    st.success("✅ Image uploaded successfully!")
                    tablet_name = st.text_input(
                        "If you know the medicine name, type it here (optional)",
                        placeholder="Example: Paracetamol or Crocin"
                    )
                else:
                    st.error("Failed to process the uploaded image. Please try again.")
            except Exception as e:
                st.error(f"Error uploading image: {str(e)}")
                logging.error(f"Upload error: {e}")

    # Get medicine information with translation
    if st.button("Tell Me About This Medicine", key="get_info"):
        if tablet_name or image_data is not None:
            with st.spinner("Looking up medicine information... Please wait."):
                medicine_info = query_tablet_assistant(tablet_name, image_data)
                if medicine_info:
                    st.session_state.tablet_details = medicine_info
                    
                    # Show original text
                    st.markdown("### 📋 Medicine Information")
                    st.markdown(f'<div class="medicine-info">{medicine_info}</div>', 
                               unsafe_allow_html=True)
                    
                    # Auto-translate if Telugu is selected
                    if selected_lang == "తెలుగు (Telugu)":
                        with st.spinner("Translating to Telugu... Please wait..."):
                            try:
                                translated_text = safe_translate(medicine_info, "te")
                                if translated_text and translated_text != medicine_info:
                                    st.markdown("### 📋 తెలుగులో సమాచారం (Information in Telugu)")
                                    st.markdown(
                                        f'<div class="telugu-text">{translated_text}</div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.warning("Translation service is experiencing delays. Showing original text for now.")
                            except Exception as e:
                                st.warning("Translation service is temporarily unavailable. Please try again in a few minutes.")
                                logging.error(f"Translation error: {e}")
        else:
            st.warning("Please either enter a medicine name or provide an image.")

    # Sidebar with instructions
    with st.sidebar:
        st.markdown("### 📖 How to Use This Tool")
        st.markdown("""
        **Check Your Medicine:**
        1. **Type the Name:**
           - Simply type your medicine name
           - Click the button to get information
        
        2. **Take a Picture:**
           - Point your camera at the medicine
           - Make sure there's good light
           - Take a clear picture
        
        3. **Use a Saved Picture:**
           - Choose a picture from your phone/computer
           - Picture should be clear and bright
        
        4. **Read in Your Language:**
           - Choose your preferred language
           - Information will be shown in your chosen language
        
        **Remember:**
        - Always follow your doctor's advice
        - This tool is only for information
        - When in doubt, ask your doctor
        """)

if __name__ == "__main__":
    main()