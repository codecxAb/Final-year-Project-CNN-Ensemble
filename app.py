# =============================================================
# STREAMLIT APP: BINARY LUNG CANCER CLASSIFIER (MALIGNANT vs BENIGN)
# - Uses trained ensemble model: Concat(VGG19 + ResNet50)
# - Includes Grad-CAM explainability
# - Full error handling & debugging info
# - Optimized for Colab Free Tier deployment
# =============================================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
import numpy as np
import cv2
from PIL import Image
import os

# ------------------------------
# CONFIGURATION
# ------------------------------
st.set_page_config(
    page_title="Lung Cancer Classifier (Malignant vs Benign)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
.prediction-box { background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; font-size: 1.2em; font-weight: bold; margin: 1rem 0; }
.confidence-box { background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; font-size: 1.1em; margin: 1rem 0; }
.warning-box { background-color: #fff3cd; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# MODEL LOADING WITH ERROR HANDLING
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_ensemble_model():
    model_path = "Concat _VGG19 _ ResNet50_.keras"
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure the model is uploaded to `saved_ensembles/`.")
        st.info(f"Expected path: `{os.path.abspath(model_path)}`")
        return None

    try:
        st.info("🧠 Loading ensemble model...")
        model = load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {str(e)}")
        st.code(str(e), language="python")
        return None

# ------------------------------
# GRAD-CAM WITH PROPER INPUT HANDLING
# ------------------------------
class GradCAM:
    def __init__(self, model, layer_name="resnet50"):
        self.model = model
        self.layer_name = layer_name
        self._build_grad_model()

    def _build_grad_model(self):
        """Build grad model for ensemble with multiple inputs"""
        try:
            # Get the target layer (ResNet50 backbone)
            target_layer = self.model.get_layer(self.layer_name)
            
            # Create a model that maps from inputs to target layer + predictions
            self.grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception as e:
            raise RuntimeError(f"Grad-CAM model build failed: {e}")

    def compute_heatmap(self, full_input_dict, class_idx, eps=1e-8):
        """Compute Grad-CAM heatmap using dictionary input
        
        Args:
            full_input_dict: Dict with keys 'resnet_input' and 'vgg_input'
            class_idx: Target class index for Grad-CAM
            eps: Small epsilon for numerical stability
        """
        # Convert dict to list in the correct order matching model.input_names
        # The model expects [resnet_input, vgg_input]
        input_list = [full_input_dict["resnet_input"], full_input_dict["vgg_input"]]
        
        with tf.GradientTape() as tape:
            # Pass inputs as a list, not a dict
            conv_outputs, predictions = self.grad_model(input_list)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("Gradients are None. Check model architecture.")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image"""
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_jet, alpha, 0)
    return np.clip(superimposed, 0, 255).astype("uint8")

def generate_gradcam_safe(model, full_input, pred_idx, original_img):
    """
    Safely generate Grad-CAM with multiple fallback strategies
    
    Returns:
        tuple: (success: bool, result: np.ndarray or error_message: str)
    """
    # Strategy 1: Standard Grad-CAM with ResNet50 layer
    try:
        gradcam = GradCAM(model, layer_name="resnet50")
        heatmap = gradcam.compute_heatmap(full_input, pred_idx)
        superimposed = overlay_gradcam(original_img, heatmap)
        return True, superimposed
    except Exception as e1:
        st.warning(f"Strategy 1 failed: {str(e1)[:100]}")
    
    # Strategy 2: Try with different layer names
    layer_attempts = ["conv5_block3_out", "conv5_block2_out", "conv4_block6_out"]
    for layer_name in layer_attempts:
        try:
            # Check if layer exists
            _ = model.get_layer(layer_name)
            gradcam = GradCAM(model, layer_name=layer_name)
            heatmap = gradcam.compute_heatmap(full_input, pred_idx)
            superimposed = overlay_gradcam(original_img, heatmap)
            st.info(f"✅ Used alternative layer: {layer_name}")
            return True, superimposed
        except Exception:
            continue
    
    # Strategy 3: Simple saliency map (gradient-based)
    try:
        input_tensor = tf.constant(full_input["resnet_input"])
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = model({"resnet_input": input_tensor, "vgg_input": full_input["vgg_input"]})
            target_class = predictions[:, pred_idx]
        
        grads = tape.gradient(target_class, input_tensor)
        grads = tf.abs(grads)
        grads = tf.reduce_max(grads, axis=-1)[0]
        grads = grads.numpy()
        
        # Normalize
        grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
        superimposed = overlay_gradcam(original_img, grads)
        st.info("✅ Used saliency map (alternative visualization)")
        return True, superimposed
    except Exception as e3:
        return False, f"All Grad-CAM strategies failed. Last error: {str(e3)}"

# ------------------------------
# IMAGE PREPROCESSING
# ------------------------------
def preprocess_image(pil_image, target_size=(224, 224)):
    """Preprocess PIL image for ensemble model"""
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    img_resized = pil_image.resize(target_size)
    img_array = np.array(img_resized)
    
    # Create dictionary input for ensemble
    resnet_input = resnet_preprocess(np.expand_dims(img_array, axis=0))
    vgg_input = vgg_preprocess(np.expand_dims(img_array, axis=0))
    
    return {
        "resnet_input": resnet_input.astype(np.float32),
        "vgg_input": vgg_input.astype(np.float32)
    }, img_array

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_ensemble(model, full_input):
    """Predict using ensemble model"""
    try:
        preds = model.predict(full_input, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        return pred_idx, confidence, preds
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

# ------------------------------
# MAIN APP
# ------------------------------
def main():
    st.markdown('<div class="main-header"><h1>🩺 Lung Cancer Classifier (Malignant vs Benign)</h1><p>AI-powered analysis with visual explanations</p></div>', unsafe_allow_html=True)
    
    # Load model
    model = load_ensemble_model()
    if model is None:
        st.stop()

    # Session state
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    col1, col2 = st.columns([1, 1])

    # Upload section
    with col1:
        st.header("📤 Upload Lung CT Image")
        uploaded_file = st.file_uploader(
            "Choose a lung CT scan (JPG/PNG)",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            try:
                pil_image = Image.open(uploaded_file)
                st.image(pil_image, caption="Uploaded Image", use_column_width=True)
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    st.session_state.analysis_done = True
                    st.session_state.uploaded_image = pil_image
            except Exception as e:
                st.error(f"❌ Invalid image file: {e}")

    # Results section
    with col2:
        st.header("📊 Analysis Results")
        
        if st.session_state.analysis_done and st.session_state.uploaded_image:
            with st.spinner('🧠 Running inference...'):
                try:
                    # Preprocess
                    full_input, original_img = preprocess_image(st.session_state.uploaded_image)
                    
                    # Predict
                    pred_idx, confidence, all_preds = predict_ensemble(model, full_input)
                    prediction = "Malignant" if pred_idx == 1 else "Benign"
                    
                    # Display results
                    st.markdown(f'<div class="prediction-box">🎯 Prediction: {prediction}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-box">📈 Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
                    
                    # Grad-CAM
                    st.subheader("🔍 Grad-CAM Explanation")
                    with st.spinner('🔍 Generating visual explanation...'):
                        success, result = generate_gradcam_safe(model, full_input, pred_idx, original_img)
                        
                        if success:
                            st.image(result, caption="Red areas = high influence on prediction", use_column_width=True)
                            st.success("✅ Visual explanation generated successfully")
                        else:
                            st.warning("⚠️ Could not generate Grad-CAM visualization")
                            st.info("📌 Prediction is still valid. Grad-CAM is only for visualization.")
                            
                            # Show debug info in expander
                            with st.expander("🔧 Debug Information"):
                                st.code(result, language="python")
                                st.write("**Available layers in model:**")
                                try:
                                    layer_names = [layer.name for layer in model.layers]
                                    st.write(layer_names[:20])  # Show first 20 layers
                                except:
                                    st.write("Could not retrieve layer names")
                    
                    # Probabilities
                    st.subheader("📈 Class Probabilities")
                    labels = ["Benign", "Malignant"]
                    for i, prob in enumerate(all_preds):
                        color = "#f5576c" if i == pred_idx else "#4facfe"
                        st.markdown(f'''
                        <div style="margin-bottom: 0.5rem">
                            <div style="font-weight: bold; color: {color}">{labels[i]}</div>
                            <div style="background-color: #e9ecef; border-radius: 5px; height: 20px">
                                <div style="background: {color}; width: {prob*100}%; height: 100%; border-radius: 5px; text-align: right; padding: 0 5px">{prob:.2%}</div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.code(str(e), language="python")
        
        elif uploaded_file:
            st.info("👆 Click 'Analyze Image' to start diagnosis")
        else:
            st.info("👈 Upload a lung CT scan to begin")

    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only. 
    It is NOT a medical diagnostic device. Always consult a qualified radiologist for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()