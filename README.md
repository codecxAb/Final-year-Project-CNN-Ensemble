# 🩺 Lung Cancer Classifier - Streamlit App

Binary classification of lung CT scans (Malignant vs Benign) with Grad-CAM explainability.

## 🚀 Quick Start

### Local Development

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model file is present**:

   - File: `Concat _VGG19 _ ResNet50_.keras`
   - Location: Same directory as `app.py`

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

4. **Open browser**:
   - URL will be shown in terminal (usually `http://localhost:8501`)

## 📁 Project Structure

```
lungCanerProject/2nd_draft/
├── app.py                              # Main Streamlit application
├── Concat _VGG19 _ ResNet50_.keras    # Trained ensemble model
├── Average _ResNet50 _ EfficientNetB0_.keras  # Alternative model
├── requirements.txt                    # Python dependencies
├── GRADCAM_FIX_SUMMARY.md            # Technical documentation
└── test_gradcam_fix.py               # Validation script
```

## 🔧 Technical Details

### Model Architecture

- **Type**: Ensemble (Concatenation)
- **Backbones**: VGG19 + ResNet50
- **Input**: Two preprocessed 224×224×3 images
- **Output**: Binary classification (Benign/Malignant)

### Features

✅ Binary classification (Malignant vs Benign)  
✅ Grad-CAM visual explanations  
✅ Confidence scores  
✅ Robust error handling  
✅ Multiple fallback strategies for visualization  
✅ Clean, professional UI

### Recent Fixes

- **Fixed Grad-CAM input format issue** (dict → list conversion)
- **Added fallback visualization strategies**
- **Enhanced error handling and user feedback**
- **Production-ready deployment**

## 🧪 Testing

Run the validation script:

```bash
python test_gradcam_fix.py
```

Expected output: All tests pass ✅

## 📊 Model Performance

Based on training (from Colab):

- **Accuracy**: ~71-75%
- **F1-Score**: ~0.72-0.76
- **Classes**: Balanced (no data leakage)
- **Validation**: Stratified split with proper separation

## 🌐 Deployment

### Streamlit Cloud

1. Push to GitHub repository
2. Connect repository to Streamlit Cloud
3. Set Python version: 3.10+
4. Deploy!

**Important**: Ensure `.keras` model files are committed to repository.

### Local Server

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔍 Troubleshooting

### "Model file not found"

- Ensure `Concat _VGG19 _ ResNet50_.keras` is in the same directory as `app.py`
- Check file permissions

### "Grad-CAM failed"

- This is **expected behavior** in some cases
- The app will show a warning but prediction remains valid
- Check debug information in expandable section
- The fix includes 3 fallback strategies

### TensorFlow errors

- Ensure TensorFlow ≥ 2.13.0 is installed
- Try: `pip install --upgrade tensorflow`

## 📝 Usage

1. **Upload Image**: Click "Choose a lung CT scan"
2. **Analyze**: Click "🔍 Analyze Image" button
3. **Review Results**:
   - Prediction (Malignant/Benign)
   - Confidence percentage
   - Grad-CAM visualization (if available)
   - Class probabilities

## ⚠️ Disclaimer

**This tool is for educational purposes only.**  
It is NOT a medical diagnostic device.  
Always consult a qualified radiologist for clinical diagnosis.

## 🛠️ Development

### Code Quality

- Type hints in critical functions
- Comprehensive error handling
- Fallback strategies for robustness
- Clean code architecture

### Model Training

Training code available in Colab notebook (see chat history).  
Key features:

- No data leakage
- Proper train/val/test split
- Balanced training data
- Ensemble approach

## 📧 Support

For issues or questions:

1. Check `GRADCAM_FIX_SUMMARY.md` for technical details
2. Run `test_gradcam_fix.py` to validate setup
3. Review Streamlit logs for errors

---

**Version**: 2.0 (Grad-CAM Fixed)  
**Last Updated**: November 2025  
**Status**: ✅ Production Ready
