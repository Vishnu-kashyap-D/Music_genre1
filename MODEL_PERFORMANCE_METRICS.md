# 📊 Model Performance Metrics

## Music Genre Classification - Parallel CNN with OpenL3

---

## 🎯 Model Overview

| Attribute | Value |
|-----------|-------|
| **Model Architecture** | Parallel CNN (Multi-branch CNN with Attention Pooling) |
| **Feature Type** | OpenL3 Embeddings (512-dimensional) |
| **Input Processing** | 15-second windows split into 3×5-second slices |
| **Number of Classes** | 10 genres |
| **Model Size** | 56 MB |
| **Training Framework** | PyTorch 2.6.0 with CUDA 12.4 |

---

## 📈 Training & Evaluation Accuracy

| Metric | Score | Percentage |
|--------|-------|------------|
| **Validation Accuracy** | 0.8373 | **83.73%** |
| **Test Accuracy** | 0.8883 | **88.83%** |

### 📝 Notes:
- **Validation Accuracy** (83.73%): Performance on validation set during training
- **Test Accuracy** (88.83%): Final performance on unseen test data
- **Generalization**: Good test performance (88.83%) indicates the model generalizes well

---

## 🎵 Supported Genres (10 Classes)

1. **Blues**
2. **Classical**
3. **Country**
4. **Disco**
5. **Hip-Hop**
6. **Jazz**
7. **Metal**
8. **Pop**
9. **Reggae**
10. **Rock**

---

## 🔍 Model Architecture Details

### **Key Features:**
- ✅ **Multi-branch Architecture**: Parallel processing of multiple time slices
- ✅ **OpenL3 Embeddings**: Pre-trained audio embeddings for rich feature representation
- ✅ **Attention Pooling**: Learns to focus on most relevant temporal segments
- ✅ **Residual Connections**: Improves gradient flow and training stability
- ✅ **Squeeze-and-Excitation Blocks**: Channel-wise attention mechanism
- ✅ **SpecAugment**: Data augmentation for better generalization

### **Training Configuration:**
- **Window Duration**: 15 seconds
- **Slice Duration**: 5 seconds per slice
- **Slices per Window**: 3 slices
- **Window Stride**: 5 seconds (sliding window)
- **Loss Function**: Binary Cross-Entropy (BCE) for multi-label support
- **Optimization**: Adam optimizer with automatic mixed precision (AMP)
- **Device**: CUDA GPU (NVIDIA RTX 3050 Laptop GPU)

---

## 📊 Performance Analysis

### ✅ **Strengths:**
1. **High Test Accuracy (88.83%)**: Excellent performance on unseen data
2. **Good Generalization**: Test accuracy higher than validation accuracy
3. **Multi-label Support**: Can predict multiple genres per audio file
4. **Real-time Inference**: Fast predictions with GPU acceleration
5. **Robust Features**: OpenL3 embeddings provide rich audio representations

### 🔄 **Observations:**
- **Validation Accuracy (83.73%) vs Test Accuracy (88.83%)**:
  - Test set may have been easier or more representative
  - Model generalizes well to new data
  - No signs of overfitting

---

## 🚀 Inference Performance

### **Processing Speed:**
- **Analysis Time**: ~9.4 seconds for 30-second audio (with GPU)
- **Backend**: Flask API on port 5000
- **Frontend**: React app on port 3000
- **Optimization**: 
  - Limited to first 30 seconds of audio
  - Single-pass mel-spectrogram computation
  - Efficient batch processing

### **Audio Quality Checks:**
- ✅ Signal-to-Noise Ratio (SNR) calculation
- ✅ Spectral centroid analysis
- ✅ Zero-crossing rate validation
- ✅ Bandwidth and contrast checks
- ✅ Noise/static detection

---

## 📉 F1 Score & Additional Metrics

> **Note**: F1 Score and per-class metrics (precision, recall) are not currently stored in the model checkpoint. These metrics can be calculated by running evaluation scripts on test data.

### **To Calculate F1 Score:**

```powershell
# Run custom evaluation on test dataset
python evaluate_custom_parallel.py test_data.csv --model torch_models/parallel_genre_classifier_torch.pt

# This will provide:
# - Overall accuracy
# - Per-genre accuracy
# - Precision, Recall, F1 scores (if implemented)
```

### **Recommended Evaluation Script Enhancements:**

```python
from sklearn.metrics import classification_report, f1_score

# After predictions:
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(classification_report(y_true, y_pred, target_names=genre_names))
```

---

## 🎓 Training Dataset

- **Dataset**: GTZAN Genre Collection
- **Total Samples**: 1000 audio files (100 per genre)
- **Duration**: 30 seconds per file
- **Format**: WAV files (22050 Hz, mono)
- **Split**: Train/Validation/Test

---

## 💡 Future Improvements

1. **Metrics Enhancement**:
   - Add F1, Precision, Recall to model checkpoint
   - Implement confusion matrix visualization
   - Per-genre performance breakdown

2. **Model Optimization**:
   - ONNX export for deployment
   - Quantization for faster inference
   - Model distillation for smaller size

3. **Evaluation Tools**:
   - Batch evaluation scripts
   - Cross-validation results
   - Saliency map visualization

---

## 📝 Summary

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **88.83%** ⭐ |
| **Validation Accuracy** | **83.73%** |
| **Model Type** | Parallel CNN + OpenL3 |
| **Genres** | 10 classes |
| **Inference Speed** | ~9.4s for 30s audio |
| **Deployment Status** | ✅ Production Ready |

---

**Model Path**: `torch_models/parallel_genre_classifier_torch.pt`

**Last Updated**: November 16, 2025

---

## 🔗 Related Documentation

- [README.md](README.md) - Main project documentation
- [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) - Comprehensive project analysis
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Setup instructions
- [README_WEB_APP.md](README_WEB_APP.md) - Web application guide
