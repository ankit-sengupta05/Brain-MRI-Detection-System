# NeuroVision AI — Brain MRI Tumor Classification Dashboard

React/Vite frontend prototype for the supplied PyTorch benchmark.

### Included
- Cool dark brain/MRI themed UI
- Separate Overview, Model Comparison, Training & Validation, Predictions, Confusion Matrix and Settings sections
- Individual model selection for Custom CNN, ResNet18 and MobileNetV3-Small
- Separate training/test loss and accuracy graphs
- Macro precision/recall/F1 graph
- Model leaderboard and metric scorecard
- Confidence + uncertainty comparison
- Prediction/inference console with demo MRI state
- Class-wise confusion matrix and error analysis
- Responsive layout
.
### Run
npm install
npm run dev

The prediction area is intentionally a frontend demo. Connect the "Run inference" action to your PyTorch/FastAPI/Flask endpoint when the backend is ready.
