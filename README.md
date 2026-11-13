---
title: Water Potability Prediction
emoji: 💧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# 💧 Water Potability Prediction

Gradio app that predicts whether a water sample is potable, using an ExtraTrees pipeline
(KNNImputer + StandardScaler + ExtraTreesClassifier) trained on the Water Potability dataset.

⚠️ Test accuracy is about 66–68%. For learning/demo purposes only, not a substitute for lab testing.
