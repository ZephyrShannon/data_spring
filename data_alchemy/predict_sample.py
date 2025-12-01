# infer_onnx.py
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("lstm_moe_model.onnx")
x = np.random.randn(1, 180, 3600).astype(np.float32)
outputs = ort_session.run(None, {'input': x})
y_pred = outputs[0]  # (1, 12)
print("ONNX 预测:", y_pred)