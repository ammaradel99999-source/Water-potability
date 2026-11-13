import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_potability_pipeline.pkl")
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

# الموديل Pipeline كامل (KNNImputer + StandardScaler + ExtraTrees) فمش محتاج أي معالجة يدوية هنا
model = joblib.load(MODEL_PATH)

# التنبؤ بعينة واحدة أسرع بكتير من غير multi-threading (الموديل اتدرّب بـ n_jobs=-1)
try:
    model.steps[-1][1].n_jobs = 1
except Exception:
    pass


def predict_water_potability(*values):
    # الخانة الفاضية بتيجي None، والـ KNNImputer جوه الـ Pipeline بيعوّضها
    row = [np.nan if v is None else float(v) for v in values]

    if any(np.isinf(row)):
        return "⚠️ فيه قيمة غير صالحة (infinity)."
    if any(v < 0 for v in row if not np.isnan(v)):
        return "⚠️ كل القياسات لازم تكون قيم موجبة (أو صفر)."
    ph = row[0]
    if not np.isnan(ph) and ph > 14:
        return "⚠️ قيمة الـ pH لازم تكون بين 0 و 14."
    if all(np.isnan(v) for v in row):
        return "⚠️ أدخل قيمة واحدة على الأقل."

    input_data = pd.DataFrame([row], columns=FEATURES)
    prediction = int(model.predict(input_data)[0])
    prob_potable = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        return f"✅ صالحة للشرب (Potable) — احتمال الصلاحية: {prob_potable * 100:.1f}%"
    return f"🚫 غير صالحة للشرب (Not Potable) — احتمال عدم الصلاحية: {(1 - prob_potable) * 100:.1f}%"


inputs = [
    gr.Number(label="pH (حموضة المياه)", value=7.0),
    gr.Number(label="Hardness (الصلابة) mg/L", value=200.0),
    gr.Number(label="Solids (المواد الصلبة الذائبة) ppm", value=20000.0),
    gr.Number(label="Chloramines (الكلورامين) ppm", value=7.0),
    gr.Number(label="Sulfate (الكبريتات) mg/L", value=330.0),
    gr.Number(label="Conductivity (التوصيل الكهربائي) μS/cm", value=400.0),
    gr.Number(label="Organic_carbon (الكربون العضوي) ppm", value=14.0),
    gr.Number(label="Trihalomethanes (الهالوميثانات الثلاثية) μg/L", value=66.0),
    gr.Number(label="Turbidity (العكارة) NTU", value=4.0),
]

demo = gr.Interface(
    fn=predict_water_potability,
    inputs=inputs,
    outputs=gr.Textbox(label="نتيجة التقييم"),
    title="💧 Water Potability Prediction System",
    description=(
        "أدخل الخصائص الكيميائية والفيزيائية لعينة المياه لمعرفة هل هي صالحة للشرب. "
        "ممكن تسيب أي خانة فاضية وهيتم تعويضها تلقائياً. "
        "⚠️ دقة النموذج حوالي 66–68% فقط، فهو للتجربة والتعلم وليس بديلاً عن التحليل المعملي."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
