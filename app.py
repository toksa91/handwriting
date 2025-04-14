import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 페이지 제목
st.set_page_config(page_title="손글씨 숫자 인식기", layout="centered")
st.title("✍️ 손글씨 숫자 인식기")
st.write("숫자가 적힌 이미지를 업로드하면 인공지능이 어떤 숫자인지 예측해줘요!")

# 모델 로드
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_model.h5')

model = load_model()

# 파일 업로드
uploaded_file = st.file_uploader("숫자가 적힌 이미지를 업로드하세요 (28x28 흑백 이미지 권장)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # 흑백으로 변환
    st.image(image, caption='업로드한 이미지', width=150)

    # 이미지 전처리 (28x28 크기로 조정)
    image = ImageOps.invert(image.resize((28, 28)))  # 흰 배경, 검정 숫자
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 784)

    # 예측
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)

    st.subheader(f"✅ 예측된 숫자: **{predicted_label}**")

    # 예측 확률 시각화
    st.write("📊 예측 확률 (0~9)")
    st.bar_chart(prediction[0])
