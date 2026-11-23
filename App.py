import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Модельді жүктеу
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

st.title("EcoVision AI - Қоқысты сұрыптау")
st.write("Қоқыстың суретін жүктеңіз, ЖИ оның түрін анықтайды.")

uploaded_file = st.file_uploader("Суретті таңдаңыз...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Жүктелген сурет', use_column_width=True)
    
    with st.spinner('ЖИ суретті талдап жатыр...'):
        results = model(image)
        
        # Нәтижені алу
        names = model.names
        detected_objects = []
        for r in results:
            for c in r.boxes.cls:
                detected_objects.append(names[int(c)])
        
        if detected_objects:
            st.success(f"Анықталған нысандар: {', '.join(set(detected_objects))}")
            
            # Қай контейнерге тастау керектігін анықтау (қарапайым логика)
            for obj in set(detected_objects):
                if obj in ['bottle', 'cup', 'wine glass']:
                    st.info(f"**{obj}** -> Пластик немесе Шыны контейнеріне тастаңыз.")
                elif obj in ['book', 'paper']:
                    st.info(f"**{obj}** -> Қағаз контейнеріне тастаңыз.")
                elif obj in ['banana', 'apple', 'orange', 'carrot']:
                    st.info(f"**{obj}** -> Органикалық қалдықтар контейнеріне тастаңыз.")
                else:
                    st.warning(f"**{obj}** -> Бұл нысанның нақты қай контейнерге жататынын тексеріңіз.")
        else:
            st.warning("Ештеңе анықталмады. Басқа сурет жүктеп көріңіз.")
