import streamlit as st
import requests
from PIL import Image
from environs import Env
import os

env = Env()
env.read_env()



BASE_PHOTOS = env('BASE_PHOTOS', default='base_photos/')
API_URL = env('API_URL', default = 'localhost:8000')

photo = st.file_uploader(
    'Предположим это автоматическое фото с камеры на сканере, но пока загрузите сюда фото продукта', 
    type=['jpg', 'png', 'jpeg', 'webp', 'avif'],
    accept_multiple_files=False,

    )


if st.button('Что это за продукт?'):
    files = {
            "img": (photo.name, photo.getvalue(), photo.type)
        }
    response = requests.post(f'http://{API_URL}/get_item', files=files)
    body = response.json()
    answers: dict = body['predict']
    print(answers)

    st.text('Наиболее похожие продукты:', text_alignment='center')
    for image_name in answers.keys():
        image_path = f'{BASE_PHOTOS}{image_name}.png'  # или .png
        if os.path.exists(image_path):
            st.image(image_path, width=500, caption=image_name)
