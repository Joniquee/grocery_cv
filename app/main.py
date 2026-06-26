from fastapi import FastAPI, UploadFile, File
from app.model.predict import predict, model, class_names
from app.data.data_preparation import transform_single_image
import io

app = FastAPI()

@app.post('/get_item')
async def get_item(img: UploadFile = File(...)):
        contents = await img.read()
        img_bytes = io.BytesIO(contents)
        data = transform_single_image(img_bytes)
        predict_data = predict(model, data)
        labels = [pair[0] for pair in predict_data[0]]
        class_names = predict_data[1]
        print(labels, class_names)
        result_dict = {}
        for i in range(len(labels)):
             class_name = class_names[i]
             label = labels[i]
             result_dict[class_name] = label
        return {'predict': result_dict}


def main():
    pass

if __name__ == "__main__":
    main()
