import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class WhiskyApplication(BaseModel):
        Body: int
        Sweetness: int
        Smoky: int
        Medicinal: int
        Honey: int
        Spicy: int
        Winey: int
        Nutty: int
        Malty: int
        Fruity: int
        Floral: int

model_ref= bentoml.sklearn.get("whisky_pred:latest")

dv = model_ref.custom_objects['DictVectorizer']
model_runner = model_ref.to_runner()

svc = bentoml.Service("whisky_pred", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=WhiskyApplication), output=JSON())
def classify(whisky_application):
        application_data = whisky_application.dict()
        vector = dv.transform(application_data)
        prediction = model_runner.predict.run(vector)
        print(prediction)
        
        result = prediction[0]
        
        if result > 0.5:
         return{"status": "It has Tobacco"}
        else: 
         return {"status": "Not has Tobacco"}
 

