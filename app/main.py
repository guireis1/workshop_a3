import logging
from fastapi import FastAPI
from starlette_prometheus import metrics, PrometheusMiddleware
from pydantic import BaseModel
import sys
from sentence_transformers import SentenceTransformer
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import scipy.spatial

sys.path.insert(0,'/./app/')

def get_fast_api_client():
    """
        Retorna um cliente FastAPI com uma rota base 
        para cenários onde há um Proxy em frente a aplicação
    """
    app = FastAPI()
    return app

app = get_fast_api_client()

"""Add Telemetry to the application"""
FastAPIInstrumentor.instrument_app(app)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

my_logger = logging.getLogger("similarity.main")
my_logger.setLevel(logging.INFO)

    
@app.on_event('startup')
async def on_startup() -> None:
    global model

    my_logger.info('Setting model')
    try:
        model = SentenceTransformer('models', device='cpu')
        #model = SentenceTransformer('distiluse-base-multilingual-cased', device='cpu')
        my_logger.info('Model set')
    except Exception as e:
        my_logger.error(e)
    
@app.get('/health')
def load_model():
    if dir(model):
        return {'res': 'OK'}
    
@app.get('/')
def index():
    return {'message': 'similarity'}

class Data(BaseModel):
    sentence: str
    sentence_compare: str
    
@app.post("/predict")
def predict(data: Data):
    """
    Recebe ideia, faz a predição, e retorna uma lista de IDs e Scores
    :param data: Descricao, Beneficio, Cliente
    :return: Lista de Ids e Scores
    """
    my_logger.info("Predicting..")
    try:
        data_dict = data.dict()
        sentence = data_dict['sentence']
        sentence_compare = data_dict['sentence_compare']        
        sentence_embed = model.encode(sentence, convert_to_tensor=False)
        sentence_compare_embed = model.encode(sentence_compare, convert_to_tensor=False)     
        my_logger.info("Embed sucess")

        result = calculate_distances(sentence_embed, sentence_compare_embed)
            
        my_logger.info("Distance sucess")
        my_logger.info("Predict sucess")
        return {"Result": 1-result[0]}

    except Exception as e:
        my_logger.error("Predict error!", e)
        return {"Result": e}
    
def calculate_distances(sentence_embed, sentence_compare_embed):

    score = scipy.spatial.distance.cdist([sentence_embed], [sentence_compare_embed], "cosine")[0]
    return score