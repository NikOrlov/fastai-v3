from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

export_file_url = 'https://drive.google.com/uc?export=download&id=1hnHnxKQgi7wpA4ANgR071-qY8GqLj7m1'
export_file_name = 'export.pkl'

export_file_url_volkswagen = 'https://drive.google.com/uc?export=download&id=1-QyiB7tNePlK9Yvzs4aACjS_3lPImrJH'
export_file_url_ford = 'https://drive.google.com/uc?export=download&id=1-0OOCPjKd29ahvIMzl1CE7QjCZTw8vj6'
export_file_url_mazda = 'https://drive.google.com/uc?export=download&id=1-LqqfkYEzymZGLx87T-IQ1GLzhI112jc'
export_file_url_opel = 'https://drive.google.com/uc?export=download&id=1-ZxN2X4kS0AZAfXro7hNJv_asjgbLwwg'
export_file_url_audi = 'https://drive.google.com/uc?export=download&id=1-fB0tDZSJhaTbGKPtnR1QRpb2RXNnPKX'
export_file_url_skoda = 'https://drive.google.com/uc?export=download&id=1-5YzY3KJ6NCaENeBLJTeb-04aW_F5028'
export_file_url_kia = 'https://drive.google.com/uc?export=download&id=1-3gJ9wnJ0sOTywHJiqRsW2_I65FXCCrk'
export_file_url_mercedes = 'https://drive.google.com/open?id=1-HcgvMkZHsoefbEQncADFEOUWG-E-CJv'
export_file_url_bmw = 'https://drive.google.com/uc?export=download&id=1b17uMihx6nnIpvKlf8Y7am3rnEljgvFK'

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner(path, export_file_url):
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise
      
            
            
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner(path,export_file_url))]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


loop = asyncio.get_event_loop()
#     tasks = [asyncio.ensure_future(setup_learner(model_path,globals()['export_file_url' + '_' + str(prediction)]))]
tasks = [asyncio.ensure_future(setup_learner(model_path,export_file_url_volkswagen))]
model_learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    
    model_path = path/str(prediction)
    


    model_prediction = model_learn.predict(img)[0]
#     return JSONResponse({'result': str(model_prediction)})
    return JSONResponse({'result': str(model_prediction)})
if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
