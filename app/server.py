import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.googleapis.com/drive/v3/files/1-0T-bnNTbRR6AWg6IcW-pxhXG5VYgxos?alt=media&key=AIzaSyAG5J8jVv_BR6_TdY9xun9s78Z_eRDYC24'
export_file_name = '1-0T-bnNTbRR6AWg6IcW-pxhXG5VYgxos'

classes =['bias',
 'clickbait',
 'conspiracy',
 'fake',
 'hate',
 'junksci',
 'political',
 'reliable',
 'rumor',
 'satire',
 'unknown',
 'unreliable']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
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
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.json()
    #data = await request.args['data']
    print("data:", data)
    # img_bytes = await (data['file'].read())
    # took out img_bytes
    # img = open_image(BytesIO(img_bytes))
    img = data["textField"]
    print("data['textField']", data["textField"])
    print("img:", img)
    # prediction = learn.predict(img)[0]
    a = prediction = learn.predict(img)
    print("prediction:", prediction)
    
    print(a[2])
    a = a[2].tolist()
    s = a[1] * a[11]
    p = a[6] 
    o = -(a[4] + a[5] + a[8])
    score = (1 + (a[7] + o) - (s + p)) * 50

    return JSONResponse({'result': str(prediction) + ' \n' + str(score)})



if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
