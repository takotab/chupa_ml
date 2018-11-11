from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1lyX8dXbLq1DBb2PiIeEfi5rsQgODpZPG'
model_file_name = 'model'
classes = ['Chupa Chups Crazy Dips Cola 14 g',
           'Chupa Chups Water Dinos Surprise 12g',
           'Chupa Chups Suikervrij 10 Stuks 110g',
           'Chupa Chups Crazy Dips Aardbei 14 g',
           'Chupa Chups Melody Pops 15g',
           'Chupa Chups Skull',
           'Chupa Chups Trolls Surprise 12 g',
           'Chupa Chups Cotton Bubble Gum 11g',
           "Chupa Chups Halloween lolly's",
           'Look O Look Flower Candy 145g',
           'Chupa Chups Mini 20 Stuks 120g',
           'Chupa Chups Crazy Dips Lemon 14g',
           'Chupa Chups 20 Stuks 240g',
           'Chupa Chups Bubble Gum Tutti Frutti Flavour 3 x 27,6 g',
           'Chupa Chups XXL 29g',
           'Chupa Chups Blik',
           'Chupa Chup Lollipops fruit',
           'Chupa Chups Fruit',
           'Chupa Chups Flower Bouquet 228g',
           'Chupa Chups Tropical',
           'Chupa Chups Despicable Me Minion Made 10 Stuks 120g',
           'Chupa Chups Mini Mega 10 Stuks 120g',
           "Chupa Chups The Best Of lolly's"]

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins = ['*'],
                   allow_headers = ['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory = 'app/static'))


async def download_file(url, dest):
    if dest.exists():
        return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_learner():
    await download_file(model_file_url, path / 'models' / f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
                                                    tfms = get_transforms(),
                                                    size = 224
                                                    ).normalize(imagenet_stats)
    learn = create_cnn(data_bunch, models.resnet34, pretrained = False)
    learn.load(model_file_name)
    return learn


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
def index(request):
    html = path / 'view' / 'index.html'
    return HTMLResponse(html.open().read())


@app.route('/analyze', methods = ['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await(data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host = '0.0.0.0', port = 8080)
