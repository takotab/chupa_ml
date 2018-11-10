from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/uc?export=download&id=1yqNfvH-Zs89jDYpopUUM3r_665ueu_e7'
model_file_name = 'model'
classes = ['Chupa Chups XXL 29g', 'Chupa Chup Xxl bubblegum lollie', 'Chupa Chups XXL 29 g',
           'Chupa Chups XXL Bubblegum', 'Chupa Chups Xxl Bubblegum', 'Chupa Chups 20 Stuks 240g',
           'Chupa Chups Flower Bouquet 228g', 'Chupa Chup Flower bouquet',
           'Chupa Chups Mini Mega 10 Stuks 120g', "Chupa Chup Mega lolly met mini's 10 stuks",
           'Chupa Chups Chupa Chups Mini meg', 'Chupa Chups Mini mega',
           'Chupa Chups Trolls Surprise 12 g',
           'Chupa Chups Despicable Me Minion Chupa + Surprise 12g',
           'Chupa Chups Bubble Gum Tutti Frutti Flavour 3 x 27,6 g', 'Chupa Chups Fruit',
           'Chupa Chups Suikervrij 10 Stuks 110g', "Chupa Chups Suikervrij lolly's",
           'Chupa Chups Suikervrij', 'Chupa Chups Chupa Chups suikervrij',
           "Chupa Chups Halloween lolly's", 'Chupa Chups Despicable Me Minion Made 10 Stuks 120g',
           'Chupa Chups Water Dinos Surprise 12g', 'Chupa Chups Mini 20 Stuks 120g',
           'CHUPA CHUPS MINI 120GR', "Chupa Chups Mini lolly's", 'Chupa Chups Mini',
           "Chupa Chups regular mini-lolly's", 'Chupa Chups Cotton Bubble Gum 11g',
           'Chupa Chups Cotton Bubble', 'Chupa Chups Cotton Bubble Gum 11 g',
           'Chupa Chup Cotton bubble gum', 'Chupa Chups Tropical', 'Chupa Chup Tropical fizz',
           'Chupa Chups Tropical Fizz', 'Chupa Chups Tropical Fizz Bag 7 x 15g',
           'Chupa Chups Tropical fizz 3D', 'Chupa Chups Tropical fizz',
           'Chupa Chups Crazy Dips Aardbei 14 g', 'Chupa Chup Crazy dips strawberry',
           'Chupa Chups Crazy Dips Aardbei 14g', 'Chupa Chups Skull',
           'Chupa Chups Skull Bag 7 x 15g', 'Chupa Chups 3d skull strawbery-lime',
           'Look O Look Flower Candy 145g', 'Look-O-Look Flower candy',
           'Look o Look Look o Look Flower candy', 'Chupa Chups Blik',
           'Chupa Chups Crazy Dips Cola 14 g', 'Chupa Chups Crazy Dips Cola',
           'Chupa Chups Crazy Dips Popping Candy + Lollipop Cola 14g',
           'Chupa Chups Crazy Dips Lemon 14g', 'Chupa Chups Melody Pops 15g',
           'Chupa Chups Melody Pops 15 g', 'Chupa Chups Melody Pops',
           "Chupa Chups The Best Of lolly's", 'Chupa Chup Lollipops the best of',
           'Chupa Chups Lollipops The Best Of 16 Stuks 192 g', 'CHUPA CHUPS BEST OF ZAK 16 ST',
           'Chupa Chups Lollipops The Best Of 16 Stuks 192g', 'Chupa Chups Lollies mix 16 stuks',
           'Chupa Chups Lollies the best of', 'Chupa Chups The best of', 'Chupa Chups Lollipops',
           'Chupa The best of chupa chups lolli pops', 'Chupa Chups lolly mix',
           'Chupa Chup Lollipops fruit', 'Chupa Chups Lollipops fruit',
           'Chupa Chups Fruit 14 Stuks', 'LOLLIES FRUIT', "Chupa Chups Fruit lolly's",
           'Chupa Chups Fruit 16 Stuks 192g', 'Chupa Chups Fruit 16 Stuks 192 g',
           'CHUPA CHUPS FRUIT ZAK 16 ST', 'Chupa Chups Chupa Chups fruit',
           'Chupa Chups Lollies fruit']
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
