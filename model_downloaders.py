from notebook_utils import download_file
from pathlib import Path
from collections import namedtuple
from model.u2net import U2NET, U2NETP
import os
from openvino.tools import mo
import torch


styles = [
    'MOSAIC',
    'RAIN-PRINCESS',
    'CANDY',
    'UDNIE',
    'POINTILISM'
]
def download_style_transfers():
    #download style models

    base_model_dir = "model"
    base_url = "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model"

    # Selected ONNX model will be downloaded in the path
    for s in styles:
        model_path = Path(base_model_dir,f"{s.lower()}-9.onnx")
        if not model_path.exists():
            style_url = f"{base_url}/{model_path}"
            print(f'start: download {s} model from {style_url}')
            download_file(style_url, directory=base_model_dir)
            print(f'end: download {s} model from {style_url}')
        else:
            print(f'{s} model is already exist')


def download_bg_remove_model():
    #download bg removal model
    model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])
    u2net_lite = model_config(
        name="u2net_lite",
        url="https://drive.google.com/uc?id=1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy",
        model=U2NETP,
        model_args=(),
    )    
    u2net_model = u2net_lite
    MODEL_DIR = "model"
    model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")
    if not model_path.exists():
        import gdown
        os.makedirs(name=model_path.parent, exist_ok=True)
        print("Start downloading model weights file... ")
        with open(model_path, "wb") as model_file:
            gdown.download(url=u2net_model.url, output=model_file)
            print(f"Model weights have been downloaded to {model_path}")
    
    net = u2net_model.model(*u2net_model.model_args)
    net.eval()

    # Load the weights.
    print(f"Loading model weights from: '{model_path}'")
    net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))
    torch.onnx.export(net, torch.zeros((1,3,512,512)), "model/u2net.onnx")



def superresolution_model():
    #download superresolution model
    # 1032: 4x superresolution, 1033: 3x superresolution
    model_name = 'single-image-super-resolution-1032'

    base_model_dir = Path('./model').expanduser()

    model_xml_name = f'{model_name}.xml'
    model_bin_name = f'{model_name}.bin'

    model_xml_path = base_model_dir / model_xml_name
    model_bin_path = base_model_dir / model_bin_name

    if not model_xml_path.exists():
        base_url = f'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/'
        model_xml_url = base_url + model_xml_name
        model_bin_url = base_url + model_bin_name

        download_file(model_xml_url, model_xml_name, base_model_dir)
        download_file(model_bin_url, model_bin_name, base_model_dir)
    else:
        print(f'{model_name} already downloaded to {base_model_dir}')


def download_models():
    download_style_transfers()
    download_bg_remove_model()
    superresolution_model()
    

if __name__ == "__main__":
    download_models()