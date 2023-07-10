# OpenVINO_NeuralStyleTransferWithBGRemover
 OpenVINO_miniproject<br>
2023.07.05 ~ 2023.07.07<br>
Intel Edge AI SW Developer @ kccistc<br>
http://www.kccistc.net/

## Requirement

```
* 9th generation Intel® CoreTM processor onwards
* At least 32GB RAM
* Windows 10
* Python 3.9
```

## Clone code

```shell
git clone https://github.com/jangyj405/OpenVINO_NeuralStyleTransferWithBGRemover
```

## Prerequite

```shell
python -m venv .venv
.venv\Scripts\activate

cd OpenVINO_NeuralStyleTransferWithBGRemover
python -m pip install -U pip
python -m pip install -r requirements.txt
python model_downloaders.py
```

## Steps to run

```shell
.venv\bin\activate
cd OpenVINO_NeuralStyleTransferWithBGRemover
python flask_app.py
```
Open "http://127.0.0.1:5000" in your browser
## Output

![./figures/output0.jpg](./figures/output0.jpg)

Upload your image and click the "submit" button

![./figures/output1.jpg](./figures/output1.jpg)

# Appendix
## Concept
![figures/concept.jpg](figures/concept.jpg)

## BG Remover
![figures/bg_remover.jpg](figures/bg_remover.jpg)
OpenVINO Notebooks 205-vision-background-removal<br>
U2-Net : https://github.com/xuebinqin/U-2-Net

## Neural Style Transfer
OpenVINO Notebooks 404-style-transfer-webcam<br>
https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style
![figures/neural_style_transfer.jpg](figures/neural_style_transfer.jpg)

## Server
![figures/server.jpg](figures/server.jpg)

## OpenVINO Async API
OpenVINO Notebooks 115-async-api
![figures/async_api.jpg](figures/async_api.jpg)

## Class AIProcessor
![figures/ai_processor.jpg](figures/ai_processor.jpg)

## Conclusion
![figures/conclusion.jpg](figures/conclusion.jpg)