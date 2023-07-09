from openvino.runtime import Core, AsyncInferQueue
from model_downloaders import styles
from openvino.runtime import CompiledModel
from openvino.tools import mo
import openvino.runtime as ov
import numpy as np
from util import *
import cv2
from datetime import datetime
import threading
import random

class AIProcessor():
    def __init__(self):
        self.ie = Core()
        self.bg_remover = self.load_bg_remover()
        self.style_transfers = self.load_transfers()
        self.upscaler = self.load_upscaler()
        self.bg_pool = BackgroundImages()
    def load_transfers(self,)->dict:
        transfers = {}
        for s in styles:
            style_transfer_path = f'model/{s.lower()}-9.onnx'
            model = self.ie.read_model(model=style_transfer_path)
            compiled_model = self.ie.compile_model(model=model, device_name='AUTO')
            transfers[s] = compiled_model
        return transfers

    def load_bg_remover(self)->CompiledModel:
        bg_remover_path = 'model/u2net.onnx'
        model_ir = mo.convert_model(
            bg_remover_path,
            mean_values=[123.675, 116.28 , 103.53],
            scale_values=[58.395, 57.12 , 57.375],
            compress_to_fp16=True
        )
        compiled_model = self.ie.compile_model(model=model_ir, device_name='AUTO')
        return compiled_model
    
    def load_upscaler(self)->CompiledModel:
        upscaler_path = 'model/single-image-super-resolution-1032.xml'
        model = self.ie.read_model(model=upscaler_path)
        compiled_model = self.ie.compile_model(model=model, device_name='AUTO')
        return compiled_model

    def callback_bgr(self ,userdata):
        print('inside bgr callback')
        req, origin, bg, styler, task_result = userdata
        res = req.get_output_tensor(0).data[0]
        frame = postprocessing_bg_removal(res, origin, bg)
        
        img = preprocessing(frame, 224, 224)
        compiled_model = styler
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        req_style = compiled_model.create_infer_request()
        req_style.set_tensor(input_layer, ov.Tensor(img))
        req_style.set_callback(callback = self.callback_styler, userdata = (req_style, origin, task_result))
        req_style.start_async()

    


    def callback_styler(self, userdata):
        print('inside styler callback')
        req, image, task_result = userdata
        res = req.get_output_tensor(0).data[0]
        frame = convert_result_to_image(res, image.shape[0], image.shape[1])
        # Encode numpy array to jpg


        #cv2.imwrite(f"{random.randint(0, 300)}.jpg", frame)
        task_result['done'] = True
        task_result['image'] = frame


    def do_inference_async(self, image=None, style = 'MOSAIC', bg=0, task_result = None):
        input_image = preprocessing_bg_removal(image)
        input_image = input_image.astype(np.float32)
        input_layer_bg_remover = self.bg_remover.input(0)
        output_layer_bg_remover = self.bg_remover.output(0)

        req_bgr = self.bg_remover.create_infer_request()
        req_bgr.set_tensor(input_layer_bg_remover, ov.Tensor(input_image))
        req_bgr.set_callback(callback = self.callback_bgr, 
                             userdata = (req_bgr, image, self.bg_pool.get_bg(bg), self.style_transfers[style], task_result))
        #req_bgr.set_callback(callback = local_callback_bgr, userdata=(req_bgr, image))
        req_bgr.start_async()
        #req_bgr.wait()
    

    def do_inference(self, image=None, style = 'MOSAIC', bg=None, skip_upscale=False):
        
        '''
        resized_image = cv2.resize(src=image, dsize=(512, 512))
        # Convert the image shape to a shape and a data type expected by the network
        # for OpenVINO IR model: (1, 3, 512, 512).
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
        '''
        input_image = preprocessing_bg_removal(image)
        input_layer_bg_remover = self.bg_remover.input(0)
        output_layer_bg_remover = self.bg_remover.output(0)
        bg_remove_result = self.bg_remover([input_image])[output_layer_bg_remover]
        
        resized_result = np.rint(
            cv2.resize(src=np.squeeze(bg_remove_result), dsize=(image.shape[1], image.shape[0]))
        ).astype(np.uint8)
        bg_removed_result = image.copy()
        bg_removed_result[resized_result == 0] = 255
        #cv2.imshow('bg_removed_result', cv2.cvtColor(bg_removed_result, cv2.COLOR_RGB2BGR))
        #cv2.imshow('resized_result', resized_result)
        background_image = cv2.cvtColor(src=cv2.imread(filename='static/sample_data/lena.jpg'), code=cv2.COLOR_BGR2RGB)
        background_image = cv2.resize(src=background_image, dsize=(image.shape[1], image.shape[0]))

        background_image[resized_result == 1] = 0
        new_image = background_image + bg_removed_result
        #cv2.imshow('new_image', cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))


        img = preprocessing(new_image, 224, 224)
        compiled_model = self.style_transfers[style]
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        stylized_image = compiled_model([img])[output_layer]
        
        result_image = convert_result_to_image(stylized_image, image.shape[0], image.shape[1])
        #cv2.imshow("test",cv2.cvtColor(result_image,cv2.COLOR_RGB2BGR))


        original_image_key, bicubic_image_key = self.upscaler.inputs
        output_key = self.upscaler.output(0)

        # Get the expected input and target shape. The `.dims[2:]` function returns the height
        # and width.The `resize` function of  OpenCV expects the shape as (width, height),
        # so reverse the shape with `[::-1]` and convert it to a tuple.
        input_height, input_width = list(original_image_key.shape)[2:]
        target_height, target_width = list(bicubic_image_key.shape)[2:]       
            
        resized_image = cv2.resize(src=result_image, dsize=(input_width, input_height))
        input_image_original = np.expand_dims(resized_image.transpose(2, 0, 1), axis=0)

        # Resize and reshape the image to the target shape with bicubic
        # interpolation.
        bicubic_image = cv2.resize(
            src=result_image, dsize=(target_width, target_height), interpolation=cv2.INTER_CUBIC
        )
        input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)

        # Do inference.
        result = self.upscaler(
            {
                original_image_key.any_name: input_image_original,
                bicubic_image_key.any_name: input_image_bicubic,
            }
        )[output_key]    
            
        result_frame = convert_upscale_result_to_image(result=result)
        result_frame = cv2.resize(result_frame, dsize=(image.shape[1], image.shape[0]))
        #cv2.imshow("upscale", cv2.cvtColor(result_frame,cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        pass

if __name__ == "__main__":
    image = cv2.cvtColor(
        src=cv2.imread(filename='static/sample_data/pexels-photo-16075175.jpeg'),
        code=cv2.COLOR_BGR2RGB,
    )
    
    processor = AIProcessor()
    #for _ in range(100000):
    #    processor.do_inference(image = image, style=styles[0], bg=0)
    for _ in range(100000):
        processor.do_inference_async(image = image, style=styles[0], bg=0, task_result={})
    while(1):
        pass    
    processor.do_inference_async(image = image, style=styles[0], bg=0, task_result={})
    t1 = threading.Thread(target = processor.do_inference_async,args=(image, styles[1], 1, {}))
    t2 = threading.Thread(target = processor.do_inference_async,args=(image, styles[2], 2, {}))
    t3 = threading.Thread(target = processor.do_inference_async,args=(image, styles[3], 3, {}))
    
    t1.start()
    t2.start()
    t3.start()
    

    
    import asyncio
    
    async def temp_async():
        task_results = {'done':False, 'image':None}
        tempThread = threading.Thread(target = processor.do_inference_async,args=(image, styles[1], 1, task_results))
        tempThread.start()
        while not task_results['done']:
            print('task not done')
            await asyncio.sleep(1)
        pass
    
    asyncio.run(temp_async())
    

    while 1:
        pass

