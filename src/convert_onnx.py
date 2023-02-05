import numpy as np
import torch.nn as nn
import torch.onnx
from torchvision import models
import onnx
from onnx import shape_inference
import onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import timm
import os
from models import ImageModel

# CreateNetwork should be modified by custom deep-learning model
def CreateNetwork():
    net = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=100)
    return net

def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
    # Reference : https://gaussian37.github.io/python-basic-numpy-snippets/
    flag = False
    try : 
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        #print(layer_name + ": no difference.")
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag

        
# parameters
channel = 3
height = 256
width = 256
onnx_path = "tf_efficientnet_b4_ns_1207_193844.onnx"

# ① 사용할 딥러닝 네트워크를 불러온 뒤 평가 모드로 설정합니다.
net = CreateNetwork()
#net = ImageModel(model_name='tf_efficientnet_b4_ns', class_n=100, mode='valid')
device = 'cuda'
checkpoint_path = '/home/kt_dev/food-kt/ckpt/tf_efficientnet_b4_ns_1207_193844/ckpt_best.pt'
check_point = torch.load(checkpoint_path, map_location=device)
net.load_state_dict(check_point['model_state_dict'], strict=True)
net = net.to(device)
net.eval()
# net.load_state_dict(torch.load('/home/kt_dev/food-kt/ckpt/tf_efficientnet_b4_ns_1207_193844/ckpt_best.pt')['model_state_dict'], strict=True)
# net.eval()

# ② torch 모델을 이용하여 onnx 모델을 생성합니다.
# (B, C, H, W) 의 dimension을 가지는 것으로 가정함
dummy_data = torch.empty(1, channel, height, width, dtype = torch.float32)
#torch.onnx.export(net, dummy_data, onnx_path, input_names = ['input'], output_names = ['output'])

#dummy_data = torch.empty(channel, height, width, dtype = torch.float32)
torch.onnx.export(net,             # 실행될 모델
                    dummy_data,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                    onnx_path,   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                    export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                    opset_version=11,          # 모델을 변환할 때 사용할 ONNX 버전
                    do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                    input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                    output_names = ['output']) # 모델의 출력값을 가리키는 이름
                    #dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                    #                'output' : {0 : 'batch_size'}})
# torch_out
torch_out = net(dummy_data)

# ③ 생성한  onnx 모델을 다시 블루어와서 torch 모델과 onnx 모델의 weight를 비교합니다.
# 입력 받은 onnx 파일 경로를 통해 onnx 모델을 불러옵니다.
onnx_model = onnx.load(onnx_path)

# onnx 모델의 정보를 layer 이름 : layer값 기준으로 저장합니다.
onnx_layers = dict()
for layer in onnx_model.graph.initializer:
    onnx_layers[layer.name] = numpy_helper.to_array(layer)

# torch 모델의 정보를 layer 이름 : layer값 기준으로 저장합니다.
torch_layers = {}
for layer_name, layer_value in net.named_modules():
    torch_layers[layer_name] = layer_value   
#print(f"onnx_layers.keys() is {onnx_layers.keys()}\n\n torch_layers.keys() is {torch_layers.keys()}")
# onnx와 torch 모델의 성분은 1:1 대응이 되지만 저장하는 기준이 다릅니다.
# onnx와 torch의 각 weight가 1:1 대응이 되는 성분만 필터합니다.
onnx_layers_set = set(onnx_layers.keys())
# onnx 모델의 각 layer에는 .weight가 suffix로 추가되어 있어서 문자열 비교 시 추가함
torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

difference_flag = False
for layer_name in filtered_onnx_layers:
    onnx_layer_name = layer_name
    torch_layer_name = layer_name.replace(".weight", "")
    onnx_weight = onnx_layers[onnx_layer_name]
    torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
    flag = compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
    difference_flag = True if flag == True else False
    
# ④ onnx 모델에 기존 torch 모델과 다른 weight가 있으면 전체 update를 한 후 새로 저장합니다.
if difference_flag:
    print("update onnx weight from torch model.")
    for index, layer in enumerate(onnx_model.graph.initializer):
        layer_name = layer.name
        if layer_name in filtered_onnx_layers:
            onnx_layer_name = layer_name
            torch_layer_name = layer_name.replace(".weight", "")
            onnx_weight = onnx_layers[onnx_layer_name]
            torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
            copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
            onnx_model.graph.initializer[index].CopyFrom(copy_tensor)
    
    print("save updated onnx model.")
    onnx_new_path = os.path.dirname(os.path.abspath(onnx_path)) + os.sep + "updated_" + os.path.basename(onnx_path)
    onnx.save(onnx_model, onnx_new_path)

# ⑤ 최종적으로 저장된 onnx 모델을 불러와서 shape 정보를 추가한 뒤 다시 저장합니다.
if difference_flag:
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_new_path)), onnx_new_path)
else:
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)

ort_session = ort.InferenceSession(onnx_path)
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_data)}
ort_outs = ort_session.run(None, ort_inputs)



# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
cal = np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print(f"cal is {cal}")
print("Exported model has been tested with ONNXRuntime, and the result looks good!")