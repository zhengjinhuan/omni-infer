import sys
import random
import base64
from pathlib import Path
from io import BytesIO
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image


def generate_random_image(seed: int, height: int, width: int):
    random.seed(seed)
    np.random.seed(seed)
    # 生成随机图片数据，范围在0到255，表示RGB值
    random_image_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    # 创建图片
    random_image = Image.fromarray(random_image_data)

    # 参考 https://blog.51cto.com/u_16213466/11902843
    img_buffer = BytesIO()
    random_image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_encode = base64.b64encode(byte_data)
    base64_str = base64_encode.decode(("utf-8"))

    return base64_str


def get_api_url(host, port, url):
    if url is not None and len(url) > 0:
        return url
    api_url = f"http://{host}:{port}/v1/chat/completions"
    return api_url


def get_request_data(
        image_base64: str,
        prompt: str,
        output_len: int,
        app_code: str = None,
        model: str = None,
        served_model_name: str = None,
):
    if app_code is not None and len(app_code) > 0:
        headers = {"User-Agent": "Benchmark Client",
                   'Content-Type': 'application/json',
                   'X-Apig-AppCode': app_code}
    else:
        headers = {"User-Agent": "Benchmark Client"}

    if served_model_name is None:
        served_name = model
    else:
        served_name = served_model_name
    pload = {
        "model": served_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": True,
    }
    confirm_error_output = True
    return headers, pload, confirm_error_output
