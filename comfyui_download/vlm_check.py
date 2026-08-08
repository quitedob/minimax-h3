#!/usr/bin/env python
import requests, base64, io, sys
from PIL import Image
img = sys.argv[1]
im = Image.open(img).convert('RGB')
im.thumbnail((512,512))
buf=io.BytesIO(); im.save(buf,'JPEG',quality=60)
b64=base64.b64encode(buf.getvalue()).decode()
prompt = ("这是AI生成的视频的一帧。请描述画面内容。如果画面正常(有具体场景/物体/人物)请详细描述;"
          "如果只是噪点/花屏/混乱纹理请明确说: 这是垃圾输出。")
r=requests.post('http://127.0.0.1:9090/chat/completions', json={
    'model':'auto','max_tokens':1200,
    'messages':[{'role':'user','content':[
        {'type':'text','text':prompt},
        {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}}
    ]}]
}, timeout=120)
print(r.json()['choices'][0]['message']['content'])
