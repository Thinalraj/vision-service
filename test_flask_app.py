"""Small browser test client for the FastAPI service.

Start FastAPI on 8000 first, then run this file on 5000:
    python test_flask_app.py
"""
import os

import requests
from flask import Flask, Response, jsonify, render_template_string, request

FASTAPI_URL = os.getenv("VISION_API_URL", "http://127.0.0.1:8000")
app = Flask(__name__)

PAGE = """
<!doctype html><title>Vision API test</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto}#stage{position:relative;display:inline-block}#preview{max-width:960px;border:1px solid #aaa}#aoi{position:absolute;border:2px solid #00aaff;display:none;pointer-events:none}img{max-width:100%;border:1px solid #aaa}button,select{padding:8px;margin:6px 4px 12px 0}pre{background:#f2f2f2;padding:12px}</style>
<h1>Vision API test client</h1>
<select id="type"><option>coin</option><option>bar</option><option>chain</option><option>ring</option><option>bangle</option><option>ornament</option></select>
<button onclick="getImage()">Get image</button><button onclick="measure()">Get size</button><button onclick="detected()">Show detected object</button><button onclick="loadSettings()">Load saved settings</button>
<button onclick="sendAoi()">Send AOI</button><span> Drag on the image to draw AOI.</span>
<div id="stage"><img id="preview" alt="Latest camera image"><div id="aoi"></div></div>
<pre id="result">No measurement yet</pre>
<script>
let rect=null, start=null;
const img=document.getElementById('preview'), box=document.getElementById('aoi');
function getImage(){img.src='/proxy/image?t='+Date.now();}
function detected(){img.src='/proxy/image/detected?t='+Date.now();}
async function measure(){const t=document.getElementById('type').value;const r=await fetch('/proxy/size?type='+t);document.getElementById('result').textContent=await r.text();}
async function loadSettings(){const t=document.getElementById('type').value;const r=await fetch('/proxy/settings?type='+t);document.getElementById('result').textContent=await r.text();}
img.addEventListener('mousedown',e=>{const r=img.getBoundingClientRect();start={x:e.clientX-r.left,y:e.clientY-r.top};rect={...start,w:0,h:0};box.style.display='block';});
img.addEventListener('mousemove',e=>{if(!start)return;const r=img.getBoundingClientRect();rect.w=e.clientX-r.left-start.x;rect.h=e.clientY-r.top-start.y;box.style.left=Math.min(start.x,start.x+rect.w)+'px';box.style.top=Math.min(start.y,start.y+rect.h)+'px';box.style.width=Math.abs(rect.w)+'px';box.style.height=Math.abs(rect.h)+'px';});
window.addEventListener('mouseup',()=>{start=null;});
async function sendAoi(){if(!rect||Math.abs(rect.w)<2||Math.abs(rect.h)<2){alert('Draw an AOI first');return;}const sx=img.naturalWidth/img.clientWidth,sy=img.naturalHeight/img.clientHeight;const x=Math.round(Math.min(rect.x,rect.x+rect.w)*sx),y=Math.round(Math.min(rect.y,rect.y+rect.h)*sy),w=Math.round(Math.abs(rect.w)*sx),h=Math.round(Math.abs(rect.h)*sy);const r=await fetch('/proxy/aoi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:document.getElementById('type').value,x,y,width:w,height:h})});document.getElementById('result').textContent=await r.text();}
</script>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/proxy/image")
def proxy_image():
    result = requests.get(FASTAPI_URL + "/image", timeout=15)
    return Response(result.content, status=result.status_code,
                    content_type=result.headers.get("content-type", "image/jpeg"))


@app.get("/proxy/image/detected")
def proxy_detected_image():
    result = requests.get(FASTAPI_URL + "/image/detected", timeout=15)
    return Response(result.content, status=result.status_code,
                    content_type=result.headers.get("content-type", "image/jpeg"))


@app.get("/proxy/size")
def proxy_size():
    object_type = request.args.get("type", "coin")
    result = requests.get(FASTAPI_URL + "/size", params={"type": object_type}, timeout=15)
    try:
        return jsonify(result.json()), result.status_code
    except ValueError:
        return result.text, result.status_code


@app.get("/proxy/settings")
def proxy_settings():
    result = requests.get(FASTAPI_URL + "/settings",
                          params={"type": request.args.get("type", "coin")}, timeout=15)
    try:
        return jsonify(result.json()), result.status_code
    except ValueError:
        return result.text, result.status_code


@app.post("/proxy/aoi")
def proxy_aoi():
    result = requests.post(FASTAPI_URL + "/aoi", json=request.get_json(), timeout=15)
    try:
        return jsonify(result.json()), result.status_code
    except ValueError:
        return result.text, result.status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("FLASK_PORT", "5000")), debug=True)
