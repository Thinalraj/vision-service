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
<style>body{font:16px sans-serif;max-width:900px;margin:30px auto} img{max-width:100%;border:1px solid #aaa} button,select{padding:8px;margin:6px 4px 12px 0} pre{background:#f2f2f2;padding:12px}</style>
<h1>Vision API test client</h1>
<button onclick="document.getElementById('preview').src='/proxy/image?t='+Date.now()">Get image</button>
<select id="type"><option>coin</option><option>bar</option><option>chain</option><option>ring</option><option>bangle</option><option>ornament</option></select>
<button onclick="measure()">Get size</button>
<br><img id="preview" alt="Latest camera image">
<pre id="result">No measurement yet</pre>
<script>async function measure(){const t=document.getElementById('type').value;const r=await fetch('/proxy/size?type='+t);document.getElementById('result').textContent=await r.text();}</script>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/proxy/image")
def proxy_image():
    result = requests.get(FASTAPI_URL + "/image", timeout=15)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("FLASK_PORT", "5000")), debug=True)
