from flask import Flask, request
import io
from PIL import Image

app = Flask(__name__)


@app.route("/hello")
def hello():
    return "Hello World!"


@app.route("/xxx", methods=["POST"])
def xxx():
    # name = request.args.get("name")
    name = request.form.get("name")
    file = request.files.get("file")
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    print(name)
    image.show()
    return name


if __name__ == '__main__':
    app.run(host="0.0.0.0")
