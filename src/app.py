# 必要なモジュールをインポート
import numpy as np
import torch
from washer import transform, AutoEncoder
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
import torch.nn.functional as F

# 学習済みモデルを元に推論を行う
def predict(img):
    # ネットワークの準備
    net = AutoEncoder().cpu().eval()
    # 学習済みモデルの重みを読み込み
    net.load_state_dict(torch.load('./src/washer_cnn.pt', map_location=torch.device('cpu')))
    # データの前処理
    img = transform(img)
    img = img[0:3]
    img_r = img.unsqueeze(0)
    y = net(img_r)
    yn = torch.reshape(y, (3, 112, 112))
    yn = yn.to('cpu').detach().numpy().copy()
    yn = np.transpose(yn, (1, 2, 0))
    
    z = np.transpose(img, (1, 2, 0))
    # z = z.reshape(z.shape[0], z.shape[1])
    imgn = z.to('cpu').detach().numpy().copy()
    
    # diff = np.sum(np.absolute(yn - imgn))
    diff = F.mse_loss(y, img_r, reduction='none')
    diff = np.sum(diff.detach().numpy(), axis=(1,2,3))

    return diff

# AEの差分の大きさでOK/NGを判定する
def getResult(diff):
    if diff < 200.0:
        return 'OKです'
    elif diff >= 200.0:
        return 'だめです'

# Flask のインスタンス化
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'gif', 'jpeg'])

# 拡張子が適切かどうかチェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

# URLにアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # POSTメソッドの定義
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect((request.url))
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allowed_file(file.filename):

            # 画像読み込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file)
            # 画像リサイズ
            img_resize = image.resize((112, 144))
            # 入力された画像に対して推論
            pred = predict(img_resize)
            # 画像データをバッファに書き込む
            img_resize.save(buf, 'png')
            # バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            base64_data = 'data:image/png;base64,{}'.format(base64_str)
            
            washer_ = getResult(pred)
            return render_template('result.html', washer=washer_, image=base64_data)
        return redirect(request.url)

    # GETメソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
