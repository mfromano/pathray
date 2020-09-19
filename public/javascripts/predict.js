var body = document.getElementById('main');
var para = document.createElement('p');
body.appendChild(para);
tens = tf.tensor([1, 2, 3, 4])
tens.data().then(data => para.appendChild(document.createTextNode(data)));


MODEL_PATH = 'https://mlmed.github.io/tools/xray/models/xrv-all-45rot15trans15scale';
AEMODEL_PATH = 'https://mlmed.github.io/tools/xray/models/ae-chest-savedmodel-64-512';

async function loadModel(path) {
    var model = await tf.loadLayersModel(path);
    return model
}
