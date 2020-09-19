var body = document.getElementById('main');
var para = document.createElement('p');

body.appendChild(para);

AEMODEL_PATH = 'file:///home/mfromano/Research/pathray/pathray/models/ae-chest-savedmodel-64-512';

async function loadModel(path) {
    var model = await tf.loadLayersModel(path);
    return model
}
const model = loadModel(AEMODEL_PATH)
model.then(data => para.appendChild(document.createTextNode(data)))