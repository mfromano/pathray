var body = document.getElementById('main');
var canvas=document.createElement("canvas");
canvas.setAttribute('id','xray')
body.appendChild(canvas);
var ctx=canvas.getContext("2d");

MODEL_PATH = 'https://mlmed.github.io/tools/xray/models/xrv-all-45rot15trans15scale/';

async function resizeImage(img, config) {
    let imgCropped = await prepare_image_resize_crop(img, config.IMAGE_SIZE);
    return imgCropped.mul(2).sub(1).mul(tf.scalar(config.IMAGE_SCALE));
}

async function featureScale(arr) {
    let max = await Math.max.apply(null, arr);
    let min = await Math.min.apply(null, arr);
    return arr.map(x => (x-min)*255/(max-min));
}

function expandArray(arr) {
    let newArr = new Uint8ClampedArray(arr.length*4);
    let newArrInd = 0;
    for (let i=0; i<arr.length; i++) {
        newArr[newArrInd] = arr[i];
        newArr[newArrInd+1] = arr[i];
        newArr[newArrInd+2] = arr[i];
        newArr[newArrInd+3] = 255;
        newArrInd += 4;
    }
    return newArr;
}

async function makeImageNode(im_age) {
    canvas.width = im_age.shape[0];
    canvas.height = im_age.shape[1];
    var rescaled = await featureScale(im_age.dataSync());
    var arr = new Uint8ClampedArray(rescaled);
    arr = expandArray(arr);
    const imageData = new ImageData(arr, canvas.width, canvas.height);
    ctx.putImageData(imageData, 0, 0);
}

async function run(im_g) {
    let config = await $.getJSON(MODEL_PATH + 'config.json')
    let model = await tf.loadGraphModel(MODEL_PATH + 'model.json');
    let imgResized = await resizeImage(im_g, config);
    await makeImageNode(imgResized);
    await predictAndPlot(model, imgResized, config);
}

async function predict(model, croppedImage) {
    croppedImage = croppedImage.reshape([1,1,croppedImage.shape[0], croppedImage.shape[1]])
    return model.predict(croppedImage);
}

function appendTableDom(results, config) {
    const labels = config.LABELS;
    const tbl = document.createElement('table');
    tbl.setAttribute('id', 'resultsTable')
    tbl.createTHead().insertRow().appendChild(document.createElement('th')
        .appendChild(document.createTextNode('Results')));
    const tblBody = tbl.createTBody()
    for (var i=0; i < labels.length; i++) {
        var row = tblBody.insertRow();
        row.insertCell().appendChild(document.createTextNode(labels[i]));
        row.insertCell().appendChild(document.createTextNode(results[i]*100));
    }
    body.appendChild(tbl);
}

async function applyClassActivationMap(mod, cla_ss=2, img) {
    return gradClassActivationMap(mod, cla_ss, img);
}

async function predictAndPlot(mod, im_age, config) {
    let pred = await predict(mod, im_age);
    var data = await pred.data()
    appendTableDom(data, config);
    let maxClass = pred.dataSync().indexOf(Math.max(...pred.dataSync()));
    im_age = im_age.reshape([1,1,im_age.shape[0], im_age.shape[1]])
    return await computeGrads_real(mod, im_age, maxClass);
}