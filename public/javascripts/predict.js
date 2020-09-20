var body = document.getElementById('main');
var canvas=document.createElement("canvas");
var ctx=canvas.getContext("2d");

AEMODEL_PATH = 'https://mlmed.github.io/tools/xray/models/xrv-all-45rot15trans15scale/';

async function loadModel(path) {
    return await tf.loadGraphModel(path + 'model.json')
}

var config = $.getJSON(AEMODEL_PATH + 'config.json', function(json) {
        return json;
    });

const model = loadModel(AEMODEL_PATH)

async function resizeImage(img) {
    var imgCropped = await config.then(data => prepare_image_resize_crop(img, data.IMAGE_SIZE));
    return config.then(data => imgCropped.mul(2).sub(1).mul(tf.scalar(data.IMAGE_SCALE)));
}

async function featureScale(arr) {
    var max = await Math.max.apply(null, arr);
    var min = await Math.min.apply(null, arr);
    return arr.map(x => (x-min)*255/(max-min));
}

function expandArray(arr) {
    var newArr = new Uint8ClampedArray(arr.length*4);
    var newArrInd = 0;
    for (var i=0; i<arr.length; i++) {
        newArr[newArrInd] = arr[i];
        newArr[newArrInd+1] = arr[i];
        newArr[newArrInd+2] = arr[i];
        newArr[newArrInd+3] = arr[1];
        newArrInd += 4;
    }
    return newArr;
}

async function makeImageNode(im_age) {
    canvas.width = await im_age.then(res => res.shape[0]);
    canvas.height = await im_age.then(res => res.shape[1]);
    var rescaled = await im_age.then(res => featureScale(res.dataSync()));
    var arr = new Uint8ClampedArray(rescaled);
    arr = expandArray(arr);
    var imdat = new ImageData(arr, canvas.width, canvas.height);
    ctx.putImageData(imdat, 10, 10);
    var img2 = new Image();
    img2.src = canvas.toDataURL();
    return img2;
}

async function predict(model, croppedImage) {
    return await model.then(res => res.predict(croppedImage.reshape([1,1,croppedImage.shape[0], croppedImage.shape[1]])));
}

async function appendTableDom(results) {
    var labels = await config.then(res => res.LABELS);
    var tbl = document.createElement('table');
    await tbl.createTHead().insertRow().appendChild(document.createElement('th')
        .appendChild(document.createTextNode('Results')));
    var tblBody = await tbl.createTBody()
    for (var i=0; i < labels.length; i++) {
        var row = tblBody.insertRow();
        await row.insertCell().appendChild(document.createTextNode(labels[i]));
        await results.then(res => row.insertCell().appendChild(document.createTextNode(res[i]*100)));
    }
    body.appendChild(tbl);
}

let img = new Image()
img.src = '../images/atelectasis.jpeg'
let img2 = resizeImage(img);
//
let imgNode = makeImageNode(img2);
imgNode.then(res =>  document.body.appendChild(res));
let pred = img2.then(res => predict(model, res));
pred.then(res => appendTableDom(res.data()))