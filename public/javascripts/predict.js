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
    var imgScaled = config.then(data => imgCropped.mul(2).sub(1).mul(tf.scalar(data.IMAGE_SCALE)));
    return imgScaled;
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

console.log(config)
var img = new Image()
img.src = '../images/xray.jpeg'
var img2 = resizeImage(img);
//
var imgNode = makeImageNode(img2);
imgNode.then(res =>  document.body.appendChild(res));
