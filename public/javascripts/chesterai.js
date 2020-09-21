//This code is released under a Attribution-NonCommercial 4.0 Generic (CC BY-NC 4.0) license;
// https://creativecommons.org/licenses/by-nc/4.0/
// prepare_image_resize_crop used, unchanged, courtesy of https://github.com/mlmed/chester-xray/blob/master/
function prepare_image_resize_crop(imgElement, size){

    orig_width = imgElement.width
    orig_height = imgElement.height
    if (orig_width < orig_height){
        imgElement.width = size
        imgElement.height = Math.floor(size*orig_height/orig_width)
    }else{
        imgElement.height = size
        imgElement.width = Math.floor(size*orig_width/orig_height)
    }

    console.log("img wxh: " + orig_width + ", " + orig_height + " => " + imgElement.width + ", " + imgElement.height)

    img = tf.browser.fromPixels(imgElement).toFloat();

    hOffset = Math.floor(img.shape[1]/2 - size/2)
    wOffset = Math.floor(img.shape[0]/2 - size/2)

    img_cropped = img.slice([wOffset,hOffset],[size,size])
    img_cropped = img_cropped.mean(2).div(255)

    return img_cropped
}

async function computeGrads_real(model, img, idx){
    //cache computation
        layer = await tf.tidy(() => {

            var gradfun = tf.grad(x => model.predict(x).reshape([-1]).gather(idx))
            const grads = gradfun(img);
            const layer = grads.mean(0).abs().max(0)
            return layer.div(layer.max())
        })
    var body = document.getElementById('main');
    var canvasOverlay =document.createElement("canvas");
    canvasOverlay.setAttribute('id','xray2')
    body.appendChild(canvasOverlay);
    await tf.browser.toPixels(layer, canvasOverlay);
    ctx = await canvasOverlay.getContext("2d");
    d = await ctx.getImageData(0, 0, canvasOverlay.width, canvasOverlay.height);
    await applyColorMap(d.data);
    ctx.putImageData(d,0,0);
    return d.data;
}

function rescale(arr) {
    var max = Math.max.apply(null, arr);
    var min = Math.min.apply(null, arr);
    return arr.map(x => (x-min)/(max-min));
}

function applyColorMap(grayScaleImage) {
    const colorMapSize = RGB_COLORMAP.length / 3;
    rescale(grayScaleImage);
    for (let i = 0; i < grayScaleImage.length; i+=4) {
        const pixelValue = grayScaleImage[i]/255;
        const row = Math.floor(pixelValue * colorMapSize);
        for (let q = 0; q < 4; q++) {
            if (q !== 3) {
                grayScaleImage[i+q] = 255 * RGB_COLORMAP[3 * row + q];
            } else {
                grayScaleImage[i+q] = 100;
            }
        }
    }
    return grayScaleImage;
}