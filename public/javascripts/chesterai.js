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