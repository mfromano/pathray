/*

MIT License

Copyright (c) 2019 Levin Moser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */
async function gradClassActivationMap(
    model,
    classLable,
    imageTensorData
) {

    // Locate the last conv layer of the model
    let lastConvLayerIndex = model.layers.length - 1;
    while (lastConvLayerIndex >= 0) {
        if (model.layers[lastConvLayerIndex].getClassName().startsWith('Conv')) {
            break;
        }
        lastConvLayerIndex--;
    }
    const lastConvLayer = model.layers[lastConvLayerIndex];

    // Create "sub-model 1", which goes from the original input to the output of the last convolutional layer
    const lastConvLayerOutput = lastConvLayer.output;
    const subModel1 = tf.model({
        inputs: model.inputs,
        outputs: lastConvLayerOutput
    });

    // Create "sub-model 2", which goes from the output of the last convolutional layer to the output before the softmax layer
    const subModel2Input = tf.input({
        shape: lastConvLayerOutput.shape.slice(1)
    });
    lastConvLayerIndex++;
    let currentOutput = subModel2Input;
    while (lastConvLayerIndex < model.layers.length) {
        if (
            model.layers[lastConvLayerIndex].getConfig()['activation'] === 'softmax'
        ) {
            break;
        }
        currentOutput = model.layers[lastConvLayerIndex].apply(
            currentOutput
        );
        lastConvLayerIndex++;
    }
    const subModel2 = tf.model({
        inputs: subModel2Input,
        outputs: currentOutput
    });

    // Generate the heatMap
    return tf.tidy(() => {
        // run the sub-model 2 and extract the slice of the probability output that corresponds to the desired class
        // @ts-ignore
        const convOutput2ClassOutput = (input) => subModel2.apply(input, { training: true }).gather([classLable], 1);

        // This is the gradient function of the output corresponding to the desired class with respect to its input (i.e., the output of the last convolutional layer of the original model)
        const gradFunction = tf.grad(convOutput2ClassOutput);

        // Calculate the values of the last conv layer's output
        const lastConvLayerOutputValues = subModel1.apply(imageTensorData);

        // Calculate the values of gradients of the class output w.r.t. the output of the last convolutional layer
        const gradValues = gradFunction(lastConvLayerOutputValues);

        // Calculate the weights of the feature maps
        const weights = tf.mean(gradValues, [0, 1, 2]);

        const weightedFeatures = lastConvLayerOutputValues.mul(weights);

        // apply ReLu to the weighted features
        var heatMap = weightedFeatures.relu();

        // normalize the heat map
        heatMap = heatMap.div(heatMap.max());

        // Up-sample the heat map to the size of the input image
        // @ts-ignore
        heatMap = tf.image.resizeBilinear(heatMap, [
            imageTensorData.shape[1],
            imageTensorData.shape[2]
        ]);

        // Apply an RGB colormap on the heatMap to convert to grey scale heatmap into a RGB one
        var gradCAM = applyColorMap(heatMap);

        // To form the final output, overlay the color heat map on the input image
        gradCAM = gradCAM.mul(2).add(imageTensorData.div(255));

        gradCAM = gradCAM.div(gradCAM.max()).mul(255);

        return gradCAM
    });
}
