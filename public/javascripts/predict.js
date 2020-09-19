var body = document.getElementById('main');
var para = document.createElement('p');
body.appendChild(para);
tens = tf.tensor([1, 2, 3, 4])
tens.data().then(data => para.appendChild(document.createTextNode(data)));