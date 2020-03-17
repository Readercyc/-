//绘制canvas部分
var model = {

}
var correctNum = 0;
function readFile(e){
    console.log(e.files);
    var file = e.files[0];//获取input输入的图片
    if(!/image\/\w+/.test(file.type)){
        alert("请确保文件为图像类型");
        return false;
    }//判断是否图片，在移动端由于浏览器对调用file类型处理不同，虽然加了accept = 'image/*'，但是还要再次判断
    var reader = new FileReader();
    reader.readAsDataURL(file);//转化成base64数据类型
    reader.onload = function(e){
        drawToCanvas(this.result);
    }
}


function drawToCanvas(imgData){
    var cvs = document.querySelector('#mycanvas');
    console.log(cvs)
    cvs.width=50;
    cvs.height=50;
    var strDataURI;
    var ctx = cvs.getContext('2d');
    var img = new Image;
        img.src = imgData;
        img.onload = function(){//必须onload之后再画
            ctx.drawImage(img,0,0,50,50);
            strDataURI = cvs.toDataURL();//获取canvas base64数据
            console.log(strDataURI);
        }
    
}
function getImageData(){
    var cvs = document.querySelector('#mycanvas');
    let imageData = cvs.getContext('2d').getImageData(0,0,50,50);
    var pixelData = [];
    let color;
    console.log(imageData);
    for(let i = 0; i < imageData.data.length; i+=4){
        color = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2])/3;
        pixelData.push(Math.round((255-color)/255*100)/100);
    }
    return pixelData;
}
// function readCorrectNum(){
//    correctNum = document.getElementById('correctNum').value;
// }
async function btnPredictClickHandler(){
    let data = getImageData();
    // console.log(data);
    // correctNum = document.getElementById('correctNum').value;
    // let targetTensor = tf.oneHot(parseInt(correctNum),20);
    //model.fit(tf.tensor([data]));
    let predictions = await this.model.predict(tf.tensor1d(data).reshape([1,50,50,1]));
    let str = predictions.arraySync()[0].toString().split(",");
    let matching = "";
    console.log(str);
    for(let i=0;i<str.length;i++){
        matching =  matching += `<p class="row-12 text-center">这张图片与第${i}种花纹的匹配度为${Math.round(str[i]*10000/100)}%</p>`
    }
    let result = predictions.argMax(1).arraySync()[0];
    document.querySelector("#result").innerHTML = result;
    document.querySelector("#matching").innerHTML = matching;

}
async function btnTrainClickHandler(){
   let data = getImageData();
   correctNum = document.getElementById('correctNum').value;
   let targetTensor = tf.oneHot(parseInt(correctNum),20);
   
   console.log("starting training");
   await model.fit(tf.tensor1d(data).reshape([1,50,50,1]),tf.tensor([targetTensor.arraySync()]),{
    epochs:30,    
    callbacks:{
           onEpochEnd(epoch,logs){
               if(epoch==0||epoch==29)
                    console.log(epoch,logs);
               document.querySelector('.train').innerHTML = "训练次数:"+ (epoch+1) + " 失误率:"+logs.loss;
           }
       }
   });
   console.log("Completed");

}
window.onload = function(){
    this.model = tf.sequential({
        layers:[
            tf.layers.conv2d({
                    inputShape: [50,50,1],
                    kernelSize: 5,
                    filters: 8,
                    strides: 1,
                    activation: 'relu',
                    kernelInitializer: 'varianceScaling'
                }
            ),
            tf.layers.maxPooling2d({
                    poolSize: [2, 2], 
                    strides: [2, 2]
                }
            ),
            tf.layers.conv2d({
                kernelSize: 5,
                filters: 16,
                strides: 1,
                activation: 'relu',
                kernelInitializer: 'varianceScaling'
            }),
            tf.layers.maxPooling2d({
                poolSize: [2, 2], 
                strides: [2, 2]
            }),
            tf.layers.flatten(),
            tf.layers.dense({
                units: 20,
                kernelInitializer: 'varianceScaling',
                activation: 'softmax'
              })
        ]
    })
    this.model.compile({
        optimizer:'sgd',
        loss:'categoricalCrossentropy',
        metrics:['accuracy']
    })
    console.log(this.data)
}