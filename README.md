# inpainting_tflite

Converting Tensorflow model to Tensorflow Lite model.
* This project is based on https://github.com/JiahuiYu/generative_inpainting

<br>
<br>

## 1. 모델 테스트

#### [ 기존 코드 ]

[test.py](https://github.com/dptmf7705/inpainting_tflite/blob/master/test.py)

```bash
...
    
# inpaint 모델
model = InpaintCAModel()

...

# [이미지]+[마스크] 붙여서 input_image 만들기
input_image = np.concatenate([image, mask], axis=2)

# TODO. 여기 선언된 tf.constant 를 tf.placeholder 로 바꿔야됨
input_image = tf.constant(input_image, dtype=tf.float32)
output = model.build_server_graph(FLAGS, input_image)
output = (output + 1.) * 127.5
output = tf.reverse(output, [-1])
output = tf.saturate_cast(output, tf.uint8)

...

# 모델 실행
result = sess.run(output)

# output 사진 저장
cv2.imwrite(args.output, result[0][:, :, ::-1])
        
```

#### [ 새로 작성한 코드 ]

[test_single_image.py](https://github.com/dptmf7705/inpainting_tflite/blob/master/test_single_image.py)

```bash

def get_input(image_path, mask_path, image_width, image_height):
    ...
    
    # 이미지, 마스크 shape 변경 : [512, 680, 3] to [1, 512, 680, 3]
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)

    # [ 이미지 + 마스크 ] 붙여서 input_image 만들기
    # input_image shape 은 [1, 512, 1360, 3] 이 됨.
    input_image = np.concatenate([image, mask], axis=2)


def test_single_image(input_image, output_path, image_height, image_width, ckpt_dir):
    ...
    
    # inpaint 모델
    model = InpaintCAModel()

    # input_image 를 tf.placeholder 로 선언
    # name='input' 은 나중에 frozen graph 만들고 tflite 변환할 때 필요함
    input_image_ph = tf.placeholder(tf.float32, name='input', shape=(1, image_height, image_width*2, 3)) 

    output = model.build_server_graph(input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)

    ...

    # 모델 실행시킬 때 feed_dict 로 input_image 넘겨줌
    result = sess.run(output, feed_dict={input_image_ph: input_image})

    # output 사진 저장
    cv2.imwrite(output_path, result[0][:, :, ::-1])

    # 체크포인트 저장 - frozen graph 만들 때 필요함
    saver = tf.train.Saver().save(sess, './model_logs/test/model.ckpt')

    # tensorboard 출력을 위한 그래프 저장 - output node 이름 찾는데 필요함
    tf.summary.FileWriter('./tbgraph', sess.graph)
```

#### test_single_image.py 실행 (tensorflow 1.7.0):

* Directory Tree 

```
- test_data
  -- input
     --- case1.png
     --- case2.png
           ...
  -- mask
     --- case1.png
     --- case2.png
           ...
- model_logs
  -- places2
     --- checkpoint
     --- <snap_name>.data
     --- <snap_name>.index
     --- <snap_name>.meta
- test_single_image.py
```

* Run  `python test_single_image.py --data_dir=test_data --ckpt_dir=model_logs/places2 --image_height=512 --image_width=680`

```bash
python test_single_image.py --data_dir=test_data --ckpt_dir=model_logs/places2 --image_height=512 --image_width=680
```

<br>
<br>

## 2. frozen graph 만들기

[freeze.py](https://github.com/dptmf7705/inpainting_tflite/blob/master/freeze.py)

```bash
...

# input node 이름 : test_single_image.py 에서 placeholder 에 선언했던 name
# output node 이름 : tensorboard 실행해서 직접 찾아야됨.
def freeze_graph(ckpt_dir, output_file, input_nodes, output_nodes):

    # test_single_image.py 에서 저장한 체크포인트 불러오기
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt.model_checkpoint_path

    saver = tf.train.import_meta_graph(ckpt_path + ".meta", clear_devices=True)

    # 그래프 불러오기
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    ...
    
    # 그래프 얼리기 : variables 를 모두 constants 로 바꿔서 frozen graph 만들기
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, output_nodes.split(",")
    )

    # 얼린 그래프를 파일에 저장
    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
```

#### freeze.py 실행 (tensorflow 1.7.0):

Run `python freeze.py --model_dir=model_logs/test/model --output_dir=tflite`

<br>
<br>

## 3. frozen graph (.pb) 를 .tflite 로 변환 

[convert_tflite_v2.py](https://github.com/dptmf7705/inpainting_tflite/blob/master/convert_tflite_v2.py)

```bash
...

# input node, output node 이름은 frozen graph 생성할 때랑 동일하게
input_node_names = 'input'
output_node_names = "saturate_cast"

# input node 의 shape
input_shapes = {'input': [1, 512, 1360, 3]}

# frozen graph 를 tflite 로 변환해주는 converter 선언
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = frozen_model,
    input_shapes = input_shapes,
    input_arrays = input_node_names.split(","),
    output_arrays = output_node_names.split(",")
)

...

# converter 실행 
tflite_model = converter.convert()

# .tflite 파일로 저장
open(output_model, "wb").write(tflite_model)
```

<br>

convert_tflite_v2.py 실행 (tensorflow=2.3.0 에서 실행함):

```bash
python convert_tflite_v2.py --frozen_model=tflite/frozen_model.pb --output_dir=tflite
```


<br>
<br>

## Tensorboard 에서 output node 이름 찾기

Tensorboard 실행 (tensorflow v1, v2 상관없음):
```bash
tensorboard --logdir=tbgraph
```

실행 결과: 
```bash
TensorBoard 2.3.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

브라우저에서 위 주소로 접속 -> 그래프 화살표 따라가면서 맨 마지막 노드 이름 찾기 -> "saturate_cast"

<br>

![image](https://user-images.githubusercontent.com/22764812/96363150-60ea3000-116d-11eb-860d-6fe91f1d7794.png)

