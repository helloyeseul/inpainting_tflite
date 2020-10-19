import os, argparse
import tensorflow as tf

def convert_to_ftlite(frozen_model, output_dir):
    
    # check output dir 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model = output_dir + "/inpaint_model.tflite"
    
    input_node_names = 'input'
    output_node_names = "inpaint_net/Tanh_1"
    
    input_shapes = {'input': [1, 512, 1360, 3]}
    
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = frozen_model,
        input_shapes = input_shapes,
        input_arrays = input_node_names.split(","),
        output_arrays = output_node_names.split(",")
    )

    converter.allow_custom_ops=True
    converter.experimental_new_converter =True
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    # converter.target_spec.support_types = [tf.float16]
    # converter.target_spec.support_types = [tf.float32]
    
    # Convert to TFLite Model
    print("convert start.")
    tflite_model = converter.convert()
    print("convert done.")
    
    open(output_model, "wb").write(tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model", type=str,
                        help="Frozen model directory to convert")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory")
    
    args = parser.parse_args()

    convert_to_ftlite(args.frozen_model, args.output_dir)
