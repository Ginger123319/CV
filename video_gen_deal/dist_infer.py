import multiprocessing
import os
import tensorflow as tf

def run_inference_on_gpu(gpu_id):
    # 将CUDA_VISIBLE_DEVICES环境变量设置为特定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # 导入TensorFlow（确保在这个进程中只导入一次）
    import tensorflow as tf
    
    # 指定使用特定GPU
    with tf.device(f'/GPU:{gpu_id}'):
        # 构建你的模型
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(100,))
        ])
        
        # 准备一些数据进行推理
        data = tf.random.normal((64, 100))
        
        # 使用模型进行推理
        output = model(data)
        
        # 打印或处理推理结果
        print(f"Inference results on GPU {gpu_id}: {output.numpy()}")

if __name__ == "__main__":
    # 创建两个进程，每个进程使用一个GPU
    p1 = multiprocessing.Process(target=run_inference_on_gpu, args=(0,))
    p2 = multiprocessing.Process(target=run_inference_on_gpu, args=(1,))
    
    # 启动进程
    p1.start()
    p2.start()
    
    # 等待进程结束
    p1.join()
    p2.join()