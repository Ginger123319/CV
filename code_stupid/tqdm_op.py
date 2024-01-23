from tqdm import tqdm
import time
epoch_size = 10
epoch = 0
Epoch = 10
with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict) as pbar:
    for i in range(10):
        # print(i)
        pbar.set_postfix(**{'i': i})
        time.sleep(0.5)
        pbar.update(1)
    pbar.close()
