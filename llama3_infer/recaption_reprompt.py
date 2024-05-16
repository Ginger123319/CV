from itertools import islice
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import sqlite3
from tqdm import tqdm

def get_prompt(frame1, frame2, frame3):
    prompt = f"""
    You are watching a video composed of the following three key frames:

    1. Description of the opening frame:
    ```
    {frame1}
    ```

    2. Description of the middle frame:
    ```
    {frame2}
    ```

    3. Description of the ending frame:
    ```
    {frame3}
    ```

    Based on the descriptions of these three key frames, please generate a complete and coherent description of the entire video.

    Not concerned about the chronological order of events in the scene.
    Do not generate content about keywords 'frame' or 'scene'.
    Pay attention to all objects in the video.
    The description should be useful for AI to re-generate the video.
    The description should be less than six sentences.
    It's good to point out the location of each object.
    If something is in motion, it is better to describe the motion in detail.

    Here are some examples of good descriptions:
    1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 
    2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 
    3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.

    Do not generate the combiled description about the chronological order of events in the scene.
    Do not return sentence like 'Here is a complete and coherent description of the entire video:',etc.
    Just return the generated description without any additional information.

    """
    return prompt

# frame1 = "In the image, there are four men walking down a city street. They are dressed in business attire, with each man wearing a suit and tie. The men are wearing sunglasses and appear to be in motion, suggesting they are walking with purpose. The street they are on is lined with buildings, and there is a stop sign visible in the background. The overall style of the image is realistic, capturing a moment in time on a city street. The men are the main focus of the image, with their actions and attire drawing the viewer's attention. The cityscape in the background provides context for the setting."
# frame2 = "In the frame, there are five young men walking down a city street. They are dressed in business attire, with each wearing a suit and tie, and they are all wearing sunglasses. The man in the center is wearing a white shirt, while the others are wearing dark-colored shirts. They are walking in a line, with the man in the center slightly ahead of the others. The street they are walking on is lined with buildings, and there is a stop sign visible in the background. The overall style of the image is realistic, with a focus on the characters and their actions."
# frame3 = "In the image, there are four young men walking down a city street. They are dressed in business attire, with each wearing a suit and tie. The man in the center is wearing a white shirt and jeans, which stands out from the others who are in dark suits. They are all wearing sunglasses and have dark hair. The street they are walking on is lined with parked cars and there is a stop sign visible in the background. The overall style of the image is casual yet professional, capturing a moment of camaraderie among the group."

# prompt = get_prompt(frame1, frame2, frame3)
# prompts = [prompt]*2
# print(len(prompts))
# exit()

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]

def get_messages_for_chat():
    system_message = {
            "role": "system",
            "content": """You are part of a team of bots that combines three captions into one. Consider three captions from the head frame, middle frame, and tail frame of a video clip.
            Generate a comprehensive description with a focus on the middle frame, supplemented by start and end frame.

    There are a few rules to follow:

    - You will only ever output a single caption per user request.
    - Utilize the information present in both the captions as much as possible.
    - Captions returned must be between 15-80 words. Extra words will be ignored.
    - Pay attention to all objects in the video.
    - The description should be useful for AI to re-generate the video.
    - If something is in motion, it is better to describe the motion in detail.
    - Do not generate any comment content about the video, frame or image, started with 'The style of the ' and others like this. 
    """,
    }

    rest_of_the_message = [
        {
            "role": "user",
            "content": """Combine the following three captions into one: 
            "In the image, there are four men walking down a city street. They are dressed in business attire, with each man wearing a suit and tie. The men are wearing sunglasses and appear to be in motion, suggesting they are walking with purpose. The street they are on is lined with buildings, and there is a stop sign visible in the background. The overall style of the image is realistic, capturing a moment in time on a city street. The men are the main focus of the image, with their actions and attire drawing the viewer's attention. The cityscape in the background provides context for the setting.", 
            "In the frame, there are five young men walking down a city street. They are dressed in business attire, with each wearing a suit and tie, and they are all wearing sunglasses. The man in the center is wearing a white shirt, while the others are wearing dark-colored shirts. They are walking in a line, with the man in the center slightly ahead of the others. The street they are walking on is lined with buildings, and there is a stop sign visible in the background. The overall style of the image is realistic, with a focus on the characters and their actions.",
            "In the image, there are four young men walking down a city street. They are dressed in business attire, with each wearing a suit and tie. The man in the center is wearing a white shirt and jeans, which stands out from the others who are in dark suits. They are all wearing sunglasses and have dark hair. The street they are walking on is lined with parked cars and there is a stop sign visible in the background. The overall style of the image is casual yet professional, capturing a moment of camaraderie among the group."
            """,
        },
        {
            "role": "assistant",
            "content": "Five young men were walking down a city street. They were dressed in business attire, each in a suit and tie, all wearing sunglasses. The man in the middle was wearing a white shirt and jeans, standing out from the other men in dark suits. They walked in a line, the man in the center slightly ahead of the others. The street they are walking on is lined with buildings and a stop sign can be seen in the background.",
        },
        {
            "role": "user",
            "content": """Combine the following three captions into one: 
            "In the frame, there is a woman standing in front of a large screen displaying a group of men. The woman is wearing a floral dress and appears to be speaking or presenting. The men on the screen are dressed in suits and sunglasses, and they are standing in a row, looking directly at the camera. The setting suggests a news or television studio, with the woman positioned in the foreground and the screen in the background. The overall style of the image is professional and polished, typical of a news broadcast or a television show.", 
            "In the frame, a woman is standing in front of a large screen displaying a group of men. She is positioned to the right of the screen, wearing a pink floral dress and appears to be speaking or presenting. The screen shows four men in suits, walking down a street with a stop sign visible in the background. The style of the image suggests it is a screenshot from a news program or a television show, with the woman possibly being a news anchor or a host. The setting is a studio with a blue panel behind the woman, and the overall atmosphere is professional and polished.",
            "In the image, a news anchor is standing in front of a large screen displaying a video of three men walking down a street. The anchor is dressed in a floral dress and is positioned on the right side of the frame. The large screen, which occupies the majority of the background, is showing the three men in the center of the image. The men are dressed in suits and sunglasses, and they are walking in a line, with the first man slightly ahead of the other two. The setting appears to be an urban environment, possibly a city street, with buildings and a stop sign visible in the background of the video. The style of the image suggests a news broadcast or a television show, with the anchor providing commentary or introducing the video segment."
            """,
        },
        {
            "role": "assistant",
            "content": "A woman stands in front of a large screen showing a group of men. She is standing on the right side of the screen, wearing a pink floral dress, and appears to be speaking or giving a presentation. The screen shows several men in suits walking down a street and a stop sign can be seen in the background. The background of the video is a studio with a blue panel behind the woman.",
        },
        {
            "role": "user",
            "content": """Combine the following three captions into one : {frame1},\n{frame2},\n{frame3}""",
        },
    ]
    return system_message, rest_of_the_message

def get_messages(frame1, frame2, frame3):
    from pprint import pprint

    system_message, rest_of_the_message = get_messages_for_chat()
    updated_prompt = rest_of_the_message[-1]["content"].format(frame1=frame1, frame2=frame2, frame3=frame3)
    rest_of_the_message[-1]["content"] = updated_prompt

    final_message = [system_message]
    final_message.extend(rest_of_the_message)

    # pprint(final_message)
    # exit()
    return final_message

def str2dict(batch):
    captions = []
    cut_ids = []
    for cut_id, caption in batch:
        try:
            json_object = json.loads(caption)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"An exception occurred while dealing with cut_id {cut_id}: {e}")
            continue
        captions.append(json_object)
        cut_ids.append(cut_id)
    return cut_ids, captions

# 迭代器，用于分批处理数据
def batch_iterator(iterable, batch_size):
    
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:  # 如果批次为空，表示迭代完成
            break
        yield str2dict(batch)

def get_messages_batch(batch):

    messages_batch = []
    cut_ids, captions = batch
    for caption in captions:
        # prompt = get_prompt(caption['frame1'], caption['frame2'], caption['frame3'])
        messages = get_messages(caption['frame1'], caption['frame2'], caption['frame3'])
        messages_batch.append(messages)
    return messages_batch, cut_ids

def get_response(messages, tokenizer, model, device):
    if not messages:
        return None
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    # 设置生成参数
    # temperature = 0.2  # 选择一个较低的温度值以减少随机性
    # top_p = 1  # 可以选择设置top_p为1，这样模型每次只选择概率最高的词
    # 从tokenizer获取pad_token_id
    pad_token_id = tokenizer.pad_token_id

    # 生成文本
    attention_mask = model_inputs['attention_mask'].to(device)  # 注意这里使用

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,  # 传递attention_mask
        pad_token_id=pad_token_id,  
        do_sample=True, 
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response



if __name__ == '__main__':

    device = "cuda" # the device to load the model onto
    
    model = AutoModelForCausalLM.from_pretrained(
        "LLM-Research/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct")
    tokenizer.padding_side = "left"  # 为了让tokenizer在左侧进行padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    # pad_token_id = tokenizer.pad_token
    # 从tokenizer获取pad_token_id
    # pad_token_id = tokenizer.pad_token_id
    # print(pad_token_id)
    # exit()
    # 连接到数据库
    conn = sqlite3.connect('video_base.db')
    cursor = conn.cursor()

    # # 查看数据库中的所有表
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # tables = cursor.fetchall()
    # print(tables)
    # for table_name in tables:
    #     table_name = table_name[0]
    #     # 查看表的结构
    #     cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    #     table_structure = cursor.fetchone()[0]
    #     print(f"Structure of table '{table_name}':")
    #     print(table_structure)

    # # 为特定表添加名为recaption的新列，类型为TEXT
    # alter_table_sql = "ALTER TABLE captions ADD COLUMN recaption TEXT;"

    # try:
    #     cursor.execute(alter_table_sql)
    #     conn.commit()  # 提交事务
    #     print("列 'recaption' 已成功添加到表中。")
    # except sqlite3.Error as e:
    #     print("添加列时出错:", e)
    # finally:
    #     # 关闭游标和连接
    #     cursor.close()
    #     conn.close()
    # exit()

    # 以16为一个批次遍历数据库中的所有数据

    # 设置每组记录的数量
    records_per_group = 8

    # 使用主键的范围查询来获取数据
    select_sql = "SELECT cut_id, caption FROM captions ;"
    cursor.execute(select_sql)
    # results = cursor.fetchall()
    # print(f"共有{len(results)}条数据")
    # exit()
    records = (record for record in cursor.fetchall())
    # for record in records:
    #     print(record)
    # exit()
    
    pbar = tqdm(desc='Processing', total=1630335//records_per_group + 1)

    for batch in batch_iterator(records, records_per_group):

        messages_batch, cut_ids = get_messages_batch(batch)
        responses_batch = get_response(messages_batch, tokenizer, model, device)
        print(responses_batch, '\n')

        if responses_batch is None:
            print(f"No response generated in {cut_ids}, skip saving to database.")
            continue
 
        # 保存生成的回复到数据库
        for cut_id, response in zip(cut_ids, responses_batch):
            update_sql = f"UPDATE captions SET recaption=? WHERE cut_id = ?;"
            try:
                cursor.execute(update_sql, [response, cut_id])
                conn.commit()  # 提交事务
                print("{}:标注数据已保存到数据库中".format(cut_id))
            except sqlite3.Error as e:
                print("{}:更新数据库出错:{}, 回滚事务".format(cut_id, e))
                conn.rollback()  # 回滚事务
                continue
        pbar.update(1)

    pbar.close()

    # 关闭游标和连接
    cursor.close()
    conn.close()