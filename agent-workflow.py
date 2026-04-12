import pandas as pd
import json
import os
from openai import OpenAI
import time

api_key = ''
TARGET_CITY = '澳门' # 如果是其他城市的新闻，则改为其他城市名称；本文研究对象为澳门，所以是澳门
if not api_key:
    raise ValueError('Please set the API_KEY environment variable')

client = OpenAI(
    api_key=api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 新闻中必须包含澳门相关新闻，否则就是归类为其他类别
categories = [
    '政治与政策', 
    '经济与金融', 
    '文化与旅游', 
    '体育赛事', 
    '教育与人才', 
    '医疗与公共健康', 
    '环境与可持续发展', 
    '交通与基础设施', 
    '法律与治安管理', 
    '应急与安全', 
    '科技与创新',
    '就业与培训',
    '民生与便民服务',
    '其他'
]

categories_HD = [
    {'value': '政治与政策', 'description': '任免、廉政反腐、纪律建设、政府工作报告、党建、理论学习、八项规定、公务员培训、国家政策等。'},
    {'value': '经济与金融', 'description': '刺激消费、产业升级、企业服务、经济发展、经济规划、高质量发展等。'},
    {'value': '文化与旅游', 'description': '传统节日、剧团、美食、历史文学作品、历史建筑、非遗、文化活动、旅游产业、演唱会等。'},
    {'value': '体育赛事', 'description': '体育赛事。'},
    {'value': '教育与人才', 'description': '创业、吸引人才、大中小学及幼儿园教育、荣誉表彰、历史人物榜样、主题教育等。'},
    {'value': '医疗与公共健康', 'description': '医疗保险、流感、病毒、医院、中医药、保健、食品安全等。'},
    {'value': '环境与可持续发展', 'description': '动植物保护、植树造林、湿地恢复、生态治理、城市环境治理等。'},
    {'value': '交通与基础设施', 'description': '交通工程、物流运输、货运服务、交通管制、市政工程等。'},
    {'value': '法律与治安管理', 'description': '市场监管、网络安全、反诈宣传、社会治安、治安警局宣传等。'},
    {'value': '应急与安全', 'description': '含天气预警、极端天气、防火提醒、消防安全、海事应急等。'},
    {'value': '科技与创新', 'description': '科技创新等。'},
    {'value': '就业与培训', 'description': '工作机会、就业培训、职业教育、求职等。'},
    {'value': '民生与便民服务', 'description': '社会工作与便民服务、民生福利、自助服务、电子政务等。'},
    {'value': '其他', 'description': '不属于以上任何类别的内容。'},
]

# 敏感词替换列表
# External filtering tool
# 自定义敏感词库
# sen_names = pd.read_csv('./stopwords/llm_sensitive_name.csv')['name'].tolist()  
sen_names = []
def construct_summ_prompt(content):
    for i in sen_names:
        content = content.replace(i, '某领导人')   
    role = '你是文本摘要助手，擅长提炼新闻文章中和' + TARGET_CITY + '相关的信息，并根据该内容写一段不超过500字的总结。'    
    prompt = f'''
    请根据新闻文章中{TARGET_CITY}相关的内容，写一段不超过500字的总结。
    **任务要求**：
    1. 不要深度揣测，内容只需要符合客观实际即可。
    2. 只能输出结果，不允许提供任何解释和推理过程的回答。
    **新闻文章内容如下：**：
    {content}

    请根据以上内容，完成任务。
    '''
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": prompt.strip()}
    ]
    
    return messages


def construct_cls_prompt(summ):
    role = '你是一名专业的新闻标注员，擅长新闻分类任务。'
    prompt = '请将新闻分配到以下14个类别中，每条新闻可以属于一个或多个类别。\n'
    prompt += '**新闻类别**：\n'
    for category in categories_HD:
        prompt += f'{category["value"]}：{category["description"]}\n'

    prompt += '**任务要求**：\n'
    prompt += '1. 根据新闻的内容，确定最相关的类别。\n'
    prompt += '2. 每条新闻至少分配 1 个类别\n'
    prompt += '3. 不可以修改任何分类的名称，必须和提供的分类名称保持100%一致，也不可以增加分类，只能从指定的新闻类别中挑选最合适的类别。\n'
    prompt += '4. 一条新闻，可以是多个类别\n'
    prompt += '5. 输出结果，最后调用get_formatted_label工具来格式化输出结果。\n'

    prompt += '以下是新闻摘要：\n'
    prompt += f'{summ}\n'
    prompt += '请根据以上内容输出对应的新闻分类标签'

    messages = [
        {'role': 'system', 'content': role},
        {'role': 'user', 'content': prompt.strip()}
    ]
    
    return messages

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_formatted_label",
            "description": "格式化新闻分类标签。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cls_text": {
                        "type": "string"
                    }
                },
                "required": ["cls_text"],
            },
        },
    },
]

def call_api(messages, tools=None):
    if tools is None:
        tools = []
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        temperature=0.1,
        tools=tools,
        tool_choice="required" if tools else None
    )
    message = response.choices[0].message

    # print(f"API response message: {message}")
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        if tool_call.function.name == 'get_formatted_label':
            arguments = json.loads(tool_call.function.arguments)
            # print(f"Raw classification text: {arguments['cls_text']}")
            cls_text = arguments['cls_text']
            return format_label(cls_text)
    else:
        return message.content

def format_label(cls_text):
    text = cls_text.strip().replace('。', '')
    result = []
    for c in categories:
        if c in text:
            result.append(c)
    return "@".join(result)

def main(input_csv, output_csv='results.csv'):
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=['idx', 'label']).to_csv(output_csv, index=False)
    
    df = pd.read_csv(input_csv)
    for index, row in df.iterrows():
        id_ = row['idx']
        text = row['text']
        
        # 如果执行中断，可以从起始id继续执行 if int(id_) >= 0:
        # 摘要
        summ_messages = construct_summ_prompt(text)
        summ = call_api(summ_messages)
        
        # 分类
        cls_messages = construct_cls_prompt(summ)
        label = call_api(cls_messages, tools=tools)
        
        # 保存数据
        temp_df = pd.DataFrame([[id_, label]], columns=['idx', 'label'])
        temp_df.to_csv(output_csv, mode='a', header=False, index=False)
        
        print(f'Processed id: {id_}')
        # time.sleep(2)  

if __name__ == '__main__':
    # 包含两列 idx,text
    main('input.csv')