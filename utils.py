import os
import re
import traceback
import openai


def extract_number(text):
    # Using regular expression to find number in text
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def is_binary_question(gts):
    for gt in gts:
        if 'yes' in gt.lower() or 'no' in gt.lower():
            return 1
    return 0


def call_openai_api_azure(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-11-20',
    region: str = 'northcentralus',
):
    API_BASE = ""   # TODO
    ENDPOINT = f"{API_BASE}/{region}"
    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found")

    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version='2024-02-01',
            azure_endpoint=ENDPOINT,
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e


def call_openai_api(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-11-20',
):
    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found")

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e


def clean_answer(data):
    # from SQA3D
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def answer_match(pred, gts):
    # return EM and refined EM
    for gt in gts:
        if pred == gt:
            return 1, 1
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return 0, 1
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


def extract_question_vicuna(text):
    # Using regular expression to find text between "USER: " and " ASSISTANT:"
    match = re.search(r"USER: (.*?) ASSISTANT:", text, re.DOTALL)
    return match.group(1) if match else None


def extract_question_qwen(text):
    match = re.search(r'<\|vision_end\|>(.*?)<\|im_end\|>', text, re.DOTALL)
    return match.group(1) if match else None
