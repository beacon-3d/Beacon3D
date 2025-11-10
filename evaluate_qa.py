import argparse
import json
import logging
import os

import numpy as np
from tqdm import tqdm

from utils import answer_match, call_openai_api_azure, clean_answer, \
                  extract_number, is_binary_question \

KNOWLEDGE_TYPES = ['class', 'appearance', 'geometry', 'spatial', 'existence', 'functionality']


class LLMEvaluator():
    def __init__(self, model, region, prompt_path, verbose=False):
        self.model = model
        self.region = region
        with open(prompt_path) as f:
            self.messages = json.load(f)
        self.verbose = verbose

    def score(self, question, answer, gt):
        messages = self.messages.copy()
        user_prompt = '\n'.join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({'role': 'user', 'content': user_prompt})
        response = call_openai_api_azure(messages=messages, model=self.model, region=self.region)
        score = extract_number(response)
        if self.verbose:
            print(user_prompt, score)
        return score


def extract_pred(data_dict):
    """
    TODO: fill this function to extract answer prediction (str) from model inference results
    """
    if 'response_pred' in data_dict:
        return data_dict['response_pred']
    elif 'answer_pred' in data_dict:
        return data_dict['answer_pred']
    elif 'text' in data_dict:
        return data_dict['text']
    elif 'answer_top10' in data_dict:
        return data_dict['answer_top10'][0]
    elif 'pred_response' in data_dict:
        return data_dict['pred_response']
    elif 'pred' in data_dict:
        return data_dict['pred']
    elif 'prediction' in data_dict:
        return data_dict['prediction']
    else:
        raise NotImplementedError()


def build_results_mapping(data, infer_results, evaluator):
    infer_results_mapping = {}
    for item1, item2 in tqdm(list(zip(data, infer_results))):
        scene_id = item1['scene_id']
        question = item1['question']
        gt = item1['answers']
        pred = extract_pred(item2)

        if not isinstance(pred, str):
            pred = pred[0]
        pred = clean_answer(pred)
        gt = [clean_answer(_gt) for _gt in gt]
        em, em_r = answer_match(pred, gt)
        if em:
            score = 5
        elif is_binary_question(gt) and em_r:
            score = 5
        else:
            score = evaluator.score(question, pred, gt)

        if scene_id not in infer_results_mapping:
            infer_results_mapping[scene_id] = {}

        tag, extra = item1['question_id'].split('_')[-2:]
        infer_results_mapping[scene_id][question.lower()] = (pred, em, em_r, score, tag, int(extra))
    return infer_results_mapping


def process_to_metadata(metadata, infer_results_mapping):
    output = []
    for scene_id, v0 in metadata.items():
        for obj_class, v1 in v0.items():
            for obj_id, v2 in v1.items():
                for item in v2['qa']:
                    q = item['question'].lower()
                    tag = item['tag']
                    extra = int(item['extra_knowledge'])
                    assert tag == infer_results_mapping[scene_id][q][4] and extra == infer_results_mapping[scene_id][q][5]
                    this_dict = {
                        'obj_id': f'{scene_id}-{obj_id}',
                        'question': q,
                        'tag': tag,
                        'extra_knowledge': extra,
                        'answer_gt': item['answer'],
                        'answer_pred': infer_results_mapping[scene_id][q][0],
                        'em': infer_results_mapping[scene_id][q][1],
                        'em_r': infer_results_mapping[scene_id][q][2],
                        'score': infer_results_mapping[scene_id][q][3],
                    }
                    output.append(this_dict)
    return output


def eval_stats(processed_qa: list):
    print("Data statistics:")
    total = len(processed_qa)
    print(f"case: {total}")
    print(f"object: {len(range(0, total, 3))}\n")

    # tag
    tag_count_wo_extra = {}
    tag_count_w_extra = {}
    count_wo_extra = 0
    count_w_extra = 0
    for item in processed_qa:
        tag = item['tag'].lower()
        extra = item['extra_knowledge']
        if extra:
            if tag not in tag_count_w_extra:
                tag_count_w_extra[tag] = 0
            tag_count_w_extra[tag] += 1
            count_w_extra += 1
        else:
            if tag not in tag_count_wo_extra:
                tag_count_wo_extra[tag] = 0
            tag_count_wo_extra[tag] += 1
            count_wo_extra += 1

    for k in KNOWLEDGE_TYPES:
        v1 = 0
        if k in tag_count_wo_extra:
            v1 = tag_count_wo_extra[k]
            print(f"{k} [w/o extra]: {v1}")
        v2 = 0
        if k in tag_count_w_extra:
            v2 = tag_count_w_extra[k]
            print(f"{k} [w/ extra]: {v2}")
        print(f"{k} [overall]: {v1 + v2}\n")

    print(f"[w/o extra]: {count_wo_extra}")
    print(f"[w/ extra]: {count_w_extra}\n")


def eval_score(processed_qa: list):
    print("Scores:")
    # per case
    total = len(processed_qa)
    case_scores = [item['score'] for item in processed_qa]
    case_metrics = (np.mean(case_scores)- 1) / 4 * 100
    print(f"case: {case_metrics:.2f}")

    # per object
    obj_scores = [case_scores[i:i+3] for i in range(0, total, 3)]
    obj_metrics = []
    for obj_score in obj_scores:
        if all([score >= 4 for score in obj_score]):
            obj_metrics.append(1)
        else:
            obj_metrics.append(0)
    obj_metrics = np.mean(obj_metrics) * 100
    print(f"object: {obj_metrics:.2f}\n")

    # tag
    tag_metrics = {}
    for item in processed_qa:
        tag = item['tag'].lower()
        score = item['score']
        if tag not in tag_metrics:
            tag_metrics[tag] = []
        tag_metrics[tag].append(score)

    for k in KNOWLEDGE_TYPES:
        if k in tag_metrics:
            v = tag_metrics[k]
            print(f"{k}: {(np.mean(v) - 1) / 4 * 100:.2f}")


def eval_chain(processed_grounding, processed_qa):
    print("\nChain analysis:")
    obj_metrics_grounding = []
    for i in range(0, len(processed_grounding), 3):
        item1, item2, item3 = processed_grounding[i:i+3]
        if item1['correct'][-1] and item2['correct'][-1] and item3['correct'][-1]:
            obj_metrics_grounding.append(1)
        else:
            obj_metrics_grounding.append(0)

    # grounding 1 qa 1, grounding 1 qa 0, grounding 0 qa 1, grounding 0 qa 0
    types_wo_extra = [0, 0, 0, 0]
    types_all = [0, 0, 0, 0]
    for i in range(len(processed_qa)):
        item = processed_qa[i]
        extra = item['extra_knowledge']
        score = item['score']
        grounding_flag = obj_metrics_grounding[i//3]
        if grounding_flag:
            if score >= 4:
                types_all[0] += 1
                if not extra:
                    types_wo_extra[0] += 1
            else:
                types_all[1] += 1
                if not extra:
                    types_wo_extra[1] += 1
        else:
            if score >= 4:
                types_all[2] += 1
                if not extra:
                    types_wo_extra[2] += 1
            else:
                types_all[3] += 1
                if not extra:
                    types_wo_extra[3] += 1

    num_wo_extra = sum(types_wo_extra)
    num_overall = sum(types_all)
    print(f"[w/o extra] four types: {types_wo_extra[0]/num_wo_extra*100:.2f}, {types_wo_extra[1]/num_wo_extra*100:.2f}, "
          f"{types_wo_extra[2]/num_wo_extra*100:.2f}, {types_wo_extra[3]/num_wo_extra*100:.2f}")
    print(f"[overall] four types: {types_all[0]/num_overall*100:.2f}, {types_all[1]/num_overall*100:.2f}, "
          f"{types_all[2]/num_overall*100:.2f}, {types_all[3]/num_overall*100:.2f}")
    print(f"R1 (higher better): {types_wo_extra[1]/(types_wo_extra[1]+types_wo_extra[3])*100:.2f}")
    print(f"R2 (lower better): {types_all[2]/(types_all[0]+types_all[2])*100:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', type=str, help='Path of raw inference results (json)')
    parser.add_argument('--output', type=str, help='Path of processed scores (json)', default=None)
    parser.add_argument('--grounding', type=str, help='Path of processed grounding results (json)', default=None)
    parser.add_argument('--domain', type=str, help='scannet | 3rscan | multiscan', default=None)
    parser.add_argument('--data', type=str, help='Path of test data (json)', default='data/scannet/scannet_qa.json')
    parser.add_argument('--metadata', type=str, help='Path of metadata (json)', default='data/scannet/metadata_scannet_qa.json')
    parser.add_argument('--model', type=str, help='OpenAI GPT model', default='gpt-4o-2024-08-06')
    parser.add_argument('--region', type=str, help='API endpoint region', default='northcentralus')
    parser.add_argument('--prompt', type=str, help='Path of system prompt', default='data/system_prompt.json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    with open(args.infer) as f:
        infer_results = json.load(f)

    output_file = args.output
    if output_file is None:
        output_file = os.path.splitext(args.infer)[0] + '_processed.json'

    if os.path.exists(output_file):
        print(f"{output_file} already exists")
        with open(output_file) as f:
            processed_qa = json.load(f)
    else:
        # build output results
        if args.domain:
            domain = args.domain.lower()
            assert domain in ['scannet', '3rscan', 'multiscan']
            args.data = f'data/{domain}/{domain}_qa.json'
            args.metadata = f'data/{domain}/metadata_{domain}_qa.json'
        with open(args.data) as f:
            data = json.load(f)
        with open(args.metadata) as f:
            metadata = json.load(f)
        infer_scene_key = 'scene_name' if 'scene_name' in infer_results[0] else 'scene_id'
        if data[0]['scene_id'] != infer_results[0][infer_scene_key]:
            print("Sort scene")
            data = sorted(data, key=lambda x: x[infer_scene_key])

        evaluator = LLMEvaluator(model=args.model, region=args.region, prompt_path=args.prompt, verbose=args.verbose)
        logging.getLogger('httpx').setLevel(logging.WARNING)

        # construct mapping from scene and question to prediction
        infer_results_mapping = build_results_mapping(data, infer_results, evaluator)

        # organize inference results according to metadata
        processed_qa = process_to_metadata(metadata, infer_results_mapping)
        with open(output_file, 'w') as f:
            json.dump(processed_qa, f, indent=2)
        print(f"Scores saved to {output_file}")

    overall_em = np.mean([item['em'] for item in processed_qa]) * 100
    overall_em_r = np.mean([item['em_r'] for item in processed_qa]) * 100
    overall_gpt_score = np.mean([item['score'] for item in processed_qa])
    overall_gpt_score = (overall_gpt_score - 1) / 4 * 100
    print(f"Finish processing inference results: EM = {overall_em:.2f} | EM-R = {overall_em_r:.2f} | GPT-Score = {overall_gpt_score:.2f}\n")

    # eval_stats(processed_qa)
    eval_score(processed_qa)

    if args.grounding and os.path.exists(args.grounding):
        with open(args.grounding) as f:
            processed_grounding = json.load(f)
        eval_chain(processed_grounding, processed_qa)


if __name__ == '__main__':
    main()
