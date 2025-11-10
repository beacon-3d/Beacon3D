import argparse
import json
import os
import numpy as np
from tqdm import tqdm

KNOWLEDGE_TYPES = ['class', 'appearance', 'geometry', 'spatial', 'functionality']


def extract_pred(data_dict):
    """
    TODO: fill this function to extract grounding inference results
    """
    if 'correct' in data_dict:
        return data_dict['correct']
    elif 'pred' in data_dict and 'ref_captions' in data_dict:
        pred = int(data_dict['pred'][4:7])
        return pred in data_dict['ref_captions']
    else:
        raise NotImplementedError()


def build_results_mapping(data, infer_results):
    infer_results_mapping = {}
    for item1, item2 in tqdm(list(zip(data, infer_results))):
        scene_id = item1['scan_id']
        text = item1['utterance']
        correct = extract_pred(item2)
        if scene_id not in infer_results_mapping:
            infer_results_mapping[scene_id] = {}

        tag = item1['item_id'].split('_')[-1].lower()
        tag = frozenset(tag.split('-'))
        infer_results_mapping[scene_id][text.lower()] = (correct, tag)
    return infer_results_mapping


def process_to_metadata(metadata, infer_results_mapping):
    output = []
    for scene_id, v0 in metadata.items():
        for obj_class, v1 in v0.items():
            for obj_id, v2 in v1.items():
                for chain_id, chain in v2.items():
                    this_dict = {
                        'obj_id': f'{scene_id}-{obj_id}',
                        'text': [],
                        'tag': [],
                        'correct': []
                    }
                    for item in chain:
                        txt = item['text'].lower()
                        tag = item['tag'].lower()
                        tag = frozenset(tag.split('-'))
                        assert tag == infer_results_mapping[scene_id][txt][1], f"{item}, {infer_results_mapping[scene_id][txt]}"
                        this_dict['text'].insert(0, txt)
                        this_dict['tag'].insert(0, '-'.join(tag))
                        this_dict['correct'].insert(0, int(infer_results_mapping[scene_id][txt][0]))
                    output.append(this_dict)
    return output


def eval_stats(infer_results_mapping: dict, processed_grounding: list):
    print("Data statistics:")
    total_case = sum(len(v) for v in infer_results_mapping.values())
    total_chain = len(processed_grounding)
    print(f"case: {total_case}")
    print(f"object: {len(range(0, total_chain, 3))}\n")

    # tag
    tag_count_marginal = {}
    tag_count_compound = {}
    for scene_id, v0 in infer_results_mapping.items():
        for txt, v1 in v0.items():
            tag = v1[1]
            if tag not in tag_count_compound:
                tag_count_compound[tag] = 0
            tag_count_compound[tag] += 1
            for k in KNOWLEDGE_TYPES:
                if k in tag:
                    if k not in tag_count_marginal:
                        tag_count_marginal[k] = 0
                    tag_count_marginal[k] += 1

    print("Marginal:")
    for k, v in tag_count_marginal.items():
        print(f"{k}: {v}")

    print("\nCompound:")
    for k, v in tag_count_compound.items():
        print(f"{'-'.join(k)}: {v}")
    print()   # newline to separate


def eval_score(infer_results_mapping: dict, processed_grounding: list):
    print("Scores:")

    def tag_score(infer_results_mapping: dict, key='overall'):
        correct = 0
        total = 0
        for scene_id, v0 in infer_results_mapping.items():
            for txt, v1 in v0.items():
                if key in v1[1] or key == 'overall':
                    correct += v1[0]
                    total += 1
        print(f"[{key}]: {correct/total*100:.2f}")

    # per case
    print("case:")
    tag_score(infer_results_mapping)
    for k in KNOWLEDGE_TYPES:
        tag_score(infer_results_mapping, key=k)

    # per chain
    chain_metrics = [all(item['correct']) for item in processed_grounding]
    print(f"\nchain: {np.mean(chain_metrics)*100:.2f}")

    # per object
    obj_metrics = []
    for i in range(0, len(processed_grounding), 3):
        this_obj_scores = []
        for item in processed_grounding[i:i+3]:
            this_obj_scores.append(item['correct'][-1])
        obj_metrics.append(all(this_obj_scores))
    print(f"object: {np.mean(obj_metrics)*100:.2f}\n")


def eval_chain(processed_grounding: list):
    print("Chain analysis:")
    # coarse 1 fine 1, coarse 1 fine 0, coarse 0 fine 1, coarse 0 fine 0
    types_all = [0, 0, 0, 0]
    for item in processed_grounding:
        flags = item['correct']
        coarse_correct = all(flags[:-1])
        fine_correct = flags[-1]
        if coarse_correct:
            if fine_correct:
                types_all[0] += 1
            else:
                types_all[1] += 1
        else:
            if fine_correct:
                types_all[2] += 1
            else:
                types_all[3] += 1

    num_overall = sum(types_all)
    print(f"four types: {types_all[0]/num_overall*100:.2f}, {types_all[1]/num_overall*100:.2f}, "
          f"{types_all[2]/num_overall*100:.2f}, {types_all[3]/num_overall*100:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer', type=str, help='Path of raw inference results (json)')
    parser.add_argument('--output', type=str, help='Path of processed scores (json)', default=None)
    parser.add_argument('--domain', type=str, help='scannet | 3rscan | multiscan', default=None)
    parser.add_argument('--data', type=str, help='Path of test data (json)', default='data/scannet/scannet_grounding.json')
    parser.add_argument('--metadata', type=str, help='Path of metadata (json)', default='data/scannet/metadata_scannet_grounding.json')
    args = parser.parse_args()
    with open(args.infer) as f:
        infer_results = json.load(f)

    output_file = args.output
    if output_file is None:
        output_file = os.path.splitext(args.infer)[0] + '_processed.json'

    # build output results
    if args.domain:
        domain = args.domain.lower()
        assert domain in ['scannet', '3rscan', 'multiscan']
        args.data = f'data/{domain}/{domain}_grounding.json'
        args.metadata = f'data/{domain}/metadata_{domain}_grounding.json'
    with open(args.data) as f:
        data = json.load(f)
    with open(args.metadata) as f:
        metadata = json.load(f)

    # construct mapping from text to results
    infer_results_mapping = build_results_mapping(data, infer_results)

    # organize inference results according to metadata
    processed_grounding = process_to_metadata(metadata, infer_results_mapping)
    with open(output_file, 'w') as f:
        json.dump(processed_grounding, f, indent=2)
    print(f"Finish processing inference results, results saved to {output_file}\n")

    # eval_stats(infer_results_mapping, processed_grounding)
    eval_score(infer_results_mapping, processed_grounding)
    eval_chain(processed_grounding)


if __name__ == '__main__':
    main()
