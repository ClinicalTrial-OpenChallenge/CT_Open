from sklearn.metrics import f1_score
import numpy as np
import ast
from tqdm import tqdm
import pickle

def evaluate_multiple_runs(list_of_eval_rets, benchmark_data, ignore_nctids={}):
    IMPLICIT_ARMS = {'no_arm', 'no_distinguishable_arm'}
    EXPLICIT_ARMS = {'distinguishable_arm', 'single_arm'}
    
    categories = ['2-way_implicit', '2-way_explicit', '3-way_explicit']
    
    # Store metrics for each run so we can average them at the end
    # Structure: all_runs_metrics[run_idx][cat] = {'overall': {...}, 'classes': {0: {...}, 1: {...}}}
    all_runs_metrics = []
    
    print(f"Processing {len(list_of_eval_rets)} runs...")
    
    for run_idx, eval_ret in enumerate(list_of_eval_rets):
        # 1. Parse data for the CURRENT run
        run_data = {cat: {'y_true': [], 'y_pred': [], 'y_prob_true': []} for cat in categories}
        for k, output in tqdm(eval_ret.items(), desc=f"Run {run_idx + 1}/{len(list_of_eval_rets)}", leave=False):
            
            nctid, outcome_measure, question_list_idx = k
            output = output.replace("*", "").replace("#", "").replace("```python", "").replace("```", "")

            if nctid in ignore_nctids or output is None:
                raise Exception('nctid in ignore_nctids or output is None')
                
            curr_data = benchmark_data[(nctid, outcome_measure)]
            curr_question = curr_data['question_list_of_lists'][question_list_idx]
            curr_answer = curr_data['answer_list_of_lists'][question_list_idx]
            
            num_options = len(curr_question)
            meta_label = curr_data['meta_label']
            
            if meta_label in IMPLICIT_ARMS:
                arm_group = 'implicit'
            elif meta_label in EXPLICIT_ARMS:
                arm_group = 'explicit'
            else:
                raise Exception('meta_label not in IMPLICIT_ARMS or EXPLICIT_ARMS')
                
            cat_key = f"{num_options}-way_{arm_group}"
            if cat_key not in categories:
                raise Exception('cat_key not in categories')
                
            curr_answer_label = curr_question.index(curr_answer)
            
            try:
                out_clean = output.replace("*", "").replace("#", "").replace("```python", "").replace("```", "")
                conf_str = out_clean.split('DECISION:')[-1].strip().split('\n')[0].strip()
                confidence_scores = ast.literal_eval(conf_str.split(')')[0] + ')')
                
                raw = np.array(confidence_scores, dtype=np.float64)
                probs = raw / raw.sum()
                probs = np.clip(probs, 1e-12, 1.0)
                probs = probs / probs.sum()
                
                model_pred = np.argmax(probs).item()
                prob_of_true_class = probs[curr_answer_label]
                
            except Exception:
                raise Exception(f'except Exception; output: {output}')
                
            run_data[cat_key]['y_true'].append(curr_answer_label)
            run_data[cat_key]['y_pred'].append(model_pred)
            run_data[cat_key]['y_prob_true'].append(prob_of_true_class)
            
        # 2. Compute metrics strictly for THIS run
        current_run_metrics = {}
        for cat in categories:
            y_t = run_data[cat]['y_true']
            y_p = run_data[cat]['y_pred']
            y_prob = run_data[cat]['y_prob_true']
            
            if not y_t:
                # raise Exception('y_t is empty')
                continue
                
            # --- Overall Category Metrics ---
            if cat == '3-way_explicit':
                # Map 0 & 1 -> 1 (Positive), Map 2 -> 0 (Negative) for overall F1/Acc
                y_t_mapped = [1 if t in [0, 1] else 0 for t in y_t]
                y_p_mapped = [1 if p in [0, 1] else 0 for p in y_p]
                
                cat_f1 = f1_score(y_t_mapped, y_p_mapped, average='macro', zero_division=0)
                
                mask_1 = [i for i, t in enumerate(y_t_mapped) if t == 1]
                acc_1 = sum(y_p_mapped[i] == 1 for i in mask_1) / len(mask_1) if mask_1 else 0.0
                
                mask_0 = [i for i, t in enumerate(y_t_mapped) if t == 0]
                acc_0 = sum(y_p_mapped[i] == 0 for i in mask_0) / len(mask_0) if mask_0 else 0.0
                
                active_classes = (1 if mask_1 else 0) + (1 if mask_0 else 0)
                cat_w_acc = (acc_1 + acc_0) / active_classes if active_classes > 0 else 0.0
            else:
                # Standard 2-way evaluation
                cat_f1 = f1_score(y_t, y_p, average='macro', zero_division=0)
                
                class_accs_run = []
                for cls in set(y_t):
                    cls_mask = [i for i, val in enumerate(y_t) if val == cls]
                    acc = sum(y_p[i] == cls for i in cls_mask) / len(cls_mask)
                    class_accs_run.append(acc)
                cat_w_acc = np.mean(class_accs_run) if class_accs_run else 0.0
                
            # Run overall cross-entropy
            cat_ce = np.mean([-np.log(max(p, 1e-12)) for p in y_prob])
            
            # --- Per-Class Metrics (Always uses ORIGINAL unmapped labels to display 3) ---
            classes_metrics = {}
            for cls in sorted(list(set(y_t))):
                cls_mask = [i for i, val in enumerate(y_t) if val == cls]
                count = len(cls_mask)
                
                if count > 0:
                    cls_acc = sum(y_p[i] == cls for i in cls_mask) / count
                    cls_ce = np.mean([-np.log(max(y_prob[i], 1e-12)) for i in cls_mask])
                else:
                    cls_acc = 0.0
                    cls_ce = 0.0
                    
                classes_metrics[cls] = {'count': count, 'acc': cls_acc, 'ce': cls_ce}
                
            current_run_metrics[cat] = {
                'overall': {'w_acc': cat_w_acc, 'f1': cat_f1, 'ce': cat_ce},
                'classes': classes_metrics
            }
            
        all_runs_metrics.append(current_run_metrics)

    # 3. Aggregate across all runs
    aggregated = {cat: {'overall': {'w_acc': [], 'f1': [], 'ce': []}, 'classes': {}} for cat in categories}
    
    for run_metrics in all_runs_metrics:
        for cat, data in run_metrics.items():
            aggregated[cat]['overall']['w_acc'].append(data['overall']['w_acc'])
            aggregated[cat]['overall']['f1'].append(data['overall']['f1'])
            aggregated[cat]['overall']['ce'].append(data['overall']['ce'])
            
            for cls, cls_data in data['classes'].items():
                if cls not in aggregated[cat]['classes']:
                    aggregated[cat]['classes'][cls] = {'count': [], 'acc': [], 'ce': []}
                aggregated[cat]['classes'][cls]['count'].append(cls_data['count'])
                aggregated[cat]['classes'][cls]['acc'].append(cls_data['acc'])
                aggregated[cat]['classes'][cls]['ce'].append(cls_data['ce'])

    # 4. Final Averaged Output
    final_metrics = {}
    print("\n" + "="*55)
    print(f"=== Averaged Results Across {len(list_of_eval_rets)} Runs ===")
    print("="*55)
    
    for cat in categories:
        res = aggregated[cat]
        if not res['overall']['f1']:
            print(f"\n" + "-"*55)
            print(f"CATEGORY: {cat.upper()}")
            print("-" * 55)
            print("No data available for this category.")
            continue
            
        # Average overall metrics across runs
        w_acc_avg = np.mean(res['overall']['w_acc'])
        f1_avg = np.mean(res['overall']['f1'])
        ce_avg = np.mean(res['overall']['ce'])
        
        final_metrics[cat] = {
            'Weighted_Accuracy': w_acc_avg,
            'Macro-F1': f1_avg, 
            'Cross_Entropy': ce_avg
        }
        
        print(f"\n" + "-"*55)
        print(f"CATEGORY: {cat.upper()}")
        print("-" * 55)
        
        f1_desc = "F1 (macro, binary mapped)" if cat == '3-way_explicit' else "F1 (macro)"
        print(f"Overall — w_acc: {w_acc_avg:.4f}, {f1_desc}: {f1_avg:.4f}, mean entropy: {ce_avg:.4f}")
        
        # Average per-class metrics across runs
        for cls in sorted(res['classes'].keys()):
            cls_data = res['classes'][cls]
            avg_count = np.mean(cls_data['count'])
            avg_acc = np.mean(cls_data['acc'])
            avg_ce = np.mean(cls_data['ce'])
            
            print(f"Class {cls} — count: {avg_count:.1f}, acc: {avg_acc:.4f}, mean entropy: {avg_ce:.4f}")

    # 5. Final Overall (Average of the 3 Categories)
    valid_cats = [c for c in categories if c in final_metrics]
    overall_w_acc = np.mean([final_metrics[cat]['Weighted_Accuracy'] for cat in valid_cats])
    overall_f1 = np.mean([final_metrics[cat]['Macro-F1'] for cat in valid_cats])
    overall_ce = np.mean([final_metrics[cat]['Cross_Entropy'] for cat in valid_cats])

    final_metrics['overall'] = {
        'Weighted_Accuracy': overall_w_acc,
        'Macro-F1': overall_f1,
        'Cross_Entropy': overall_ce
    }

    print(f"\n" + "="*55)
    print(f"TOTAL: OVERALL (Simple Average of {len(valid_cats)} Categories)")
    print("="*55)
    print(f"Overall — w_acc: {overall_w_acc:.4f}, F1 (macro): {overall_f1:.4f}, mean entropy: {overall_ce:.4f}\n")
        
    return final_metrics

if __name__ == '__main__':
    
    # TODO: Define the paths to the eval ret files and the benchmark data
    
    list_of_eval_rets = []
    with open(PATH_TO_RET_1, 'rb') as f:
        ret_1 = pickle.load(f)
    
    with open(PATH_TO_RET_2, 'rb') as f:
        ret_2 = pickle.load(f)
        
    with open(PATH_TO_RET_3, 'rb') as f:
        ret_3 = pickle.load(f)
    
    with open(PATH_TO_BENCHMARK_DATA, 'rb') as f:
        benchmark_data = pickle.load(f)
        
    list_of_eval_rets.append(ret_1)
    list_of_eval_rets.append(ret_2)
    list_of_eval_rets.append(ret_3)
    
    # Evaluate across multiple runs
    evaluate_multiple_runs(list_of_eval_rets, benchmark_data)
