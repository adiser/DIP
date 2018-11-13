import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse

def generate_predictions(x):
    p_sah = x['p_sah']
    p_sdh = x['p_sdh']
    preds = []
    for i in range(len(p_sah)):
        class_1_score = p_sah.iloc[i]
        class_2_score = p_sdh.iloc[i]
        
        if class_1_score > class_2_score:
            preds.append(0)
        else:
            preds.append(1)
            
    return preds

def eval_scores(files, scores):
    
    if len(files) != len(scores):
        raise ValueError("Different numbers of scores and files")

    dividing_factor = sum(scores)
    result = pd.DataFrame(columns = ['p_sah', 'p_sdh'])

    for i in range(len(files)):
        score = scores[i]
        df = pd.read_csv(files[i], index_col = 'Unnamed: 0')

        if i == 0:
            result['p_sah'] = score * df['p_sah']
            result['p_sdh'] = score * df['p_sdh']
            continue

        result['p_sah'] += score * df['p_sah']
        result['p_sdh'] += score * df['p_sdh']

    final_pred = generate_predictions(result)
    y_true = list(df['targets'])

    result.to_csv('result.csv')

    print("Accuracy: \n {}".format(accuracy_score(y_true,final_pred)))
    print("Classification report: \n {}".format(classification_report(y_true,final_pred)))
    print("Confusion matrix: \n {}".format(confusion_matrix(y_true,final_pred)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--files', type=str, nargs='+')
    parser.add_argument('--scores', type=float, nargs='+')
    
    args = parser.parse_args()
    eval_scores(args.files, args.scores)

