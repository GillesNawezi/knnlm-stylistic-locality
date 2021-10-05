import torch
from fairseq.data import Dictionary

dictionary = Dictionary.load('data-bin/java-huge-bpe-2000-half/dict.txt')
bpe_cont = '@@'
bpe_toks = {
    i
    for i in range(len(dictionary))
    if dictionary[i].endswith(bpe_cont)
}

saved_prediction = torch.load('prediction.pt', map_location=torch.device('cpu'))

for acc_at_k in [1, 5, 10, 20]:
    total_num = 0
    correct = 0
    for topk_pred, ref in zip(saved_prediction['topk'], saved_prediction['ref']):
        pred = topk_pred[:, :acc_at_k]
        is_correct = True
        for p, t in zip(pred, ref):
            if t.item() in bpe_toks:
                is_correct = is_correct and (t in p)
            else:
                total_num += 1
                if is_correct and (t in p):
                    correct += 1
                is_correct = True
    print('acc at k', acc_at_k, correct/total_num)
