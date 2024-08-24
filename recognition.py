from conv_mamba_block import conv_mamba_block
from classification import classification
import torch.nn as nn
from Bio import SeqIO
from torch.utils.data import DataLoader
from Processing import Processing
import copy
import torch
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score



def build_dataset(fasta_file):
    sequences = []
    labels = []
    fp = open(fasta_file)
    for seq_record in SeqIO.parse(fp, 'fasta'):
        seq = seq_record.seq.upper()
        label = seq_record.id
        if len(seq) <= "Length of the longest sequence in the data set":
            seq += '_' * ("Length of the longest sequence in the data set" - len(seq))
        sequences.append(str(seq))
        labels.append(int(label))
    return sequences, labels


test_seqs, test_labels = build_dataset('your_dataset')
testDataset = Processing(test_seqs, test_labels)
testDataloader = DataLoader(testDataset, batch_size=64, shuffle=True, drop_last=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.n_vocab = 7
        self.d_model = 128
        self.d_hidden = 384
        self.drop_out_rate = 0.1
        self.num_encoder = 2
        self.seq_len = "Length of the longest sequence in the data set"
        self.feedforward_factor = 1
        self.n_heads = 8
        self.embedding = nn.Embedding(self.n_vocab, self.d_model, padding_idx=self.n_vocab - 1)
        self.encoder = conv_mamba_block(self.d_model, self.d_hidden, self.drop_out_rate, self.feedforward_factor, self.n_heads, self.seq_len)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(self.num_encoder)])
        self.classification = classification(self.seq_len, self.d_model, self.d_hidden)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.permute(0, 2, 1)
        x = self.classification(x)
        
        return x


    
    

the_model = Model().cuda()
the_model.load_state_dict(torch.load("best_performance_model"))
"""For example, if you want to recognise mouse enhancers, you can load the model using the_model.load_state_dict(torch.load(The-Nucleotide-Mamba_best_model\Nucleotide_Mamba_best_performance_on_dummy_mouse_enhancers_ensembl.pkl))"""



total_train_step = 0
total_test_step = 0
best_predict_accuracy = 0




print("-------------第 {} 轮训练开始------------".format(i + 1))
total_training_accuracy = 0
total_training_loss = 0
total_test_accuracy = 0
total_test_loss = 0
total_valid_accuracy = 0
total_valid_loss = 0
    

the_model.eval()
with torch.no_grad():
    k = 0
    for data in testDataloader:
        seq, targets = data
        seq = seq.cuda()
        targets = targets.cuda()
        outputs = the_model(seq.long())
        preds = outputs.argmax(1)
        if k == 0:
            predict_labels = preds
            true_labels = targets
        else:
            predict_labels = torch.cat((predict_labels, preds), dim=0)
            true_labels = torch.cat((true_labels, targets), dim=0)
        k = 1
        accuracy = (preds == targets).sum()
        total_test_accuracy = total_test_accuracy + accuracy
        total_test_step = total_test_step + 1

    predict_labels = predict_labels.cpu().detach().numpy()
    true_labels = true_labels.cpu().detach().numpy()
    print('Precision: ', precision_score(predict_labels, true_labels))
    print('Recall: ', recall_score(predict_labels, true_labels))
    print('Accuracy: ', accuracy_score(predict_labels, true_labels))
    print('MCC: ', matthews_corrcoef(predict_labels, true_labels))
    print('F1:', f1_score(predict_labels, true_labels))
gc.collect()
torch.cuda.empty_cache()