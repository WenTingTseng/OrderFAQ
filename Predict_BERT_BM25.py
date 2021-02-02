from transformers import BertTokenizer
import torch
import similarity
import pickle
import jieba
import sys
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AdamW

jiebaFile='BM25/userdict.txt'
pklFile='trained_model/data_features_FAQ.pkl'
configFile='trained_model/config.json'
modelFile='trained_model/pytorch_model.bin'
questionTestFile='Datatset/question_test.txt'
AnswerTestFile='Datatset/answer_test.txt'
AnswerTypeQFile='Datatset/anstype_q.txt'
AnswerTypeFile='Datatset/anstype.txt'
AnswerTypeIdxFile='Datatset/anstype_idx.txt'
stopwordFile='BM25/stopword.txt'
SegQusFile='BM25/seg_question.txt'
AnswerTestLabelFile='Datatset/answer_test_label.txt'
jieba.load_userdict(jiebaFile)

def toBertIds(q_input):
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q_input)))

if __name__ == "__main__":
    # load and init
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese-vocab.txt')
    pkl_file = open(pklFile, 'rb')
    data_features = pickle.load(pkl_file)
    answer_dic = data_features['answer_dic']
    question_dic = data_features['question_dic']
   
    bert_config, bert_class, bert_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
    config = bert_config.from_pretrained(configFile)
    model = bert_class.from_pretrained(modelFile, from_tf=bool('.ckpt' in 'bert-base-chinese'),config=config)
    model.eval()
    #讀取驗證資料問題集(user query)
    q = open(questionTestFile, "r",encoding="utf-8")
    q_inputs = q.readlines()
    q.close()

    #讀取驗證資料回答集(ans of query)
    a = open(AnswerTestFile, "r",encoding="utf-8")
    answer = a.readlines()
    a.close()

    #讀取train data question
    train = open(AnswerTypeQFile,"r",encoding="utf-8")
    trainQ = train.readlines()
    train.close()

    #讀取train data answer
    train = open(AnswerTypeFile, "r",encoding="utf-8")
    trainA = train.readlines()
    train.close()

    train = open(AnswerTypeIdxFile, "r",encoding="utf-8")
    anstypeIdx = train.readlines()
    train.close()

    #讀取stop word
    stopWords=[]
    with open(stopwordFile, 'r', encoding='UTF-8') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)

   
    #讀取BM25的訓練資料(標準問句斷過詞)
    BM25_train_data=[]
    with open(SegQusFile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            BM25_train_data.append(line.replace("\r", "").replace("\n", ""))

     #讀取驗證資料回答集(ans of query)
    a = open(AnswerTestLabelFile, "r",encoding="utf-8")
    answer_lablel = a.readlines()
    a.close()

    predict_ans=[]

    #設定使用者單句問句
    q_inputs=["火腿雞蛋堡內容物?"]

    for q_input in q_inputs:
        # BM25欲處理去除stopword
        words = [w for w in q_input.split() if w not in stopWords]
        q_input="".join(words)
        ##################### BM25 ##############################
        seg_list_query = " ".join(jieba.cut(q_input))
        dataset=dict()
        dataset["".join(seg_list_query)]=dict()
        for train in BM25_train_data:
            sim=similarity.ssim(seg_list_query,train,model='bm25')
            dataset["".join(seg_list_query)][train]=dataset["".join(seg_list_query)].get(train,0)+sim
      
        for r in dataset:  
            top=sorted(dataset[r].items(),key=lambda x:x[1],reverse=True)    
            bert_ids = toBertIds(q_input)
            assert len(bert_ids) <= 512
            input_ids = torch.LongTensor(bert_ids).unsqueeze(0)
            # predict
            outputs = model(input_ids)
            
            predicts = outputs[:2]
            predicts = predicts[0]
            
            sort_val=torch.sort(predicts,descending=True) 
        
            BM25Rank=dict()
            for i in range(len(top)):
                BM25Rank[top[i][0].replace(" ", "")]=top[i][1]
      
            BERTRank=[]
            for i in range(len(sort_val[0].detach().numpy()[0])):
                BERTRank.append(sort_val[0].detach().numpy()[0][i])
            
            ReRank=dict()
            Rank_Q=[]
            Rank_score=[]
            Rank_idx=[]
            
            for i in range(len(BERTRank)):
                ReRank[sort_val[1].numpy()[0][i]]=BERTRank[i]+BM25Rank[trainQ[sort_val[1].numpy()[0][i]].replace("\r", "").replace("\n", "").replace("，", "").replace("？", "").replace("?", "").replace(" ", "").replace("、", "").replace("？", "")]
            Rank=sorted(ReRank.items(),key=lambda x:x[1],reverse=True)
            
            for (i,score) in enumerate(Rank):
                Rank_Q.append(trainQ[score[0]])
                Rank_idx.append(anstypeIdx[score[0]])
                Rank_score.append(score[1])
            
        
            max_val_label=Rank[0][0]
            input("預測答案")
            print(trainA[max_val_label])
            predict_ans.append(max_val_label)
    
    #單句問句所預測的答案

    #整批預測
    # count=0
   
    # for q,i,j in zip(q_inputs,answer_lablel,predict_ans):
    #     if int(i)==int(j):
    #         count+=1
   
    # acc=(count/len(answer))*100
    # print("accuracy:",acc)
