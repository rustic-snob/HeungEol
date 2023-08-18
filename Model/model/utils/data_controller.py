import re
import pickle
import torch
import yaml
import pandas as pd
import pytorch_lightning as pl
import re

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

class Dataset(Dataset):
    """
    Dataloader에서 불러온 데이터를 Dataset으로 만들기
    """

    def __init__(self, data, train=False):
        self.inputs = data[0]
        self.syl_structure = data[1]
        self.train = train

    def __getitem__(self, idx):
        if self.train:
            inputs = {key: val[idx].clone().detach()
                    for key, val in self.inputs.items()}
            syl_structure = self.syl_structure[idx]
            
            return inputs, syl_structure
        else:
            return self.inputs[idx]
        
    def __len__(self):
        if self.train:        
            return len(self.inputs['input_ids'])
        else:
            return len(self.inputs)
        
class Dataloader(pl.LightningDataModule):
    
    """
    원본 데이터를 불러와 전처리 후 Dataloader 만들어 Dataset에 넘겨 최종적으로 사용할 데이터셋 만들기
    """

    def __init__(self, tokenizer, CFG):
        super(Dataloader, self).__init__()
        self.CFG = CFG
        self.tokenizer = tokenizer
        
        self.train_valid_df, self.predict_df = load_data()

        self.train_dataset = None
        self.val_dataset = None
        
        predict = self.preprocessing(self.predict_df)
        self.predict_dataset = Dataset(predict)

    def tokenizing(self, x, train=False):
        
        """ 
        Arguments:
        x: pd.DataFrame

        Returns:
        inputs: Dict({'input_ids', 'attention_mask', 'labels', ...}), 각 tensor(num_data, max_length)
        """
        
        x['freq'] = [len(self.tokenizer(text)['input_ids']) for text in tqdm(x.gen_lyrics)]
        x = x[x['freq'] <= 300]
        
        if not self.CFG['induce_align']:
            instruction = "다음 조건에 어울리는 가사를 생성하시오. 주어진 음절 수를 절대 벗어나지 말 것. 제목과 장르에 어울려야 할 것. 생성 형식은 [가사 / 가사 / 가사 / 가사]와 같음."
            
        else:
            instruction = "다음 조건에 어울리는 가사를 생성하시오. 주어진 음절 수를 절대 벗어나지 말 것. 제목과 장르에 어울려야 할 것. 생성 형식은 [(음절)가사 / (음절)가사 / (음절)가사 / (음절)가사]와 같음."
            
        x['instruction'] = instruction
        
        prompts_list = [f"### Instruction(명령어):\n{row['instruction']}\n\n### Input(입력):\n음절 수는 [{row['gen_notes']}], 제목은 [{row['제목']}], 장르는 [{row['장르']}]이다.\n\n### Response(응답): \n" for _ , row in x.iterrows()]
            
        if train:
            if self.CFG['induce_align']:
                x['gen_lyrics'] = x['gen_lyrics'].apply(lambda elem: ' / '.join(['(' + str(len(e.replace(' ',''))) + ')' + e for e in elem.split(' / ')]))
                
            answers_list = [f"{row['gen_lyrics']}" for _ , row in x.iterrows()]
            
            inputs = self.tokenizer(
                prompts_list,
                answers_list,
                return_tensors='pt',
                padding=True,
                truncation='only_first',
                max_length=self.CFG['train']['token_max_len'],
                add_special_tokens=True,
            )
            
            labels = [[-100] * (x.tolist().index(1)) + inputs['input_ids'][idx][x.tolist().index(1):].tolist() for idx, x in tqdm(enumerate(inputs['token_type_ids']))]
            inputs['labels'] = torch.tensor(labels)
            
            return inputs

        else:            
            return prompts_list

    def preprocessing(self, x, train=False):
        DC = DataCleaning(self.CFG['select_DC'])
        DA = DataAugmentation(self.CFG['select_DA'])

        if train:
            # x = DC.process(x)
            # x = DA.process(x)         
            
            # 텍스트 데이터 토큰화
            
            # 1st finetuning
            if 'HeungEol' not in self.CFG['train']['model_name']:
                train_x, val_x = train_test_split(x,
                                                test_size=self.CFG['train']['test_size'],
                                                shuffle=True,
                                                random_state=self.CFG['seed'])
                
            # 2nd finetuning
            else:
                val_x = x.sample(n=73, random_state=self.CFG['seed'])
                train_x = x.drop(val_x.index)
            
            train_inputs = self.tokenizing(train_x, train=True)
            
            val_inputs = self.tokenizing(val_x, train=True)

            return (train_inputs, list(train_x['gen_notes'])), (val_inputs, list(val_x['gen_notes']))
        
        else:
            # x = DC.process(x)

            # 텍스트 데이터 토큰화
            predict_inputs = self.tokenizing(x, train=False)
            
            return predict_inputs

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터 준비
            train, val = self.preprocessing(self.train_valid_df, train=True)
            self.train_dataset = Dataset(train, train=True)
            self.val_dataset = Dataset(val, train=True)
        else:
            # 평가 데이터 호출
            predict = self.preprocessing(self.predict_df)
            self.predict_dataset = Dataset(predict)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.CFG['train']['batch_size'], shuffle=False)
    
class DataCleaning():
    """
    config select DC에 명시된 Data Cleaning 기법을 적용시켜주는 클래스
    """
    def __init__(self, select_list):
        self.select_list = select_list
        
    def process(self, df):
        if self.select_list:
            for method_name in self.select_list:
                print('method name: '+ method_name)
                
                print('before:')
                print(df.head(1))
                
                method = eval("self." + method_name)
                df = method(df)
                
                print('after:')
                print(df.head(1))

        return df

    """
    data cleaning 코드
    """
    def data_cleaning_demo(self, df):
        """ 
        Arguments:
        df: Cleaning을 수행하고자 하는 DataFrame
        
        Return:
        df: Cleaning 작업이 완료된 DataFrame
        """
                
        return df


class DataAugmentation():
    """
    config select DA에 명시된 Data Augmentation 기법을 적용시켜주는 클래스
    """

    def __init__(self, select_list):
        self.select_list = select_list

    def process(self, df):
        if self.select_list:
            aug_df = pd.DataFrame(columns=df.columns)

            for method_name in self.select_list:
                method = eval("self." + method_name)
                aug_df = pd.concat([aug_df, method(df)])

            df = pd.concat([df, aug_df])

        return df
    
    """
    data augmentation 코드
    """
    
    def data_augmentation_demo(self, df):
        """

        Arguments:
        df: augmentation을 수행하고자 하는 DataFrame

        Return:
        df: augmentation된 DataFrame (exclude original)
        """
        
        return df


def load_data():
    """
    학습 데이터와 테스트 데이터 DataFrame 가져오기
    """
    with open('./config/use_config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)
        
    # 1st finetuning
    if 'HeungEol' not in CFG['train']['model_name']:
        df = pd.read_csv('../Data/Dataset/ready/' + CFG['data_name']) 
        df.dropna(inplace=True)
        df['units'] = [len(x.split(' / ')) for x in df.gen_notes]
        df = df[df['units']<70]
        df = df[df['units'] != 1]
        df['syls'] = [map(int, x.split(' / ')) for x in df.gen_notes]
        df = df[~df['장르'].str.contains('랩')]
        df = df[~df['장르'].str.contains('CCM')]
        df = df[~df['장르'].str.contains('J-POP')]
        df = df[~df['장르'].str.contains('-')]
        train_valid_df, predict_df = train_test_split(df, shuffle = True, test_size=CFG['train']['test_size'], random_state=CFG['seed'])
        train_valid_df = train_valid_df[:CFG['max_sample_num']]
    
    # 2nd finetuning
    else:
        df = pd.read_csv('./dataset/dataset_2nd_finetune_v1.0.csv')
        predict_df = df.sample(n=100, random_state=CFG['seed'])
        train_valid_df = df.drop(predict_df.index)
        
    return train_valid_df, predict_df