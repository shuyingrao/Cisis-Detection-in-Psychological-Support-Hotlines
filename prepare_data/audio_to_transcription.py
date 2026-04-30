# -*- coding: utf-8 -*-
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import time
import requests
import urllib
import pandas as pd
from sympy import re

lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa


    def upload(self):
        print("上传部分：")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict["fileSize"] = file_len
        param_dict["fileName"] = file_name
        param_dict["duration"] = 2000
        param_dict["trackMode"] = 2
        param_dict["eng_vad_margin"] = 1
        print("upload参数：", param_dict)
        data = open(upload_file_path, 'rb').read(file_len)

        response = requests.post(url=lfasr_host + api_upload+"?"+urllib.parse.urlencode(param_dict),
                                 headers={"Content-type": "application/json"}, data=data)
        print("upload_url:", response.request.url)
        result = json.loads(response.text)
        print("upload resp:", result)
        return result


    def get_result(self):
        uploadresp = self.upload()
        orderId = uploadresp['content']['orderId']
        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict['orderId'] = orderId
        param_dict['resultType'] = "transfer"  # , predict
        print("")
        print("查询部分：")
        print("get result参数：", param_dict)
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/x-www-form-urlencoded"})
            # print("get_result_url:",response.request.url)
            result = json.loads(response.text)
            # print(result)
            status = result['content']['orderInfo']['status']
            print("status=",status)
            if status == 4:
                break
            time.sleep(5)
        # print("get_result resp:",result)
        return result



if __name__ == '__main__':
    base_dir = ".../hotline"
    audio_dir = Path(base_dir) / "data" / "audio" / "2023_Y"
    xlsx_dir = Path(base_dir) / "data" / "transcript" / "2023_Y"
    for file_wav in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, file_wav)
        api = RequestApi(appid="1943ba0e",
                         secret_key="...",
                         upload_file_path=audio_path)
        # r"audio/lfasr_涉政.wav"
        a = api.get_result()
        b = json.loads(a['content']['orderResult'])
        num_sentence = len(b['lattice'])
        s = []
        for sen_i in range(num_sentence):
            temp_sen = json.loads(b['lattice'][sen_i]['json_1best'])
            num_word = len(temp_sen['st']['rt'][0]['ws'])
            s1 = ''
            for word_i in range(num_word):
                s1 = s1+temp_sen['st']['rt'][0]['ws'][word_i]['cw'][0]['w']
            s.append([temp_sen['st']['bg'], temp_sen['st']['ed'], s1])
        df = pd.DataFrame(s,columns=['begin', 'end', 'content'])  # ms
        filename = os.path.join(xlsx_dir, os.path.basename(audio_path)[: -4] + ".xlsx")
        df.to_excel(filename, index=False)

    sentences_dir  = Path(base_dir) / "data" / "Sentences" / "2023_Y" 
    for filename in os.listdir(xlsx_dir):
        file = xlsx_dir + filename
        texts = pd.read_excel(file)
        temp = texts['content'].tolist()
        # temp = texts[texts['Speaker'] == 1]
        df_filtered = temp.dropna()
        sentence_list = df_filtered['content'].tolist()
        numbered_sentences = [f"{i}. {sentence}" for i, sentence in enumerate(sentence_list)]

        save_df = df_filtered.copy()
        save_df['begin'] = df_filtered['begin']
        save_df['end'] = df_filtered['end']
        save_df['content'] = numbered_sentences

        save_df.to_excel(sentences_dir + filename, index=False)

    transcript_csv_dir = Path(base_dir) / "data" / "transcript" / "transcript_2023_Y.csv"
    with open(transcript_csv_dir, 'w') as fwrite:
        for file in os.listdir(xlsx_dir):
            text_file = xlsx_dir + file
            texts = pd.read_excel(text_file)
            # temp = texts[texts['Speaker'] == 1]
            temp = texts
            temp = temp['content'].tolist()
            new_list = [x for x in temp if pd.isnull(x) == False]
            str = ''
            doc_text = str.join(new_list)
            filename = os.path.basename(text_file)[: -5]
            w_line = filename + "," + doc_text + "\n"
            fwrite.write(w_line)

    # first column is subject, second column is content
    transcript_json_dir = Path(base_dir) / "data" / "transcript" / "transcript_2023_Y.json"
    data = pd.read_csv(transcript_csv_dir, encoding='gb18030')
    data.set_index('Subject', inplace=True)
    with open(transcript_json_dir,"w+",encoding='utf-8') as f:
        for name in data.index:
            dict1 = {}
            dict1['subject'] = name
            doc_token = data.loc[name, 'Content']
            # 只保留中文、大小写字母和阿拉伯数字
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            doc_token = re.sub(reg, '', doc_token)
            # print(doc_token)
            dict1['doc_text'] = doc_token
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)