---
tags: Github
---

# tbrain-tomofun-dog-audio-classification
## Dataset Folder Structure
```bash
dataset
  ├── public_test
  ├── meta_train.csv
  └── train
```

## Survay
- [BERT](https://paperswithcode.com/method/bert)
- [Transformers](https://huggingface.co/transformers/)
- [Github: Transformers](https://github.com/huggingface/transformers)
- [Github: SpecAugment](https://github.com/DemisEom/SpecAugment)
- [第十九屆旺宏科學獎 成果報告書](https://www.mxeduc.org.tw/scienceaward/history/projectDoc/19th/doc/SA19-226_final.pdf)
- [Environmental Sound Classification Base on CNN and LightGBM](https://image.hanspub.org/Html/10-1541547_32564.htm)
- [librosa语音信号处理](https://www.cnblogs.com/LXP-Never/p/11561355.html)
## Discussion
- Wav2Vec2CTCTokenizer
- BertForSequenceClassification
- BertForMultipleChoice
- 將librosa.load(sr=22050)，可以讓nfccs的變成128 x 216，原本是128x79。
	```python=
	audio, sampling_rate = librosa.load(input_file, sr=22050)
	mfccs = librosa.feature.mfcc(audio, sr=22050, n_mfcc=128)
	```