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
- [基于CNN和LightGBM的环境声音分类](https://www.hanspub.org/journal/PaperInformation.aspx?PaperID=32564&#f5)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [Day 16 — LightGBM](https://medium.com/@falconives/day-16-lightgbm-aa447494a763)
- [LightGBM两种使用方式](https://www.cnblogs.com/chenxiangzhen/p/10894306.html)
- [librosa语音信号处理](https://www.cnblogs.com/LXP-Never/p/11561355.html)
- [urbansound8資料集 Kaggle也有](https://urbansounddataset.weebly.com/urbansound8k.html)
- [urbansound8資料集 download link](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz)
- [pytorch提取中間層輸出](https://segmentfault.com/a/1190000039426499)
- [LGBM參數設定](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
## Discussion
- Wav2Vec2CTCTokenizer
- BertForSequenceClassification
- BertForMultipleChoice
- 將librosa.load(sr=22050)，可以讓nfccs的變成128 x 216，原本是128x79。
	```python=
	audio, sampling_rate = librosa.load(input_file, sr=22050)
	mfccs = librosa.feature.mfcc(audio, sr=22050, n_mfcc=128)
	```

## Resnet18
```
for k,v in net.named_parameters():
    print(k, v.requires_grad)
'''
conv1.weight False
bn1.weight False
bn1.bias False
layer1.0.conv1.weight False
layer1.0.bn1.weight False
layer1.0.bn1.bias False
layer1.0.conv2.weight False
layer1.0.bn2.weight False
layer1.0.bn2.bias False
layer1.1.conv1.weight False
layer1.1.bn1.weight False
layer1.1.bn1.bias False
layer1.1.conv2.weight False
layer1.1.bn2.weight False
layer1.1.bn2.bias False
layer2.0.conv1.weight False
layer2.0.bn1.weight False
layer2.0.bn1.bias False
layer2.0.conv2.weight False
layer2.0.bn2.weight False
layer2.0.bn2.bias False
layer2.0.downsample.0.weight False
layer2.0.downsample.1.weight False
layer2.0.downsample.1.bias False
layer2.1.conv1.weight False
layer2.1.bn1.weight False
layer2.1.bn1.bias False
layer2.1.conv2.weight False
layer2.1.bn2.weight False
layer2.1.bn2.bias False
layer3.0.conv1.weight True
layer3.0.bn1.weight True
layer3.0.bn1.bias True
layer3.0.conv2.weight True
layer3.0.bn2.weight True
layer3.0.bn2.bias True
layer3.0.downsample.0.weight True
layer3.0.downsample.1.weight True
layer3.0.downsample.1.bias True
layer3.1.conv1.weight True
layer3.1.bn1.weight True
layer3.1.bn1.bias True
layer3.1.conv2.weight True
layer3.1.bn2.weight True
layer3.1.bn2.bias True
layer4.0.conv1.weight True
layer4.0.bn1.weight True
layer4.0.bn1.bias True
layer4.0.conv2.weight True
layer4.0.bn2.weight True
layer4.0.bn2.bias True
layer4.0.downsample.0.weight True
layer4.0.downsample.1.weight True
layer4.0.downsample.1.bias True
layer4.1.conv1.weight True
layer4.1.bn1.weight True
layer4.1.bn1.bias True
layer4.1.conv2.weight True
layer4.1.bn2.weight True
layer4.1.bn2.bias True
fc.weight True
fc.bias True
'''
'''