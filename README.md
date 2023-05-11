# VITS for Japanese

*VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*

*Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In the repository, I will introduce a VITS model for Japanese on pytorch version 2.0.0 that customed from [VITS model](https://github.com/jaywalnut310/vits).*

We also provide the [pretrained models](https://www.dropbox.com/s/e0h13tufx2oobn2/G_523000.pth?dl=0).
<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
0. Download datasets
    1. Download and extract the [Japanese Speech dataset](https://sites.google.com/site/shinnosuketakamichi/publication/jsut), then choose `basic5000` dataset and move to `jp_dataset` folder. 
0. Run preprocessing if you use your own datasets.
```sh
# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for Japanese dataset have been already provided.
python preprocess.py --text_index 1 --filelists filelists/jp_audio_text_train_filelist.txt filelists/jp_audio_text_val_filelist.txt filelists/jp_audio_text_test_filelist.txt
```


## Training Example
```sh
# JP Speech
python train.py -c configs/jp_base.json -m jp_base
```


## Inference Example
To get pretrained model for Japanese:
```sh 
sh startup.sh
```
See [vits_apply.ipynb](vits_apply.ipynb) or run `streamlit run app.py` to see demo on streamlit-share. 
