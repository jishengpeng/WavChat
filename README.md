<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/image-20241111160012489.png" alt="image-20241111160012489" style="zoom: 30%;" />
</div>

# 🚀Quick Start

1. [Introduction](#introduction)
2. [Overall](#overall)
   - [1. The organization of this survey](#1-the-organization-of-this-survey)
   - [2. General classification of spoken dialogue systems](#2-general-classification-of-spoken-dialogue-systems)
   - [3. Key capabilities of speech dialogue systems](#3-key-capabilities-of-speech-dialogue-systems)
   - [4. Publicly Available Speech Dialogue Models](#4-publicly-available-speech-dialogue-models)
3. [Representations of Spoken Dialogue Models](#representations-of-spoken-dialogue-models)
4. [Training Paradigm of Spoken Dialogue Model](#training-paradigm-of-spoken-dialogue-model)
5. [Streaming, Duplex, and Interaction](#streaming-duplex-and-interaction)
6. [Training Resources and Evaluation](#training-resources-and-evaluation)
   - [1. Training resources](#1-training-resources)
   - [2. Evaluation](#2-evaluation)
7. [Cite](#cite)

# 🔥What's new

- 2024.11.14 Release the first version of the WavChat (The full paper about 60 pages will be updated 11.14 on the arxiv)! 🎉 

## Introduction

This repository is the official repository of the **WavChat: A Survey of Spoken Dialogue Models** [![Paper page](https://huggingface.co/datasets/huggingface/badges/raw/main/paper-page-sm-dark.svg)](https://arxiv.org).

<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/image-20241112151419833.png" alt="img1-paper-list" style="zoom: 20%;" />

Figure 1: The timeline of existing spoken dialogue models in recent years.
</div>

> Abstract
>
> Recent advancements in spoken dialogue models, exemplified by systems like GPT-4o, have captured significant attention in the speech domain. In the broader context of multimodal models, the speech modality offers a direct interface for human-computer interaction, enabling direct communication between AI and users. Compared to traditional three-tier cascaded spoken dialogue models that comprise speech recognition (ASR), large language models (LLMs), and text-to-speech (TTS), modern spoken dialogue models exhibit greater intelligence. These advanced spoken dialogue models not only comprehend audio, music, and other speech-related features, but also capture stylistic and timbral characteristics in speech. Moreover, they generate high-quality, multi-turn speech responses with low latency, enabling real-time interaction through simultaneous listening and speaking capability. Despite the progress in spoken dialogue systems, there is a lack of comprehensive surveys that systematically organize and analyze these systems and the underlying technologies. To address this, **we have first compiled existing spoken dialogue systems in the chronological order and categorized them into the cascaded and end-to-end paradigms.** We then provide an in-depth overview of the core technologies in spoken dialogue models, covering aspects such as **speech representation, training paradigm, streaming, duplex, and interaction capabilities.** Each section discusses the limitations of these technologies and outlines considerations for future research. Additionally, we present a thorough review of **relevant datasets, evaluation metrics, and benchmarks** from the perspectives of training and evaluating spoken dialogue systems. We hope this survey will contribute to advancing both academic research and industrial applications in the field of spoken dialogue systems.

## Overall

#### 1. The organization of this survey

<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/img4-framework-v1_00.png" alt="WavChat - 副本" style="zoom:16%;" />

Figure 2: Orgnization of this survey.
</div>

#### 2. General classification of spoken dialogue systems

<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/image-20241112151732341.png" alt="img2-method" style="zoom: 30%;" />

Figure 3: A general overview of current spoken dialogue systems.
</div>

#### 3. Key capabilities of speech dialogue systems

<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/image-20241111165006367.png" alt="image-20241111165006367" style="zoom: 25%;" />

Figure 4: An overview of the spoken dialogue systems' nine ideal capabilities.
</div>

#### 4. Publicly Available Speech Dialogue Models

<div align='center'>
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>URL</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>AudioGPT</td>
            <td><a href="https://github.com/AIGC-Audio/AudioGPT">https://github.com/AIGC-Audio/AudioGPT</a></td>
        </tr>
        <tr>
            <td>SpeechGPT</td>
            <td><a href="https://github.com/0nutation/SpeechGPT">https://github.com/0nutation/SpeechGPT</a></td>
        </tr>
        <tr>
            <td>Freeze-Omni</td>
            <td><a href="https://github.com/VITA-MLLM/Freeze-Omni">https://github.com/VITA-MLLM/Freeze-Omni</a></td>
        </tr>
        <tr>
            <td>Baichuan-Omni</td>
            <td><a href="https://github.com/westlake-baichuan-mllm/bc-omni">https://github.com/westlake-baichuan-mllm/bc-omni</a></td>
        </tr>
        <tr>
            <td>GLM-4-Voice</td>
            <td><a href="https://github.com/THUDM/GLM-4-Voice">https://github.com/THUDM/GLM-4-Voice</a></td>
        </tr>
        <tr>
            <td>Mini-Omni</td>
            <td><a href="https://github.com/gpt-omni/mini-omni">https://github.com/gpt-omni/mini-omni</a></td>
        </tr>
        <tr>
            <td>Mini-Omni2</td>
            <td><a href="https://github.com/gpt-omni/mini-omni2">https://github.com/gpt-omni/mini-omni2</a></td>
        </tr>
        <tr>
            <td>FunAudioLLM</td>
            <td><a href="https://github.com/FunAudioLLM">https://github.com/FunAudioLLM</a></td>
        </tr>
       <tr>
            <td>Qwen-Audio</td>
            <td><a href="https://github.com/QwenLM/Qwen-Audio">https://github.com/QwenLM/Qwen-Audio</a></td>
        </tr>
        <tr>
            <td>Qwen2-Audio</td>
            <td><a href="https://github.com/QwenLM/Qwen2-Audio">https://github.com/QwenLM/Qwen2-Audio</a></td>
        </tr>
        <tr>
            <td>LLaMA3.1</td>
            <td><a href="https://www.llama.com">https://www.llama.com</a></td>
        </tr>
        <tr>
            <td>Audio Flamingo</td>
            <td><a href="https://github.com/NVIDIA/audio-flamingo">https://github.com/NVIDIA/audio-flamingo</a></td>
        </tr>
        <tr>
            <td>Spirit LM</td>
            <td><a href="https://github.com/facebookresearch/spiritlm">https://github.com/facebookresearch/spiritlm</a></td>
        </tr>
        <tr>
            <td>dGSLM</td>
            <td><a href="https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/dgslm">https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/dgslm</a></td>
        </tr>
        <tr>
            <td>Spoken-LLM</td>
            <td><a href="https://arxiv.org/abs/2305.11000">https://arxiv.org/abs/2305.11000</a></td>
        </tr>
        <tr>
            <td>LLaMA-Omni</td>
            <td><a href="https://github.com/ictnlp/LLaMA-Omni">https://github.com/ictnlp/LLaMA-Omni</a></td>
        </tr>
        <tr>
            <td>Moshi</td>
            <td><a href="https://github.com/kyutai-labs/moshi">https://github.com/kyutai-labs/moshi</a></td>
        </tr>
        <tr>
            <td>SALMONN</td>
            <td><a href="https://github.com/bytedance/SALMONN">https://github.com/bytedance/SALMONN</a></td>
        </tr>
        <tr>
            <td>LTU-AS</td>
            <td><a href="https://github.com/YuanGongND/ltu">https://github.com/YuanGongND/ltu</a></td>
        </tr>
        <tr>
            <td>VITA</td>
            <td><a href="https://github.com/VITA-MLLM/VITA">https://github.com/VITA-MLLM/VITA</a></td>
        </tr>
        <tr>
            <td>SpeechGPT-Gen</td>
            <td><a href="https://github.com/0nutation/SpeechGPT">https://github.com/0nutation/SpeechGPT</a></td>
        </tr>
        <tr>
            <td>Westlake-Omni</td>
            <td><a href="https://github.com/xinchen-ai/Westlake-Omni">https://github.com/xinchen-ai/Westlake-Omni</a></td>
        </tr>
        <tr>
            <td>MooER-Omni</td>
            <td><a href="https://github.com/MooreThreads/MooER">https://github.com/MooreThreads/MooER</a></td>
        </tr>
        <tr>
            <td>Hertz-dev</td>
            <td><a href="https://github.com/Standard-Intelligence/hertz-dev">https://github.com/Standard-Intelligence/hertz-dev</a></td>
        </tr>
        <tr>
            <td>Fish-Agent</td>
            <td><a href="https://github.com/fishaudio/fish-speech">https://github.com/fishaudio/fish-speech</a></td>
        </tr>
        <tr>
            <td>SpeechGPT2</td>
            <td><a href="https://0nutation.github.io/SpeechGPT2.github.io/">https://0nutation.github.io/SpeechGPT2.github.io/</a></td>
        </tr>
    </tbody>
</table>

Table 1: The list of publicly available speech dialogue models and their URL
</div>

## Representations of Spoken Dialogue Models

In the section Representations of Spoken Dialogue Models, we provide insights into how to represent the data in a speech dialogue model for better understanding and generation of speech. The choice of representation method directly affects the model's effectiveness in processing speech signals, system performance, and range of applications. The section covers two main types of representations: **semantic representations** and **acoustic representations**.

|              | Advantages of the comprehension side | Performance of unify music and audio | Compression rate of speech | Convert to historical context | Emotional and acoustic information | Pipeline for post-processing |
| ------------ | ------------------------------------ | ------------------------------------ | -------------------------- | ----------------------------- | ---------------------------------- | ---------------------------- |
| **Semantic** | Strong                               | Weak                                 | High                       | Easy                          | Less                               | Cascade                      |
| **Acoustic** | Weak                                 | Strong                               | Low                        | Difficult                     | More                               | End-to-end                   |

<div align='center'>
Table 2: The comparison of semantic and acoustic representations
</div>

## **Training Paradigm of Spoken Dialogue Model**

In the Training Paradigm of Spoken Dialogue Model section, we focuse on how to adapt text-based large language models (LLMs) into dialogue systems with speech processing capabilities. The **selection and design of training paradigms** have a direct impact on the **performance, real-time performance, and multimodal alignment** of the model.

<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/architecture1_new_00.png" alt="architecture1_new_00" style="zoom: 27%;" />  <img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/architecture3_00.png" alt="architecture3_00" style="zoom:22%;" />

Figure 5: Categorization Diagram of Spoken Dialogue Model Architectural Paradigms (above) and Diagram of Multi-stage Training Steps (below)
</div>

## Streaming, Duplex, and Interaction

The Streaming, Duplex, and Interaction section mainly discusses the implementation of **streaming processing, duplex communication, and interaction capabilities** inspeech dialogue models. These features are crucial for improving the response speed, naturalness, and interactivity of the model in real-time conversations.

<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/tu_03.png" alt="tu_03" style="zoom: 33%;" />

Figure 6: The Example Diagram of Duplex Interaction
</div>

## Training Resources and Evaluation

#### 1. Training resources

<div align='center'>
<table>
    <caption>Datasets used in the various training stages</caption>
    <thead>
        <tr>
            <th>Stage</th>
            <th>Task</th>
            <th>Dataset</th>
            <th>Size</th>
            <th>URL</th>
            <th>Modality</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="13"><b>Modal Alignment</b></td>
            <td>Mandarin ASR</td>
            <td>AISHELL-1</td>
            <td>170 hrs</td>
            <td><a href="https://www.openslr.org/33/">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>Mandarin ASR</td>
            <td>AISHELL-2</td>
            <td>1k hrs</td>
            <td><a href="https://github.com/kaldi-asr/kaldi/tree/master/egs/aishell2">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>Mandarin TTS</td>
            <td>AISHELL-3</td>
            <td>85 hrs, 88,035 utt., 218 spk.</td>
            <td><a href="https://www.aishelltech.com/aishell_3">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>TTS</td>
            <td>LibriTTS</td>
            <td>585 hrs</td>
            <td><a href="https://www.openslr.org/60/">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>TED-LIUM</td>
            <td>452 hrs</td>
            <td><a href="https://lium.univ-lemans.fr/ted-lium3/">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>VoxPopuli</td>
            <td>1.8k hrs</td>
            <td><a href="https://github.com/facebookresearch/voxpopuli">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>Librispeech</td>
            <td>1,000 hrs</td>
            <td><a href="https://www.openslr.org/12">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>MLS</td>
            <td>44.5k hrs</td>
            <td><a href="https://www.openslr.org/">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>TTS</td>
            <td>Wenetspeech</td>
            <td>22.4k hrs</td>
            <td><a href="https://wenet.org.cn/WenetSpeech/">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>Gigaspeech</td>
            <td>40k hrs</td>
            <td><a href="https://github.com/SpeechColab/GigaSpeech">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>VCTK</td>
            <td>300 hrs</td>
            <td><a href="https://paperswithcode.com/dataset/voice-bank-demand">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>TTS</td>
            <td>LJSpeech</td>
            <td>24 hrs</td>
            <td><a href="https://keithito.com/LJ-Speech-Dataset/">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>ASR</td>
            <td>Common Voice</td>
            <td>2,500 hrs</td>
            <td><a href="https://commonvoice.mozilla.org/zh-CN">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td rowspan="7"><b>Dual-Stream Processing</b></td>
            <td>Instruction</td>
            <td>Alpaca</td>
            <td>52,000 items</td>
            <td><a href="https://huggingface.co/datasets/tatsu-lab/alpaca">Link</a></td>
            <td>Text + TTS</td>
        </tr>
        <tr>
            <td>Instruction</td>
            <td>Moss</td>
            <td>-</td>
            <td><a href="https://huggingface.co/fnlp/moss-moon-003-sft">Link</a></td>
            <td>Text + TTS</td>
        </tr>
        <tr>
            <td>Instruction</td>
            <td>BelleCN</td>
            <td>-</td>
            <td><a href="https://github.com/LianjiaTech/BELLE/tree/main">Link</a></td>
            <td>Text + TTS</td>
        </tr>
        <tr>
            <td>Dialogue</td>
            <td>UltraChat</td>
            <td>1.5 million</td>
            <td><a href="https://github.com/thunlp/UltraChat">Link</a></td>
            <td>Text + TTS</td>
        </tr>
        <tr>
            <td>Instruction</td>
            <td>Open-Orca</td>
            <td>-</td>
            <td><a href="https://huggingface.co/datasets/Open-Orca/OpenOrca">Link</a></td>
            <td>Text + TTS</td>
        </tr>
        <tr>
            <td>Noise</td>
            <td>DNS</td>
            <td>2425 hrs</td>
            <td><a href="https://github.com/microsoft/DNS-Challenge">Link</a></td>
            <td>Noise data</td>
        </tr>
        <tr>
            <td>Noise</td>
            <td>MUSAN</td>
            <td>-</td>
            <td><a href="https://www.openslr.org/17/">Link</a></td>
            <td>Noise data</td>
        </tr>
        <tr>
            <td rowspan="4"><b>Conversation Fine-Tune</b></td>
            <td>Dialogue</td>
            <td>Fisher</td>
            <td>964 hrs</td>
            <td><a href="https://catalog.ldc.upenn.edu/LDC2004T19">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>Dialogue</td>
            <td>GPT-Talker</td>
            <td>-</td>
            <td><a href="https://github.com/AI-S2-Lab/GPT-Talker">Link</a></td>
            <td>Text, Speech</td>
        </tr>
        <tr>
            <td>Instruction</td>
            <td>INSTRUCTS2S-200K</td>
            <td>200k items</td>
            <td><a href="https://github.com/ictnlp/LLaMA-Omni">Link</a></td>
            <td>Text + TTS</td>
        </tr>
        <tr>
            <td>Instruction</td>
            <td>Open Hermes</td>
            <td>900k items</td>
            <td><a href="https://ollama.com/library/openhermes">Link</a></td>
            <td>Text + TTS</td>
        </tr>
    </tbody>
</table>

Table 3: Datasets used in the various training stages
</div>

<div align='center'>
<table>
<caption>Music and Non-Speech Sound Datasets</caption>
<thead>
    <tr>
    <th>Dataset</th>
    <th>Size</th>
    <th>URL</th>
    <th>Modality</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>ESC-50</td>
    <td>2,000 clips (5s each)</td>
    <td><a href="https://github.com/karoldvl/ESC-50">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>UrbanSound8K</td>
    <td>8,732 clips (<=4s each)</td>
    <td><a href="https://urbansounddataset.weebly.com/urbansound8k.html">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>AudioSet</td>
    <td>2000k+ clips (10s each)</td>
    <td><a href="https://research.google.com/audioset/">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>TUT Acoustic Scenes 2017</td>
    <td>52,630 segments</td>
    <td><a href="https://zenodo.org/record/400515">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>Warblr</td>
    <td>10,000 clips</td>
    <td><a href="https://warblr.net/">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>FSD50K</td>
    <td>51,197 clips (total 108.3 hours)</td>
    <td><a href="https://zenodo.org/record/4060432">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>DCASE Challenge</td>
    <td>varies annually</td>
    <td><a href="http://dcase.community/">Link</a></td>
    <td>Sound</td>
    </tr>
    <tr>
    <td>IRMAS</td>
    <td>6,705 audio files (3s each)</td>
    <td><a href="https://www.upf.edu/web/mtg/irmas">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>FMA</td>
    <td>106,574 tracks</td>
    <td><a href="https://github.com/mdeff/fma">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>NSynth</td>
    <td>305,979 notes</td>
    <td><a href="https://magenta.tensorflow.org/datasets/nsynth">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>EMOMusic</td>
    <td>744 songs</td>
    <td><a href="https://cvml.unige.ch/databases/emoMusic/">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>MedleyDB</td>
    <td>122 multitrack recordings</td>
    <td><a href="https://medleydb.weebly.com/">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>MagnaTagATune</td>
    <td>25,863 clips (30s each)</td>
    <td><a href="https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>MUSDB</td>
    <td>150 songs</td>
    <td><a href="https://paperswithcode.com/dataset/musdb18">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>M4Singer</td>
    <td>700 songs</td>
    <td><a href="https://github.com/M4Singer/M4Singer">Link</a></td>
    <td>Music</td>
    </tr>
    <tr>
    <td>Jamendo</td>
    <td>600k songs</td>
    <td><a href="https://www.jamendo.com/?language=en">Link</a></td>
    <td>Music</td>
    </tr>
</tbody>
</table>

Table 4: Music and Non-Speech Sound Datasets
</div>

#### 2. Evaluation

Evaluation is a crucial aspect of training and testing spoken dialogue models. In this section, we provide a comprehensive overview of the evaluation from **11 aspects**. The evaluation metrics are categorized into **two main types**: **Basic Evaluation**, and **Advanced Evaluation**.
<div align='center'>
<img src="https://cdn.jsdelivr.net/gh/MYJOKERML/imgbed/matebook14/image-20241111195959293.png" alt="image-20241111195959293" style="zoom:18%;" />

Table 5: This table evaluates model performance across various abilities, common tasks, representative benchmarks, and corresponding metrics.
</div>

## Cite

```bibtex
@article{WavChat,
  title={WavChat: A Survey of Spoken Dialogue Models},
  ...,
}
```
