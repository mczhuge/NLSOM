## Usage

* Install
```
conda create --name mindstorm python=3.8.5  
conda activate mindstorm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
pip install pandas==1.4.3
pip install git+https://github.com/huggingface/transformers.git #@add_blip2_ydshieh
pip install rouge==1.0.1
pip install sentence-transformers==2.2.2
pip install nltk==3.6.6
pip install evaluate==0.4.0
pip install rouge_score==0.1.2
pip install openai==0.27.0
pip install accelerate
pip install modelscope
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install librosa
pip install fairseq
python -c "from modelscope.pipelines import pipeline;print(pipeline('image-captioning')('https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg'))"
pip install setuptools==59.5.0
```

* Get TARA dataset
```
sh download_tara.sh
```

* Set OPENAI API
```
export OPENAI_API_KEY=${Your OpenAI API}
```

* Run
```
conda activate mindstorm
python run.py
```
