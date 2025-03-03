<h1 align="center">LOVO</h1>

<p align="center"><a href="#data">🎞️ Data Preparation</a> - <a href="#setup">🛠️ Setup</a> - <a href="#examples">🤸 Examples</a>  <br>  - <a href="#acknowledgement">👏 Acknowledgment</a> </p>

LOVO is a novel system that enables 🔥 ***complex object query*** 🔥 in the large-scale video datasets. This code demonstrates the core operations of the LOVO system in a research-grade implementation (i.e., not suitable for production use), featuring an orthogonal module design for flexibility and maintainability. Future updates will include a Docker-based version as well as visualization for enhanced reproducibility.

<a id="data"></a>
## 🎞️ Data Preparation

1. Download the public video datasets and extract it to `dataset/`

2. Apply key frame extraction to the videos and put all the frame to csv format (ID, frame path) just like ``dataset/cityscapes/stuggart_overall.csv`` prepared for processing


<a id="setup"></a>
## 🛠️ Setup

1. Install the dependencies

    1. Install PyTorch and Transformer
    2. Install [Milvus](https://github.com/milvus-io/milvus/blob/master/DEVELOPMENT.md#building-milvus-on-a-local-osshell-environment) vector database
    3. Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) for the accurate rerank
    4. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
    5. Install NVIDIA TensorRT

2. Install the LOVO packages.

    ```bash
    git clone https://github.com/Aph-xin/complex-object-query-in-large-scale-datasets

    cd complex-object-query-in-large-scale-datasets

    conda lovo create -f environment.yml

    conda activate lovo

    pip install -r requirements.txt
    
    python3 setup.py develop --user
    ```

3. (optional)Build the TensorRT engine for the vision encoder

    ```bash
    mkdir -p model
    cd src
    python3 -m encoder.build_image_encoder_engine \
        model/owl_image_encoder_patch32.engine
    ```

4. Build Milvus on a local OS/shell environment

    ```bash
   cd etcd
    ./etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

    sudo ./minio server /home/data --console-address ":9001"

    sudo ./milvus/milvus/bin/milvus run standalone
    ```


5. Run an example object query

    ```bash
    cd src
    python3 main.py \
        --prompt="a person carrying a black backpack wearing blue jeans white shirt, walking on the crosswalk" \
        --threshold=0.1 \
        --dataset="cityscapes" \
        --csv-path="../dataset/cityscapes/stuggart_overall.csv" \
        --image_encoder_engine=../model/owl_image_encoder_patch32.engine
    ```

If everything is working properly, you should see a visualization saved to ``results`` folder.  

<a id="examples"></a>
## 🤸 Examples

### Example - Basic object query


The example ``example/demo.ipynb`` demonstrates how to use the LOVO to
querying objects in video datasets, and the output as well as its log are shown in ``results`` and ``logs`` folders.

Furthermore, to run object query on the single frame with text descriptions of their labels, first navigate to the test folder

```bash
cd test
```

Then run the example

```bash
python3 test_owl_rerank.py
```

By default the output will be saved to ``assets/cityscapes_out.jpg``. 


<a id="acknowledgement"></a>
## 👏 Acknowledgement

Thanks to the authors of [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), and [NanoOWL](https://github.com/NVIDIA-AI-IOT/nanoowl) for the great work about complex object query.
