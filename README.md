# Fine-Tuning a Adapter for Any Embedding Model

## 概要
埋め込みモデル(text-embedding-ada-002)のベクトルをデータセットに最適化するために、CNN-baseのAdapterを学習することで検索精度の改善を検証。  
検証内容は、`FineTuning.ipynb`を参照。公開コードのため、実際の検証結果は含めていない。  
**結果は、改善は見られず。**  
モデル構造、Loss関数の設計、等の追加検証が必要。


## 検証方法
- 埋め込みベクトルは全て text-embedding-ada-002 で作成
- Queryの埋め込みベクトルと正解チャンクの埋め込みベクトルを近づけるように学習
- Adapterは軽量な1D-ConvAutoEncoderモデルを利用(NNモデルも`model.py`にあり。)
- Loss関数は MSE or Cosine、もしくは Weighted Loss で検証


## Reference
- [Fine-Tuning a Linear Adapter for Any Embedding Model](https://www.llamaindex.ai/blog/fine-tuning-a-linear-adapter-for-any-embedding-model-8dd0a142d383)
- [Finetuning an Adapter on Top of any Black-Box Embedding Model](https://docs.llamaindex.ai/en/stable/examples/finetuning/embeddings/finetune_embedding_adapter/#finetuning-an-adapter-on-top-of-any-black-box-embedding-model)
- [Llama-Adapter](https://huggingface.co/docs/peft/package_reference/llama_adapter)
- [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](https://arxiv.org/pdf/2303.16199)