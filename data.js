const BDH_DATA = {
  "version": "test",
  "model": "BDH",
  "total_params": 25296896,
  "num_inputs": 1,
  "inputs": [{
    "text": "The cat sat",
    "tokens": [84,104,101,32,99,97,116,32,115,97,116],
    "token_count": 11,
    "captures": [
      {"module_path":"embed","type":"Embedding","output":{"name":"embed_out","shape":[1,11,256],"numel":2816,"mean":0,"std":0.02,"min":-0.06,"max":0.06,"sparsity":0.0,"sample":[0.1,-0.2,0.3,-0.1,0.2,0.0,-0.3,0.1,0.4,-0.2]},"input":{"name":"embed_in","shape":[1,11],"numel":11,"mean":90,"std":30,"min":32,"max":116,"sparsity":0.0,"sample":[84,104,101,32,99,97,116,32,115,97]}},
      {"module_path":"ln","type":"LayerNorm","output":{"name":"ln_out","shape":[1,1,11,256],"numel":2816,"mean":0,"std":0.99,"min":-3.1,"max":3.3,"sparsity":0.0,"sample":[0.5,-0.3,0.2,0.1,-0.4,0.3,-0.2,0.6,-0.1,0.4]},"input":{"name":"ln_in","shape":[1,1,11,256],"numel":2816,"mean":0,"std":0.02,"min":-0.06,"max":0.06,"sparsity":0.0,"sample":[0.1,-0.2,0.3,-0.1,0.2,0.0,-0.3,0.1,0.4,-0.2]}},
      {"module_path":"attn","type":"Attention","output":{"name":"attn_out","shape":[1,4,11,256],"numel":11264,"mean":0,"std":309,"min":-1403,"max":1314,"sparsity":0.09,"sample":[0,0,0,0,0,0,0,0,0,0]}},
      {"module_path":"drop","type":"Dropout","output":{"name":"drop_out","shape":[1,4,11,8192],"numel":360448,"mean":0.01,"std":0.05,"min":0,"max":1.2,"sparsity":0.77,"sample":[0,0,0,0,0,0,0,0,0,0]},"input":{"name":"drop_in","shape":[1,4,11,8192],"numel":360448,"mean":0.01,"std":0.05,"min":0,"max":1.2,"sparsity":0.77,"sample":[0,0,0,0,0,0,0,0,0,0]}}
    ],
    "final_output": {"name":"final","shape":[1,11,256],"numel":2816,"mean":0.008,"std":0.33,"min":-0.96,"max":0.99,"sparsity":0.0,"sample":[0.2,-0.1,0.3,-0.2,0.1,0.4,-0.3,0.0,-0.5,0.2]}
  }]
}; 
