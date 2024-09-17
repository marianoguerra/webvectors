
download-deps:
    mkdir -p Xenova/all-MiniLM-L6-v2/onnx/
    wget https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx?download=true -O Xenova/all-MiniLM-L6-v2/onnx/model_quantized.onnx

    wget https://cdn.jsdelivr.net/npm/echarts-gl/dist/echarts-gl.min.js -O echarts-gl.js
    wget https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js -O echarts.js
    wget https://raw.githubusercontent.com/PAIR-code/umap-js/main/lib/umap-js.min.js -O umap-js.js
    wget https://raw.githubusercontent.com/karpathy/tsnejs/master/tsne.js
    wget https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2 -O transformers.js
