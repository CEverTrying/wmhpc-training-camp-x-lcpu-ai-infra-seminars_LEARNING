最终修复总结
根因链条(VSCode/cpptools 侧):
1. 缺 CUDA 头文件 → 装了 cuda-cudart-dev-13-1 + cuda-crt-13-1(后者是 CUDA 13 打包变更后 crt/ 头文件的专用包)
2. 缺 nvcc → cpptools 的 CUDA IntelliSense 硬性依赖 → 装了 cuda-nvcc-13-1
3. cpptools 原生代码靠 PATH 找 nvcc(不认 compilerPath 设置)→ sudo ln -s /usr/local/cuda/bin/nvcc /usr/local/bin/nvcc 放到全局 PATH,一步解决
4. includePath:用户级 settings + workspace 级 assignment01/.vscode/c_cpp_properties.json,覆盖 CUDA 头文件和 common.h