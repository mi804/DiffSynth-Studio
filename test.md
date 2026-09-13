Tongyi-MAI/Z-Image:transformer/*.safetensors:;Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors:xxx;bitsandbytes_nf4/qkv;Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors:bitsandbytes_nf4;

我的量化框架已经好了，我想为他写一个modelscope的notebook，即为此写一个在线使用示例；我想的故事是这样的：首先，简单介绍量化框架；然后，以Minimax-H3为例讲解为什么需要量化，即模型很大，diffsyn
   th里只能通过vram
   offload来运行（加运行例子代码：t2va），然后，可以教如何通过在线量化降低模型大小：先给一个torchao的w8a16的量化；然后再使用nf4的量化，发现nf4量化效果不够；然后通过计算量化误差，这个地方，你可以直接写代码看模型量化层的权重误差，甚至也可以直接把这个代码做成一个函数，直接放到quantconfig里作为一个接口，供用户分析量化误差（如果你写了这个接口，最后告诉我一下，新增了什么代码）；得到误差后，我们nf4使
   用的exlude_modules，借此引出我们的quantconfig配置介绍，然后成功Nf4量化。 最后再加个进阶节：混合量化，用8位和4位量化，平衡误差。
