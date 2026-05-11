**Introduction**\
Fine-tuning is essential for adapting models to downstream tasks. Traditional full fine-tuning requires updating all parameters, which becomes expensive as model sizes grow, motivating the development of parameter-efficient fine-tuning (PEFT) methods like Low-Rank Adaptation (LoRA), which take a small subset of parameters for adapting to downstream tasks [2]. While LoRA addressed this by injecting trainable low-rank matrices, it falls short of full fine-tuning performance due to its coupled treatment of weight magnitude and direction.

To solve this problem, DoRA: Weight-Decomposed Low-Rank Adaptation by Liu et al. was proposed [1]. DoRA decomposes pretrained weights into magnitude and direction components, tuning each independently. This decomposition allows DoRA to more closely mirror the learning behavior of full fine-tuning compared to LoRA with negligible additional overhead. This project reproduces the results of DoRA from Liu et al. 

**Chosen Result**\
**GitHub Contents**\
**Re-implementation Details**\
**Reproduction Steps**\
**Results/Insights**\
**Conclusion**\
**References**\
**Acknowledgements**\

