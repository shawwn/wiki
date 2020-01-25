---
url: https://medium.com/@NPCollapse/addendum-evaluation-of-my-model-e6734b51a830
title: Addendum: Evaluation of My Model
author: Connor Leahy
date: Jun 12, 2019
doi:
...
As a mercifully short addendum, I’d like to quickly address a few questions about my model. Please read my update post to hear my important updated beliefs on this situation, because I believe the details of how powerful my model is or not are not actually very important to the overall situation.

As described in my technical post, my model is not identical to OpenAI’s, because I simply didn’t have all the details of what they did. The truth is also that the samples and metrics I have shown aren’t 100% accurate. For one, my metric code is flawed, I made several rookie mistakes in setting up accurate evaluation (let train and eval data mix, used metrics whose math I didn’t understand etc), and the model I used to generate the samples is in fact not the final trained model, but one about halfway through the training. I didn’t take my time to evaluate the strength of my model, I simply saw I had the same amount of hardware as OpenAI and code as close to the paper as possible and went with it. The reason for this is a simple human flaw: I got cold feet once I realized what I was sitting on and acted rashly. I made a mistake, I did something stupid, that’s all there is to it.

Thanks to help from OpenAI it is now safe to say that my model is not as powerful as OpenAI’s. The metric results for WikiText2, LAMBADA and PTB are (lower is better):

GPT2: 18.67 / 8.63 / 36.51
Mine: 43.79 / 109.47 / 202.29

Although I used the same amount of hardware (or more), the differences in my training setup and hyperparameters made a significant difference. Which is an unfortunate reality to anyone familiar with reproducing deep learning papers. I don’t think my model in its current state is even as dangerous as 117M in its text generating abilities. But I believe to have found the quirks in my setup that have held the model back, and they are easy to fix. I am very tempted to continue tinkering with the model and seeing if I can improve it…but I will be holding back for now.
---
url: https://medium.com/@NPCollapse/replicating-gpt2-1-5b-86454a7f26af
title: Replicating GPT2–1.5B
author: Connor Leahy
date: Jun 6, 2019
doi: 
...
In this post, I want to quickly talk about the technical and organizational questions around my recent replication of GPT2–1.5B. Please read my main post for the full story. I will try to keep this post brief.

**The important facts**

Code: [https://github.com/ConnorJL/GPT2](https://github.com/ConnorJL/GPT2)
Samples: [https://github.com/ConnorJL/GPT2/tree/master/samples](https://github.com/ConnorJL/GPT2/tree/master/samples)

The code should run out of the box on GPUs and TPUs (and CPUs, if you’re really desperate). I used the parameters specified in 1.5B.json and trained it on a preemptible v3–512 TPU pod (which is actually more powerful than the machine OpenAI used) for around a week (with interruptions). Code and instructions for generating the dataset are also included in the repo.

You can download my models with the script in the repo. Currently I have a weaker version of 117M, and a model I call PrettyBig which is slightly larger than OpenAI’s 345M, which means it is technically the largest GPT2 model currently publicly available.

I will be releasing 1.5B to the public on July 1st, if, and only if, no one shows me a convincing reason not to. When I do, it will be downloadable just like my other models.
