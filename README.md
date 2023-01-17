# Word weight system baseed on trained attentions

I demonstrate construction of a word weight system by using attentions that are trained with hierarchical attention network (HAN). I use Financial Phrase Bank provided by Malo and Korhonen et al (2014) to demonstrate our proposed method. Readers may refer to the link below. 
https://huggingface.co/datasets/financial_phrasebank

The main proces toward construction of the word weight system is as follows: <br />
(1) word2vec training <br />
(2) HAN training <br />
(3) collection of trained attentions at a word level 
Note that the provided code is designed for regression analysis, but readers may change the loss function and apply it for classification problem. The resulting word importance weights highlight key words that attract high attentions of HAN. The top 10 words are as follows: 

word        excess_attention
net         0.390077
profit      0.259981
sales       0.185049
operating   0.169306
eur         0.153330
loss        0.104787
up          0.087095
from        0.076385
quarter     0.057741
euro        0.051478

Malo P., Sinha A., Korhonen P., Wallenius P., Takala P. (2014) 
Good debt or bad debt: Detecting semantic orientations in economic texts,
*Journal of the Association for Information Science and Technology* 
**65**
