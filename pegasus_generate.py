# coding:utf-8

import torch
from transformers import BertTokenizer, PegasusForConditionalGeneration, Text2TextGenerationPipeline, PegasusConfig
import numpy as np
import summarize_4zh


# 获取参数量
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


text = "四海网讯,近日,有媒体报道称:章子怡真怀孕了!报道还援引知情人士消息称,“章子怡怀孕大概四五个月,预产期是年底前后,现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息,23日晚8时30分," \
       "华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士,这位人士向华西都市报记者证实说:“子怡这次确实怀孕了。她已经36岁了,也该怀孕了。章子怡怀上汪峰的孩子后,子怡的父母亲十分高兴。子怡的母亲," \
       "已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时,华西都市报记者为了求证章子怡怀孕消息,又电话联系章子怡的亲哥哥章子男,但电话通了," \
       "一直没有人接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来,就被传N遍了!不过,时间跨入2015年,事情却发生着微妙的变化。2015年3月21日,章子怡担任制片人的电影《从天儿降》开机," \
       "在开机发布会上几张合影,让网友又燃起了好奇心:“章子怡真的怀孕了吗?”但后据证实,章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日,《太平轮》新一轮宣传,章子怡又被发现状态不佳,不时深呼吸," \
       "不自觉想捂住肚子,又觉得不妥。然后在8月的一天,章子怡和朋友吃饭,在酒店门口被风行工作室拍到了,疑似有孕在身!今年7月11日,汪峰本来在上海要举行演唱会,后来因为台风“灿鸿”取消了。而消息人士称," \
       "汪峰原来打算在演唱会上当着章子怡的面宣布重大消息,而且章子怡已经赴上海准备参加演唱会了,怎知遇到台风,只好延期,相信9月26日的演唱会应该还会有惊喜大白天下吧。 "

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('uer/pegasus-base-chinese-cluecorpussmall')

config = PegasusConfig()
# config.activation_dropout = 0.1
config.activation_function = 'relu'
config.d_model = 384
config.decoder_attention_heads = 6
config.decoder_ffn_dim = 1536
config.decoder_start_token_id = 101
config.decoder_layers = 6
config.dropout = 0.0  # 关闭dropout，保证每次预测结果相同
config.encoder_attention_heads = 6
config.encoder_ffn_dim = 1536
config.encoder_layers = 6
config.forced_eos_token_id = 102
config.scale_embedding = True
config.vocab_size = 21128

model = PegasusForConditionalGeneration(config)
model.load_state_dict(torch.load('pegasus_summary_6666.pt', map_location=device))
# model = PegasusForConditionalGeneration.from_pretraimodel = PegasusForConditionalGeneration.from_pretrained('uer/pegasus-base-chinese-cluecorpussmall',
# #                                                         # d_model=768,
# #                                                         # decoder_attention_heads=12,
# #                                                         # decoder_ffn_dim=1536,
# #                                                         decoder_layers=12,
# #                                                         # encoder_attention_heads=12,
# #                                                         # encoder_ffn_dim=1536,
# #                                                         encoder_layers=12)ned('uer/pegasus-base-chinese-cluecorpussmall')

model.to(device)
config = model.config
# print(config)
# print(get_parameter_number(model))


def generate_title(text, num_beam=3):
    ids = tokenizer.encode(text, return_tensors='pt')
    ids = torch.LongTensor(ids).to(device)

    outputs = model.generate(ids,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             eos_token_id=tokenizer.sep_token_id,
                             top_k=1,
                             max_length=100,
                             num_beams=num_beam,
                             no_repeat_ngram_size=2,
                             num_return_sequences=num_beam,
                             return_dict_in_generate=True,
                             output_scores=True)  # .cpu().numpy()


    result = []
    delete_ids = np.array(
        [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id])
    for seq in outputs.sequences:
        seq = np.setdiff1d(seq.cpu(), delete_ids, True)  # 删除[CLS] [SEP] [PAD]
        result.append(''.join(tokenizer.decode(seq)).replace(' ', ''))  # 解码
    return result, outputs.sequences_scores.tolist()


# 加载完生成式模型要先运行一遍，因为第一遍的运行时间会比较长
generate_title(text[:1025])
# while True:
#     text = input('---')
#     res_textrank = summarize_4zh.textrank_top(text, 10, 2)
#     text = ''
#     for r in res_textrank:
#         text += r['sentence']
#     print(text)
#     print(len(text))
#     results = generate_title(text[:1025])  # 输入长度限制
#     for r in results:
#         print(r)
