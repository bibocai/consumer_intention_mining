#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
import name_convert as nc
import title_to_tree as tot
import predict as pt

segmentor = Segmentor()
segmentor.load_with_lexicon("./system/ltp_data/cws.model","./system/ltp_data/plain.txt")

postagger = Postagger()
postagger.load_with_lexicon("./system/ltp_data/pos.model","./system/ltp_data/postagger.txt")

parser = Parser()
parser.load("./system/ltp_data/parser.model")

def get_result(company_name,news_titles):
	title_tree = []
	for sentence in news_titles:
		words = segmentor.segment(sentence)
		words = nc.convert(words)
		if company_name not in words:
			add_word = [company_name, u':'.encode('utf8')]
			add_word.extend(words)
			words = add_word
		# print ("\t".join(words))
		postags = postagger.postag(words)
		# print ("\t".join(postags))
		arcs = parser.parse(words,postags)
		# print ("\t".join("%d:%s" % (arc.head,arc.relation) for arc in arcs))
		tree = tot.convert(company_name,words,postags,arcs)
		# print(tree)
		if tree != '###':
			title_tree.append([words,tree])
	if not title_tree:
		return '###'
	else:
		pre,con = pt.predict(title_tree)
		return pre,con



if __name__ == '__main__':

	company_name = "新湖中宝"
	sentence = ["中国进出口银行与新湖中宝加强合作"]
	get_result(company_name,sentence)
	segmentor.release()
	postagger.release()
	parser.release()
