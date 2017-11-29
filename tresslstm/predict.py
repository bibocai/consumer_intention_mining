#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import create_tree as ct
# import create_wordvec as cw
import numpy_treelstm as nt
from gensim.models import word2vec

input_size = 200
hidden_size = 100
# word = cw.create_wordvec()
model = word2vec.Word2Vec.load('model')

def softmax(x):
	f = np.exp(x)
	return f/sum(f)

def create_graph(root):
	if root.key.isdigit():
		if title[int(root.key)] in model.vocab:
			input = np.array(model[title[int(root.key)]])
		else:
			input = np.random.randn(input_size)
		hlx = np.zeros(hidden_size)
		clx = np.zeros(hidden_size)
		hrx = np.zeros(hidden_size)
		crx = np.zeros(hidden_size)
		return nt.Cell(input,hlx,clx,hrx,crx)
	hlx,clx = create_graph(root.left)
	hrx,crx = create_graph(root.right)
	input = np.zeros(input_size)
	return nt.Cell(input,hlx,clx,hrx,crx)

def predict(title_tree):
	ho = np.zeros(hidden_size)
	for t in title_tree:
		global title
		title = t[0]
		tree = t[1]
		ctree = ct.create_tree(tree)
		h,c = create_graph(ctree)
		ho += h
	ho = ho/(len(title_tree))
	scores = np.dot(nt.w,ho)+nt.b
	sc = softmax(scores)
	sort = np.argmax(sc)
	confidence = np.max(sc)
	return sort,confidence

if __name__ == '__main__':
	predict()






