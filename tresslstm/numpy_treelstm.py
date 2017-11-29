#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cPickle as cp

f = open('parameter','rb')
w_ih = cp.load(f)
w_hh = cp.load(f)
b_ih = cp.load(f)
b_hh = cp.load(f)

wl_ih = cp.load(f)
wl_hh = cp.load(f)
bl_ih = cp.load(f)
bl_hh = cp.load(f)

wr_ih = cp.load(f)
wr_hh = cp.load(f)
br_ih = cp.load(f)
br_hh = cp.load(f)

w = cp.load(f)
b = cp.load(f)
f.close()

def sigmoid(vec):
	return 1.0/(1.0+np.exp(-vec))

def Cell(input,hlx,clx,hrx,crx):
	hx = hlx+hrx
	g = np.dot(w_ih,input)+b_ih+np.dot(w_hh,hx)+b_hh
	gl = np.dot(wl_ih,input)+bl_ih+np.dot(wl_hh,hlx)+bl_hh
	gr = np.dot(wr_ih,input)+br_ih+np.dot(wr_hh,hrx)+br_hh

	i,c,o = np.split(g,3)

	ingate = sigmoid(i)
	cellgate = np.tanh(c)
	outgate = sigmoid(o)
	forgetgatel = sigmoid(gl)
	forgetgater = sigmoid(gr)

	cy = (ingate*cellgate) + (forgetgatel*clx) + (forgetgater*crx)
	hy = outgate*np.tanh(cy)

	return hy,cy

if __name__ == '__main__':
	print(w_ih)
	print(w_hh)
	print(b_ih)
	print(b_hh)
	print(wl_ih)
	print(wl_hh)
	print(bl_ih)
	print(bl_hh)
	print(wr_ih)
	print(wr_hh)
	print(br_ih)
	print(br_hh)
	print(w)
	print(b)











