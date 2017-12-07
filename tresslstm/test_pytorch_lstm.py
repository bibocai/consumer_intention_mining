#!/usr/bin/env python
# -*- coding: utf-8 -*-

from treelstm_with_pytorch import TreeLstmCell
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from  torch.autograd import Variable
import numpy as np
input_size = 10
hidden_size =10

embedding_dic={'专门':[1,1,1,1,1,1,1,1,1,1]}
class TreeNode(object):
    def __init__(self,k=None,l=None,r=None):
        self.key = k
        self.left = l
        self.right = r

def create_tree(s):
    #print(s)
    s = s.strip()[2:-2].split(' ')
    #print(s)
    root = TreeNode()
    if len(s) == 2:
        root.key = s[1].strip()
        return root
    root.key = s[0].strip()
    flag = 1
    for i in range(2,len(s)):
        if s[i] == '(':
            flag += 1
        if s[i] == ')':
            flag -= 1
        if flag == 0:
            root.left = create_tree(' '.join(s[1:i+1]))
            root.right = create_tree(' '.join(s[i+1:]))
            break
    return root


def show_tree(s_tree,_iter_for_dispaly):
    if not s_tree:
        return
    print '\t'*_iter_for_dispaly,

    _iter_for_dispaly+=1
    print(s_tree.key)
    show_tree(s_tree.left,_iter_for_dispaly)
    show_tree(s_tree.right,_iter_for_dispaly)
    return

class TreeLstm(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(TreeLstm,self).__init__()
        self.Cell=TreeLstmCell(input_size,hidden_size)

    def forward(self,sentence,root):
        self.sentence=sentence
        h,c=self.create_graph(root)
        return h
    def create_graph(self,root):
        if root.key.isdigit():
            if self.sentence[int(root.key)] in embedding_dic:
                input = np.array(embedding_dic[self.sentence[int(root.key)]]).astype(np.float32)
            else:
                input = np.random.randn(input_size).astype(np.float32)
            hlx = Variable(torch.zeros(hidden_size))
            clx = Variable(torch.zeros(hidden_size))
            hrx = Variable(torch.zeros(hidden_size))
            crx = Variable(torch.zeros(hidden_size))
            input = Variable(torch.from_numpy(input))
            return self.Cell(input,hlx,clx,hrx,crx)
        hlx,clx = self.create_graph(root.left)
        hrx,crx = self.create_graph(root.right)
        input = Variable(torch.zeros(input_size))
        return self.Cell(input,hlx,clx,hrx,crx)

if __name__ == '__main__':
   # s = "( ATT ( ni 0 ) ( SBV ( ATT ( ATT ( v 1 ) ( n 2 ) ) ( ATT ( n 3 ) ( COO ( LAD ( c 5 ) ( n 6 ) ) ( n 4 ) ) ) ) ( ADV ( POB ( ATT ( d 8 ) ( ATT ( v 9 ) ( n 10 ) ) ) ( d 7 ) ) ( VOB ( n 14 ) ( RAD ( u 13 ) ( VOB ( n 12 ) ( v 11 ) ) ) ) ) ) )"
    s='( VOB ( n 9 ) ( VOB ( SBV ( r 4 ) ( ADV ( v 5 ) ( ADV ( ADV ( d 6 ) ( v 7 ) ) ( v 8 ) ) ) ) ( COO ( ADV ( v 2 ) ( v 3 ) ) ( ADV ( d 0 ) ( v 1 ) ) ) ) )'
    s_tree = create_tree(s)
    show_tree(s_tree,0)

    model=TreeLstm(10,10)
    h=model(['专门','打电话', '来', '问', '我', '要', '不','要' ,'买', '手机'],s_tree)
    print h
