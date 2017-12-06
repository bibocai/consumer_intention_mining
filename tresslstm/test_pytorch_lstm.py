#!/usr/bin/env python
# -*- coding: utf-8 -*-

from treelstm_with_pytorch import TreeLstmCell
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

INPUT_SIZE = 100
HIDDEN_SIZE =200 
VOCAL_SIZE =
embedding_dic={}
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

Class TreeLstm(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(TreeLstm,self).__init__()
        self.Cell=TreeLstmCell(INPUT_SIZE,HIDDEN_SIZE)
        
    def forward(self,sentence,root):
        h,c=self.create_graph(root)
        return h
    def create_graph(self,sentence,root):
        if root.key.isdigit():
            if sentence[int(root.key)] in embedding_dic:
                input = np.array(embedding_dic[sentence[int(root.key)]])
            else:
                input = np.random.randn(input_size)
            hlx = Variable(torch.zeros(hidden_size))
            clx = Variable(torch.zeros(hidden_size))
            hrx = Variable(torch.zeros(hidden_size))
            crx = Variable(torch.zeros(hidden_size))
            input = Variable(torch.from_numpy(input))
            return nt.Cell(input,hlx,clx,hrx,crx)
        hlx,clx = create_graph(root.left)
        hrx,crx = create_graph(root.right)
        input = Variale(np.zeros(input_size))
        return nt.Cell(input,hlx,clx,hrx,crx)

if __name__ == '__main__':
    s = "( ATT ( ni 0 ) ( SBV ( ATT ( ATT ( v 1 ) ( n 2 ) ) ( ATT ( n 3 ) ( COO ( LAD ( c 5 ) ( n 6 ) ) ( n 4 ) ) ) ) ( ADV ( POB ( ATT ( d 8 ) ( ATT ( v 9 ) ( n 10 ) ) ) ( d 7 ) ) ( VOB ( n 14 ) ( RAD ( u 13 ) ( VOB ( n 12 ) ( v 11 ) ) ) ) ) ) )"
    s_tree = create_tree(s)
    show_tree(s_tree,0)

    ctree = create_tree(s_tree) 
    model=TreeLstm(10,10)
    h=model('',ctree)       
