#-*- coding:utf-8 -*-
fr = open('pattern.txt','r')
dic = {}
for line in fr.readlines():
	flag = line.strip().split('|')
	flag = [i.strip() for i in flag]
	l = len(flag)
	for i in range(1,l):
		dic[flag[i]] = flag[0]
fr.close()

def convert(s):
	count = 0
	while 1:
		item = s[count] + ' ' + s[count+1]
		if item in dic:
			s[count] = dic[item]
			del s[count+1]
		count += 1
		if count >= len(s)-1:
			break
	for i,v in enumerate(s):
		if v in dic:
			s[i] = dic[v]
	return s


if __name__ == '__main__':
	s = ['平安','银行','与','中行','加强','合作']
	new_s = convert(s)
	print(new_s)

