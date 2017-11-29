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

def show_tree(s_tree):
	if not s_tree:
		return
	print(s_tree.key)
	show_tree(s_tree.left)
	show_tree(s_tree.right)
	return

if __name__ == '__main__':
	s = "( ATT ( ni 0 ) ( SBV ( ATT ( ATT ( v 1 ) ( n 2 ) ) ( ATT ( n 3 ) ( COO ( LAD ( c 5 ) ( n 6 ) ) ( n 4 ) ) ) ) ( ADV ( POB ( ATT ( d 8 ) ( ATT ( v 9 ) ( n 10 ) ) ) ( d 7 ) ) ( VOB ( n 14 ) ( RAD ( u 13 ) ( VOB ( n 12 ) ( v 11 ) ) ) ) ) ) )"
	s_tree = create_tree(s)
	show_tree(s_tree)
