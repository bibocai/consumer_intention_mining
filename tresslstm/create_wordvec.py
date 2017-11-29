def create_wordvec():
	word = {}
	fr = open('title_word_vector','r')
	for line in fr.readlines():
		l = line.strip().split('\t')
		w = l[0].strip()
		vec = l[1].strip().split(' ')
		word[w] = [float(v) for v in vec]
	fr.close()
	return word

if __name__ == '__main__':
	create_wordvec()
