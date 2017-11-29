#!/usr/bin/env python
# -*- coding: utf-8 -*-
def convert_tree(dep_list, target_idx, target_pos):
	v = "( " + target_pos + " " + str(target_idx) + " )"
	for i in range(len(dep_list)):
		if dep_list[i][3] == -1: #used edge
			continue
		mod_idx = i
		mod_pos = dep_list[i][1]
		head_idx = int(dep_list[i][2])
		head_pos = dep_list[head_idx][1]
		rel = dep_list[i][3]

		if mod_idx == target_idx:
			dep_list[i][3] = -1
			v = "( " + rel + " " + convert_tree(dep_list, mod_idx, mod_pos) + " " + convert_tree(dep_list, head_idx, head_pos) + " )"

	for i in range(len(dep_list)):
		if i == target_idx:
			break
		if dep_list[i][3] == -1:
			continue
		mod_idx = i
		mod_pos = dep_list[i][1]
		head_idx = int(dep_list[i][2])
		head_pos = dep_list[head_idx][1]
		rel = dep_list[i][3]

		if head_idx == target_idx:
			dep_list[i][3] = -1
			v = "( " + rel + " " + convert_tree(dep_list, mod_idx, mod_pos) + " " + convert_tree(dep_list, head_idx, head_pos) + " )"

	for i in reversed(range(len(dep_list))):
		if i == target_idx:
			break
		if dep_list[i][3] == -1:
			continue
		mod_idx = i
		mod_pos = dep_list[i][1]
		head_idx = int(dep_list[i][2])
		head_pos = dep_list[head_idx][1]
		rel = dep_list[i][3]

		if head_idx == target_idx:
			dep_list[i][3] = -1
			v = "( " + rel + " " + convert_tree(dep_list, mod_idx, mod_pos) + " " + convert_tree(dep_list, head_idx, head_pos) + " )"

	return v

def convert(company_name,words,postags,arcs):
	arc_head = []
	arc_relation = []
	for arc in arcs:
		head = arc.head - 1
		arc_head.append(head)
		arc_relation.append(arc.relation)
	dep_parsed_sent_list = zip(list(words),list(postags),arc_head,arc_relation)
	#print(dep_parsed_sent_list)
	dep_parsed_sent_list = [list(d) for d in dep_parsed_sent_list]
	#print(dep_parsed_sent_list)

	#find the target
	target_idx = -1
	target_pos = ""
	for i,dep in enumerate(dep_parsed_sent_list):
		if company_name == dep[0]:
			#print (dep[0])
			target_idx = i
			target_pos = dep[1]
		if dep[2] == -1:
			dep[3] = -1
	if target_idx == -1:
		return '###'

	#build tree according to the target
	target_tree = convert_tree(dep_parsed_sent_list, target_idx, target_pos)
	#print('dep_tree:',target_tree)

	#check all the dep have been used
	for dep in dep_parsed_sent_list:
		if dep[3] != -1:
			return '###'

	return target_tree


if __name__ == "__main__":
	main()
