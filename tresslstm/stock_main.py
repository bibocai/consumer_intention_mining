# -*- coding:utf-8 -*-
import MySQLdb
import time,re,datetime
import numpy as np
import get_result as gr

db = MySQLdb.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='root',
    db='stocknews',
    init_command='set names utf8'
)
cur = db.cursor()


# sql_select5 = "SELECT price \
#                FROM stock_date \
#                WHERE date_time = %s"
#
# sql_update2 = "UPDATE stock_recommend \
# 			   SET recommend_stock = %s \
#                WHERE date_time = %s"
#
# sql_update3 = "UPDATE stock_recommend \
# 			   SET profit = %f,random_profit = %f \
#                WHERE date_time = %s"

def get_date():
	now_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
	year_match = int(re.match(r'(\d{4})-\d{2}-\d{2}', now_date).group(1))
	month_match = int(re.match(r'\d{4}-(\d{2})-\d{2}', now_date).group(1))
	day_match = int(re.match(r'\d{4}-\d{2}-(\d{2})', now_date).group(1))
	last_date = (datetime.datetime(year=year_match, month=month_match, day=day_match) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
	return now_date,last_date

def stock_predict(date_time):
	sql_select1 = "SELECT stock_company.company_name,stock_date.id \
	               FROM stock_company,stock_date \
	               WHERE stock_company.id = stock_date.company_id and stock_date.date_time = %s"
	cur.execute(sql_select1,date_time)
	select_result1 = cur.fetchall()
	#判断有今天这个日期
	if select_result1:
		for item in select_result1:
			company_name = item[0]
			date_id = item[1]
			# print company_name
			# print date_id
			sql_select2 = "SELECT news_title \
						   FROM stock_news \
						   WHERE date_id = %s"
			cur.execute(sql_select2,date_id)
			select_result2 = cur.fetchall()
			news_titles = []
			for s in select_result2:
				news_titles.append(s[0])
			re = gr.get_result(company_name,news_titles)
			if re != '###':
				predict = re[0]
				precise = round(re[1],4)
				print company_name
				print predict, precise
				# try:
				sql_update1 = "UPDATE stock_date \
							   SET predict = %s,confidence = %s \
							   WHERE id = %s"
				cur.execute(sql_update1,(predict,precise,date_id))
				db.commit()
				# except:
				# 	db.rollback()

def stock_recommend(date_time):
	#compute recommend rise
	sql_select3 = "SELECT stock_company.company_name,stock_date.confidence \
	               FROM stock_company,stock_date \
	               WHERE stock_company.id = stock_date.company_id and stock_date.date_time = %s and stock_date.predict = 1\
	               ORDER BY stock_date.confidence DESC"
	cur.execute(sql_select3,date_time)
	select_result3 = cur.fetchmany(10)
	if select_result3:
		recommend_stock = []
		recommend_confidence = []
		init = []
		for i in select_result3:
			recommend_stock.append(i[0])
			recommend_confidence.append(str(i[1]))
			init.append(str(0.0))
		recommend_stock = ','.join(recommend_stock)
		recommend_confidence = ','.join(recommend_confidence)
		init = ','.join(init)
		update_recommend_stock_confidence = "insert into stock_recommend(date_time, profit, random_profit, market_profit," \
											"recommend_stock, recommend_price, recommend_confidence, hushen_profit, random300_profit)" \
											"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
		try:
			cur.execute(update_recommend_stock_confidence,(date_time, 0.0, 0.0, 0.0, recommend_stock, init, recommend_confidence, 0.0, 0.0))
			db.commit()
		except:
			db.rollback()

now_date,last_date = get_date()
#更新预测涨跌幅方向和置信度
stock_predict(now_date)
print "update stock_date.predict stock_date.confidence"
#更新今日推荐公司和推荐置信度
stock_recommend(now_date)
print "update stock_recommend.recommend_stock and stock_recommend.recommend_confidence"
print "success!"

f = open("predict.txt",'w')
select_s_c="select recommend_stock,recommend_confidence from stock_recommend WHERE date_time=%s"
cur.execute(select_s_c, now_date)
name_confidence = cur.fetchall()[0]
name = name_confidence[0].split(',')
confidence = name_confidence[1].split(',')
# print name
# print confidence
code = []
for name_one in name:
	select_stockcode = "select stock_code from stock_company where company_name = %s"
	cur.execute(select_stockcode, name_one)
	code.append(cur.fetchall()[0][0])
f.write("\"笨笨\"推荐股票（"+time.strftime('%Y年%m月%d日', time.localtime(time.time()))+"，下午14：30）\n\n")
count = 1
for name_one,confidence_one,stock_code in zip(name, confidence, code):
	f.write("("+str(count)+")"+name_one+"（"+stock_code+"）"+"\t置信度:"+confidence_one+'\n')
	select_t_u = "select news_title,news_url,stock_code from stock_company,stock_date,stock_news where stock_company.id = " \
		 "stock_date.company_id and stock_date.id = stock_news.date_id and date_time = %s and company_name = %s"
	cur.execute(select_t_u, (now_date, name_one))
	title_url = cur.fetchall()
	for item in title_url:
		f.write(item[0]+'\n')
		f.write(item[1]+'\n')
	f.write('\n')
	count += 1
	# break
f.close()
db.close()


