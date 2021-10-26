import os, sys
import datetime
import subprocess
import instaloader
from time import sleep


def instagram_login():
	L = instaloader.Instaloader(
		download_videos=False,
		download_video_thumbnails=False, 
		download_comments=False,
		compress_json=False)
	USER = input("Enter Instagram USER: ")
	L.interactive_login(USER)
	return L


def input_crawling_intervals():
	date_entry = input('Enter a start date in YYYY-MM-DD format: ')
	year, month, day = map(int, date_entry.split('-'))
	date1 = datetime.datetime(year, month, day, 0, 0, 0)

	date_entry = input('Enter a end date in YYYY-MM-DD format (enter: all): ')
	date2 = None
	if date_entry != '':
		year, month, day = map(int, date_entry.split('-'))
		date2 = datetime.datetime(year, month, day+1, 0, 0, 0)

	print("Crawling images {} <= ~ < {}".format(date1, date2))
	return date1, date2
	

def crawl_by_location_id(L, loc_id, sleep_img_crawl, sleep_post_crawl, date1, date2):
	print("Crawling: {} ({}, {})".format(loc_id, sleep_img_crawl, sleep_post_crawl))
	for post in L.get_location_posts(loc_id):
		if post.date_local < date1:
			break
		if date1 <= post.date_local and (date2 == None or post.date_local < date2):
			print("[crawl] " + post.shortcode + ' ' + str(post.date_local))
			if os.path.isfile('./%{}/{}'.format(loc_id, post.shortcode)):
				print("  ... already crawled")
				continue
			with L.context.error_catcher('Download location {}'.format(loc_id)):
				downloaded = L.download_post(post, target='%' + loc_id)
				with open('./%{}/{}'.format(loc_id, post.shortcode), 'w') as fp:
					pass
			sleep(sleep_img_crawl)
		else:
			print("<skip> {} {}".format(post.shortcode, str(post.date_local)))
			sleep(sleep_post_crawl)


def crawling_trial(L, loc_id, t1, t2, date1, date2):
	if not os.path.isdir('./%' + loc_id):
		os.mkdir('./%' + loc_id)
	crawl_by_location_id(L, loc_id, t1, t2, date1, date2)
	return True
	try:
		crawl_by_location_id(L, loc_id, t1, t2, date1, date2)
	except:
		return False
	return True


if __name__ == "__main__":
	L = instagram_login()
	date1, date2 = input_crawling_intervals()

	loc_file_name = 'location_ids.txt'
	location_ids = []
	if os.path.isfile(loc_file_name):
		with open(loc_file_name) as fp:
			for line in fp:
				location_ids.append(line.strip())
	else:
		print("[{}] file not found.".format(loc_file_name))
		sys.exit()

	last_id = None
	for loc_id in location_ids:
		if loc_id[0] == '#': continue
		if os.path.isdir('./%'+loc_id):
			last_id = loc_id
			continue
		
		if last_id != None:
			for i in range(1, 11):
				print("Trial({}) for {}".format(i, last_id))
				if crawling_trial(L, last_id, i, i/10, date1, date2):
					break
				else:
					print("[Error 1] sleeping 2 mins...")
					sleep(120)
			last_id = None
			
		for i in range(1, 11):
			print("Trial({}) for {}".format(i, loc_id))
			if crawling_trial(L, loc_id, i, i/10, date1, date2):
				break
			else:
				print("[Error 2] sleeping 2 mins...")
				sleep(120)
		
