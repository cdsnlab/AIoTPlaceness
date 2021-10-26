import json
import time
from datetime import datetime
import os
import pymongo
import subprocess
from pymongo import MongoClient

print("running \'python3 count_files.py\'")
subprocess.call(['python3', 'count_files.py'])
client = MongoClient('mongodb://localhost')

db = client.placeness

db.posts.drop()
db.images.drop()
db.posts.create_index([("loc", pymongo.GEO2D)])
db.posts.create_index([('post_shortcode', pymongo.ASCENDING)], unique=True)
db.posts.create_index([('timestamp',  pymongo.ASCENDING)])
db.images.create_index([("loc", pymongo.GEO2D)])
db.images.create_index([('image_shortcode', pymongo.ASCENDING)], unique=True)
db.images.create_index([('timestamp',  pymongo.ASCENDING)])


print(db.list_collection_names())
print(db.sample.find_one())

jpg_dict = {}
image_local_id = 0
with open("tot_jpg_list.txt") as rf:
	for line in rf:
		parent_dir, loc_dir, image_path = line.strip().split('/')
		loc_id = loc_dir
		if parent_dir not in jpg_dict:
			jpg_dict[parent_dir] = {}
		if loc_id not in jpg_dict[parent_dir]:
			jpg_dict[parent_dir][loc_id] = []
		jpg_dict[parent_dir][loc_id].append(image_path)

json_list = []
with open("tot_json_list.txt") as rf:
	for line in rf:
		json_list.append(line.strip())

#print(jpg_list[:5], json_list[:5])

for json_path in json_list:
	parent_dir, loc_dir, json_path = json_path.split('/')

	loc_id = loc_dir
	#print(parent_dir, loc_id, json_path)

	try:
		with open(parent_dir + '/%' + loc_id + '/' + json_path) as json_file:
			json_data = json.load(json_file)
	except:
		continue

	file_path = json_path[:-5]
	#print(parent_dir, loc_id, file_path)
	
	#print(str(json_data).encode('utf8'))
	#print(json_data['node']['shortcode'])
	shortcode = json_data['node']['shortcode']
	children = []
	image_count = 0
	image_list = []
	if json_data['node']['__typename'] == 'GraphImage':
		assert(os.path.isfile(parent_dir + '/%' + loc_id + '/' + file_path + '.jpg'))
		#children.append(item['node']['__typename'])
		children.append('')
		image_count = 1
		item = json_data
		image_list.append({
			'image_path': parent_dir + '/%25' + loc_id + '/' + file_path + '_' + str(idx) + '.jpg',
			'image_name': file_path + '.jpg',
			'image_id': item['node']['id'],
			'image_shortcode': item['node']['shortcode'],
			'dimensions': item['node']['dimensions'],
			"accessibility_caption": item['node']['accessibility_caption']
		})
		pass
	elif json_data['node']['__typename'] == 'GraphSidecar':
		if ('edge_sidecar_to_children' in json_data['node']):
			children_count = len(json_data['node']['edge_sidecar_to_children']['edges'])
			if children_count > 1:
				idx = 0
				for item in json_data['node']['edge_sidecar_to_children']['edges']:
					idx += 1
					#children.append(item['node']['__typename'])
					if item['node']['__typename'] == 'GraphImage':
						#print(parent_dir + '/%' + loc_id + '/' + file_path + '_' + str(idx) + '.jpg')
						children.append('_' + str(idx))
						image_list.append({
							'image_path': parent_dir + '/%25' + loc_id + '/' + file_path + '_' + str(idx) + '.jpg',
							'image_name': file_path + '_' + str(idx) + '.jpg',
							'image_id': item['node']['id'],
							'image_shortcode': item['node']['shortcode'],
							'dimensions': item['node']['dimensions'],
							"accessibility_caption": item['node']['accessibility_caption']
						})
						assert(os.path.isfile(parent_dir + '/%' + loc_id + '/' + file_path + '_' + str(idx) + '.jpg'))
						image_count += 1
					else:
						pass
				#print(image_list)
						
				#print(len(json_data['node']['edge_sidecar_to_children']['edges']))
			else:
				#print(parent_dir, loc_id, json_path)
				#print(len(json_data['node']['edge_sidecar_to_children']['edges']))
				break
		else:
			#print(parent_dir, loc_id, json_path)
			break
	elif json_data['node']['__typename'] == 'GraphVideo':
		#print(json_data['node']['__typename'], parent_dir, loc_id, json_path)
		continue

	lat = json_data['node']['location']['lat']
	lng = json_data['node']['location']['lng']

	post_caption = ''
	if len(json_data['node']['edge_media_to_caption']['edges']) > 0:
		post_caption = json_data['node']['edge_media_to_caption']['edges'][0]['node']['text']
	
	#print(parent_dir + '/%25' + loc_id + '/' + file_path, len(children))
	new_post = {
		'parent_dir': parent_dir,
		'loc_id': loc_id,
		'file_name': file_path,
		'file_path': parent_dir + '/%25' + loc_id + '/' + file_path,
		'post_shortcode': shortcode,
		'post_id': json_data['node']['id'],
		'children': children,
		'image_count': image_count,
		'loc': [lat, lng],
		'place_name': json_data['node']['location']['name'],
		'timestamp': json_data['node']['taken_at_timestamp'],
		'caption': post_caption
	}
	#print(datetime.fromtimestamp(json_data['node']['taken_at_timestamp']), file_path)
	#db.posts.insert_one(new_post)
	if not db.posts.find_one({'post_shortcode': new_post['post_shortcode']}):
		db.posts.insert_one(new_post)

	for image_item in image_list:
		new_image = {
			'image_path': image_item['image_path'],
			'image_name': image_item['image_name'],
			'image_id': image_item['image_id'],
			'image_shortcode': image_item['image_shortcode'],
			'dimensions': image_item['dimensions'],
			'accessibility_caption': image_item['accessibility_caption'],
			'post_shortcode': shortcode,
			'parent_dir': parent_dir,
			'loc_id': loc_id,
			'loc': [lat, lng],
			'timestamp': json_data['node']['taken_at_timestamp'],
			'image_local_id': image_local_id,
			'caption': post_caption
		}
		image_local_id += 1
		#print(new_image)
		if not db.images.find_one({'image_shortcode': new_image['image_shortcode']}):
			db.images.insert_one(new_image)



		
